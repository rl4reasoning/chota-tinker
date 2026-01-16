"""
Custom GEM environment for INTELLECT-3-RL dataset with multi-turn Python REPL.
"""

import atexit
import contextlib
import io
import json
import os
import re
import random
import signal
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, Tuple
from datasets import Dataset, load_dataset
from gem.core import Env
from gem.utils.sandbox import run_python
from utils.fast_eval import EvalTask, evaluate_task, evaluate_tasks


@contextlib.contextmanager
def _time_limit(seconds: Optional[float]):
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError("Execution timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _exec_interaction_code(code: str, timeout_s: Optional[float]) -> tuple[bool, str, str]:
    """Execute interaction code in-process (no sandbox)."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    original_os_exit = os._exit

    def _safe_os_exit(exit_code: int = 0):
        raise SystemExit(exit_code)

    os._exit = _safe_os_exit
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            with _time_limit(timeout_s):
                exec(code, {"__name__": "__main__"})
        return True, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    except SystemExit as exc:
        if exc.code in (None, 0):
            return True, stdout_buffer.getvalue(), stderr_buffer.getvalue()
        stderr_buffer.write(f"SystemExit: {exc}\n")
        return False, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    except Exception:
        tb_lines = traceback.format_exc().splitlines()
        filtered_lines: list[str] = []
        skip_next = False
        for line in tb_lines:
            if skip_next:
                if line.startswith("    ") and not line.lstrip().startswith("File "):
                    skip_next = False
                    continue
                skip_next = False
            if "intellect_env.py" in line:
                skip_next = True
                continue
            filtered_lines.append(line)
        stderr_buffer.write("\n".join(filtered_lines) + "\n")
        return False, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    finally:
        os._exit = original_os_exit


_INTERACTION_EXECUTOR: Optional[ProcessPoolExecutor] = None
_INTERACTION_EXECUTOR_WORKERS: Optional[int] = None
_EVAL_EXECUTOR: Optional[ProcessPoolExecutor] = None
_EVAL_EXECUTOR_WORKERS: Optional[int] = None


def _shutdown_persistent_executors() -> None:
    global _INTERACTION_EXECUTOR, _INTERACTION_EXECUTOR_WORKERS
    global _EVAL_EXECUTOR, _EVAL_EXECUTOR_WORKERS
    if _INTERACTION_EXECUTOR is not None:
        _INTERACTION_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _INTERACTION_EXECUTOR = None
        _INTERACTION_EXECUTOR_WORKERS = None
    if _EVAL_EXECUTOR is not None:
        _EVAL_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EVAL_EXECUTOR = None
        _EVAL_EXECUTOR_WORKERS = None


atexit.register(_shutdown_persistent_executors)


def _get_persistent_executor(kind: str, max_workers: int) -> ProcessPoolExecutor:
    global _INTERACTION_EXECUTOR, _INTERACTION_EXECUTOR_WORKERS
    global _EVAL_EXECUTOR, _EVAL_EXECUTOR_WORKERS

    if kind == "interaction":
        if _INTERACTION_EXECUTOR is not None and _INTERACTION_EXECUTOR_WORKERS == max_workers:
            return _INTERACTION_EXECUTOR
        if _INTERACTION_EXECUTOR is not None:
            _INTERACTION_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        mp_context = mp.get_context("spawn")
        _INTERACTION_EXECUTOR = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        )
        _INTERACTION_EXECUTOR_WORKERS = max_workers
        return _INTERACTION_EXECUTOR

    if kind == "eval":
        if _EVAL_EXECUTOR is not None and _EVAL_EXECUTOR_WORKERS == max_workers:
            return _EVAL_EXECUTOR
        if _EVAL_EXECUTOR is not None:
            _EVAL_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        mp_context = mp.get_context("spawn")
        _EVAL_EXECUTOR = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        )
        _EVAL_EXECUTOR_WORKERS = max_workers
        return _EVAL_EXECUTOR

    raise ValueError(f"Unknown executor kind: {kind}")


class IntellectCodeEnv(Env):
    """
    Multi-turn code environment using INTELLECT-3-RL dataset.
    
    - LLM submits code in <interact>...</interact> -> executed, output returned in <output>...</output>
    - LLM submits final answer in ```python``` -> evaluated against tests
    """

    def __init__(
        self,
        system_prompt: str = "",
        config: str = "code",
        split: str = "train",
        max_turns: int = 5,
        max_tests: int = 12,
        interaction_timeout_s: Optional[float] = None,
        sandbox_type: str = "none",
        seed: int = 0,
        dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
        problem_index: Optional[int] = None,
        dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tests = max_tests
        self.interaction_timeout_s = interaction_timeout_s
        self.sandbox_type = sandbox_type
        self.seed = seed
        self.dataset_name = dataset_name
        self.problem_index = problem_index
        
        # Use pre-loaded dataset if provided, otherwise load it
        if dataset is not None:
            self.dataset = dataset
        elif dataset_name.startswith("bicycleman15/"):
            self.dataset = load_dataset(dataset_name, split=split)
        else:
            self.dataset = load_dataset(dataset_name, config, split=split)
        
        # If problem_index specified, use that; otherwise iterate through dataset
        if problem_index is not None:
            self._use_fixed_index = True
            if problem_index < 0 or problem_index >= len(self.dataset):
                raise ValueError(f"problem_index {problem_index} out of range [0, {len(self.dataset)})")
        else:
            self._use_fixed_index = False
            self.dataset_iter = iter(self.dataset)
        
        self.current_turn = 0
        self.question = ""
        self.tests = {}
        self.history = []
        self.has_interacted = False  # Track if model has interacted at least once

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        
        if self._use_fixed_index:
            data = self.dataset[self.problem_index]
        else:
            data = next(self.dataset_iter)
        self.question = data["question"]
        
        info = json.loads(data["info"])
        self.tests = json.loads(info["tests"])
        self.fn_name = self.tests.get("fn_name", None)
        
        self.current_turn = 0
        self.history = []
        self.has_interacted = False
        
        obs = self._build_observation(self.question)
        return obs, {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        self.current_turn += 1
        
        # check for interactive python code FIRST (takes priority over final answer)
        python_code = self._extract_interact_code(action)
        if python_code:
            return self._handle_interact(python_code)
        
        # check for final answer (only if no <interact> tag)
        answer_code = self._extract_answer_code(action)
        if answer_code:
            return self._handle_final(answer_code)
        
        # no valid code found
        return self._handle_invalid()

    def _build_observation(self, content: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{content}"
        return content

    @staticmethod
    def _extract_interact_code(text: str) -> Optional[str]:
        # match <interact>...</interact> for interactive execution
        pattern = r"<interact>(.*?)</interact>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            content = matches[-1].strip()
            # handle case where code is wrapped in ```python``` inside <interact>
            code_pattern = r"```(?:python)?\n?(.*?)```"
            code_matches = re.findall(code_pattern, content, re.DOTALL | re.IGNORECASE)
            if code_matches:
                return code_matches[-1].strip()
            return content
        return None

    @staticmethod
    def _extract_answer_code(text: str) -> Optional[str]:
        # match ```python for final answer
        pattern = r"```python\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    def _handle_interact(self, python_code: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        try:
            success, stdout, stderr = _exec_interaction_code(
                python_code, self.interaction_timeout_s
            )
            if success:
                output = stdout if stdout else "(no output during interaction -- did you forget to print? did you enclose the code correctly in <interact></interact>?)"
            else:
                output = f"Error:\n{stderr}" if stderr else "Error: execution failed."
        except Exception as e:
            output = f"Error: failed to execute code - {str(e)}"

        self.history.append({"code": python_code, "output": output})
        self.has_interacted = True  # Mark that interaction occurred
        obs = f"<output>\n{output}</output>"

        if self.current_turn >= self.max_turns:
            return obs, 0.0, True, True, {"truncated": True}

        return obs, 0.0, False, False, {}

    def _handle_requires_interaction(self) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        obs = "<output>You must interact at least once before submitting your final answer. Use <interact> ... your code here ... </interact> to test your code first. Remember to pass in the inputs yourself.</output>"
        if self.current_turn >= self.max_turns:
            return obs, 0.0, True, True, {"truncated": True}
        return obs, 0.0, False, False, {}

    def _handle_invalid(self) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        obs = "<output>No valid code block found. Use <interact></interact> or ```python```.</output>"
        if self.current_turn >= self.max_turns:
            return obs, 0.0, True, True, {"truncated": True}
        return obs, 0.0, False, False, {}

    def _handle_final(self, answer_code: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        # Require at least one interaction before final answer
        if not self.has_interacted:
            return self._handle_requires_interaction()

        # Enforce Solution-only when fn_name exists (align with fast_eval)
        if self.fn_name and "class Solution" not in answer_code:
            return "", 0.0, True, True, {"final": True, "invalidated": True}

        reward = self._evaluate(answer_code)
        return "", reward, True, False, {"final": True}

    def build_eval_task(self, action: str, timeout_s: Optional[float]) -> Optional[EvalTask]:
        if self._extract_interact_code(action):
            return None
        if not self._extract_answer_code(action):
            return None
        if not self.has_interacted:
            return None
        return EvalTask(
            response=action,
            tests=self.tests,
            max_tests=self.max_tests,
            timeout_s=timeout_s,
            require_solution_class=True,
        )

    def _evaluate(self, code: str) -> float:
        tests = self.tests
        
        total_tests = len(tests["inputs"])
        if total_tests > self.max_tests:
            tests = {
                "inputs": list(tests["inputs"][:self.max_tests]),
                "outputs": list(tests["outputs"][:self.max_tests]),
            }
        
        passed = 0
        for inp, expected in zip(tests["inputs"], tests["outputs"]):
            if isinstance(inp, list):
                inp = "\n".join(map(str, inp))
            if isinstance(expected, list):
                expected = "\n".join(map(str, expected))
            
            # wrap code with test harness for LeetCode-style Solution class
            if self.fn_name and "class Solution" in code:
                harness = f"_input = {inp}\n_sol = Solution()\n_result = _sol.{self.fn_name}(_input)\nprint(_result)"
                wrapped_code = code + "\n\n" + harness
                success, stdout, _ = run_python(wrapped_code, self.sandbox_type)
            else:
                success, stdout, _ = run_python(code, self.sandbox_type, stdin=inp)
            
            # strip outer quotes from expected if present (LeetCode format)
            expected_clean = expected.strip()
            if expected_clean.startswith('"') and expected_clean.endswith('"'):
                expected_clean = expected_clean[1:-1]
            
            # Normalize output for comparison
            actual = stdout.strip()
            
            # Handle boolean case-insensitivity (Python "True"/"False" vs JSON "true"/"false")
            if actual.lower() in ("true", "false") and expected_clean.lower() in ("true", "false"):
                if success and actual.lower() == expected_clean.lower():
                    passed += 1
            elif success and actual == expected_clean:
                passed += 1
        
        return passed / len(tests["inputs"])

    def sample_random_action(self) -> str:
        if random.random() < 0.8:
            return "<interact>\nprint('hello')\n</interact>"
        else:
            return "```python\nprint('hello')\n```"


def _run_python_batch_item(item: tuple[str, Optional[float]]) -> tuple[bool, str, str]:
    """Helper function to run interaction code in a worker process (must be at module level for pickling)."""
    code, timeout_s = item
    return _exec_interaction_code(code, timeout_s)


def _run_python_batch(
    codes_and_timeouts: list[tuple[str, Optional[float]]],
    max_workers: int,
    batch_size: int,
    executor: Optional[ProcessPoolExecutor] = None,
) -> list[tuple[bool, str, str]]:
    """Batch execute interaction code in parallel (no sandbox)."""
    if not codes_and_timeouts:
        return []
    
    if executor is not None:
        return list(executor.map(_run_python_batch_item, codes_and_timeouts))

    if max_workers <= 1 and batch_size <= 1:
        return [_exec_interaction_code(code, timeout_s) for code, timeout_s in codes_and_timeouts]
    
    if batch_size <= 0:
        batch_size = 1
    
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        results = list(executor.map(_run_python_batch_item, codes_and_sandboxes))
    
    return results


def step_batch(
    envs: list[IntellectCodeEnv],
    actions: list[str],
    eval_workers: int,
    eval_batch_size: int,
    eval_timeout_s: Optional[float],
    show_progress: bool = False,
    use_persistent_pool: bool = True,
    interact_timeout_s: Optional[float] = None,
) -> list[Tuple[str, float, bool, bool, dict[str, Any]]]:
    if len(envs) != len(actions):
        raise ValueError("envs and actions must be the same length")

    results: list[Optional[Tuple[str, float, bool, bool, dict[str, Any]]]] = [None] * len(envs)
    eval_tasks: list[EvalTask] = []
    eval_indices: list[int] = []
    
    # Collect interaction tasks for parallel execution
    interact_tasks: list[tuple[int, str]] = []
    interact_indices: list[int] = []

    for idx, (env, action) in enumerate(zip(envs, actions)):
        env.current_turn += 1

        python_code = env._extract_interact_code(action)
        if python_code:
            interact_tasks.append((idx, python_code))
            interact_indices.append(idx)
            continue

        answer_code = env._extract_answer_code(action)
        if answer_code:
            if not env.has_interacted:
                results[idx] = env._handle_requires_interaction()
                continue
            eval_tasks.append(
                EvalTask(
                    response=action,
                    tests=env.tests,
                    max_tests=env.max_tests,
                    timeout_s=eval_timeout_s,
                    require_solution_class=True,
                )
            )
            eval_indices.append(idx)
            continue

        results[idx] = env._handle_invalid()

    # Batch execute interactions in parallel
    if interact_timeout_s is None:
        interact_timeout_s = eval_timeout_s

    if interact_tasks:
        codes_and_timeouts = [
            (python_code, interact_timeout_s)
            for idx, python_code in interact_tasks
        ]
        if len(codes_and_timeouts) == 1:
            interact_results = [_exec_interaction_code(codes_and_timeouts[0][0], codes_and_timeouts[0][1])]
        else:
            interaction_executor = None
            if use_persistent_pool and eval_workers > 1:
                interaction_executor = _get_persistent_executor("interaction", eval_workers)
            interact_results = _run_python_batch(
                codes_and_timeouts,
                max_workers=eval_workers,
                batch_size=eval_batch_size,
                executor=interaction_executor,
            )
        
        for (idx, python_code), (success, stdout, stderr) in zip(interact_tasks, interact_results):
            env = envs[idx]
            if success:
                output = stdout if stdout else "(no output during interaction -- did you forget to print? did you enclose the code correctly in <interact></interact>?)"
            else:
                output = f"Error:\n{stderr}" if stderr else "Error: execution failed."
            
            env.history.append({"code": python_code, "output": output})
            env.has_interacted = True
            obs = f"<output>\n{output}</output>"
            
            if env.current_turn >= env.max_turns:
                results[idx] = (obs, 0.0, True, True, {"truncated": True})
            else:
                results[idx] = (obs, 0.0, False, False, {})

    if eval_tasks:
        if len(eval_tasks) == 1:
            eval_results = [evaluate_task(eval_tasks[0])]
        elif eval_workers <= 1 and eval_batch_size <= 1:
            eval_results = [evaluate_task(task) for task in eval_tasks]
        else:
            eval_executor = None
            if use_persistent_pool and eval_workers > 1:
                eval_executor = _get_persistent_executor("eval", eval_workers)
            eval_results = evaluate_tasks(
                eval_tasks,
                max_workers=eval_workers,
                batch_size=eval_batch_size,
                show_progress=show_progress,
                executor=eval_executor,
            )
        for idx, eval_result in zip(eval_indices, eval_results):
            results[idx] = ("", eval_result.reward, True, eval_result.truncated, {"final": True})

    for result in results:
        if result is None:
            raise RuntimeError("Missing batch step result")

    return results

