from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import re
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EvalTask:
    response: str
    tests: dict[str, Any]
    max_tests: int
    timeout_s: Optional[float]


@dataclass(frozen=True)
class EvalResult:
    reward: float
    terminated: bool
    truncated: bool


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


def _exec_code(code: str, stdin: Optional[str], timeout_s: Optional[float]) -> tuple[bool, str, str]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    old_stdin = sys.stdin
    try:
        if stdin is None:
            sys.stdin = io.StringIO("")
        else:
            sys.stdin = io.StringIO(stdin)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            with _time_limit(timeout_s):
                exec(code, {"__name__": "__main__"})
        return True, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    except SystemExit as exc:
        if exc.code in (None, 0):
            return True, stdout_buffer.getvalue(), stderr_buffer.getvalue()
        stderr_buffer.write(f"SystemExit: {exc}\n")
        return False, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    except TimeoutError as exc:
        stderr_buffer.write(f"{exc}\n")
        return False, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    except Exception:
        traceback.print_exc(file=stderr_buffer)
        return False, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    finally:
        sys.stdin = old_stdin


def _normalize_io(value: Any) -> Any:
    if isinstance(value, list):
        return "\n".join(map(str, value))
    return value


def _extract_interact_code(text: str) -> Optional[str]:
    pattern = r"<interact>(.*?)</interact>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        content = matches[-1].strip()
        code_pattern = r"```(?:python)?\n?(.*?)```"
        code_matches = re.findall(code_pattern, content, re.DOTALL | re.IGNORECASE)
        if code_matches:
            return code_matches[-1].strip()
        return content
    return None


def _extract_answer_code(text: str) -> Optional[str]:
    pattern = r"```python\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def _evaluate_code(code: str, tests: dict[str, Any], max_tests: int, timeout_s: Optional[float]) -> float:
    total_tests = len(tests["inputs"])
    if total_tests > max_tests:
        indices = sorted(
            range(total_tests),
            key=lambda i: len(tests["inputs"][i]),
            reverse=True,
        )[:max_tests]
        inputs = [tests["inputs"][i] for i in indices]
        outputs = [tests["outputs"][i] for i in indices]
    else:
        inputs = tests["inputs"]
        outputs = tests["outputs"]

    fn_name = tests.get("fn_name", None)
    use_solution = bool(fn_name and "class Solution" in code)

    passed = 0
    for inp, expected in zip(inputs, outputs):
        inp = _normalize_io(inp)
        expected = _normalize_io(expected)

        if use_solution:
            harness = (
                f"_input = {inp}\n"
                f"_sol = Solution()\n"
                f"_result = _sol.{fn_name}(_input)\n"
                f"print(_result)"
            )
            wrapped_code = f"{code}\n\n{harness}"
            success, stdout, _ = _exec_code(wrapped_code, None, timeout_s)
        else:
            stdin_value = str(inp) if inp is not None else ""
            success, stdout, _ = _exec_code(code, stdin_value, timeout_s)

        expected_clean = expected.strip()
        if expected_clean.startswith('"') and expected_clean.endswith('"'):
            expected_clean = expected_clean[1:-1]

        actual = stdout.strip()
        if actual.lower() in ("true", "false") and expected_clean.lower() in ("true", "false"):
            if success and actual.lower() == expected_clean.lower():
                passed += 1
        elif success and actual == expected_clean:
            passed += 1

    return passed / len(inputs)


def evaluate_task(task: EvalTask) -> EvalResult:
    if _extract_interact_code(task.response):
        return EvalResult(reward=0.0, terminated=True, truncated=True)

    answer_code = _extract_answer_code(task.response)
    if not answer_code:
        return EvalResult(reward=0.0, terminated=True, truncated=True)

    reward = _evaluate_code(answer_code, task.tests, task.max_tests, task.timeout_s)
    return EvalResult(reward=reward, terminated=True, truncated=False)


def evaluate_tasks(
    tasks: list[EvalTask],
    max_workers: int,
    show_progress: bool = False,
) -> list[EvalResult]:
    if not tasks:
        return []

    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        results_iter = executor.map(evaluate_task, tasks)
        if show_progress:
            from tqdm import tqdm

            results_iter = tqdm(results_iter, total=len(tasks), desc="Evaluating")
        return list(results_iter)
