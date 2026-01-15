from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import re
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
import json
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


def _select_tests(tests: dict[str, Any], max_tests: int) -> tuple[list[Any], list[Any]]:
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
    return inputs, outputs


def _build_multi_test_harness(
    code: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: Optional[str],
    timeout_s: Optional[float],
) -> str:
    safe_timeout = timeout_s if timeout_s and timeout_s > 0 else None
    return f"""
import contextlib
import io
import json
import signal
import sys

_code = {repr(code)}
_inputs = {repr(inputs)}
_expected = {repr(outputs)}
_fn_name = {repr(fn_name)}
_timeout_s = {repr(safe_timeout)}

def _run_with_timeout(func, timeout_s):
    if not timeout_s:
        return True, func()
    def _handler(signum, frame):
        raise TimeoutError("Execution timed out")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return True, func()
    except Exception:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

def _normalize_expected(val):
    expected_clean = val.strip()
    if expected_clean.startswith('"') and expected_clean.endswith('"'):
        expected_clean = expected_clean[1:-1]
    return expected_clean

def _compare(actual, expected_clean):
    if actual.lower() in ("true", "false") and expected_clean.lower() in ("true", "false"):
        return actual.lower() == expected_clean.lower()
    return actual == expected_clean

def _run_solution_case():
    global_ns = {{"__name__": "__main__"}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(_code, global_ns)
    sol_cls = global_ns.get("Solution")
    if sol_cls is None:
        return None
    try:
        sol = sol_cls()
    except Exception:
        return None
    passed = 0
    total = len(_inputs)
    for input_expr, expected in zip(_inputs, _expected):
        def _call():
            local = {{}}
            exec("_input = " + input_expr, {{}}, local)
            _input = local["_input"]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _result = getattr(sol, _fn_name)(_input)
                print(_result)
            return buf.getvalue().strip()
        ok, actual = _run_with_timeout(_call, _timeout_s)
        expected_clean = _normalize_expected(expected)
        if ok and _compare(actual, expected_clean):
            passed += 1
    return {{"passed": passed, "total": total}}

def _run_script_case():
    compiled = compile(_code, "<solution>", "exec")
    passed = 0
    total = len(_inputs)
    for inp, expected in zip(_inputs, _expected):
        def _call():
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            old_stdin = sys.stdin
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdin = io.StringIO(inp)
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                exec(compiled, {{"__name__": "__main__"}})
                return stdout_buffer.getvalue().strip()
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        ok, actual = _run_with_timeout(_call, _timeout_s)
        expected_clean = _normalize_expected(expected)
        if ok and _compare(actual, expected_clean):
            passed += 1
    return {{"passed": passed, "total": total}}

_result = None
if _fn_name and "class Solution" in _code:
    _result = _run_solution_case()
if _result is None:
    _result = _run_script_case()
print(json.dumps(_result))
"""


def _parse_harness_output(stdout: str) -> float:
    if not stdout:
        return 0.0
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
        total = int(payload.get("total", 0))
        if total <= 0:
            return 0.0
        passed = int(payload.get("passed", 0))
        return passed / total
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0


def _evaluate_code(code: str, tests: dict[str, Any], max_tests: int, timeout_s: Optional[float]) -> float:
    inputs, outputs = _select_tests(tests, max_tests)
    normalized_inputs = [_normalize_io(inp) for inp in inputs]
    normalized_outputs = [_normalize_io(expected) for expected in outputs]

    harness = _build_multi_test_harness(
        code=code,
        inputs=[str(inp) if inp is not None else "" for inp in normalized_inputs],
        outputs=[str(exp) if exp is not None else "" for exp in normalized_outputs],
        fn_name=tests.get("fn_name", None),
        timeout_s=timeout_s,
    )
    success, stdout, _ = _exec_code(harness, None, None)
    if not success:
        return 0.0
    return _parse_harness_output(stdout)


def evaluate_task(task: EvalTask) -> EvalResult:
    if _extract_interact_code(task.response):
        return EvalResult(reward=0.0, terminated=True, truncated=True)

    answer_code = _extract_answer_code(task.response)
    if not answer_code:
        return EvalResult(reward=0.0, terminated=True, truncated=True)

    reward = _evaluate_code(answer_code, task.tests, task.max_tests, task.timeout_s)
    return EvalResult(reward=reward, terminated=True, truncated=False)


def _evaluate_task_batch(tasks: list[EvalTask]) -> list[EvalResult]:
    return [evaluate_task(task) for task in tasks]


def evaluate_tasks(
    tasks: list[EvalTask],
    max_workers: int,
    batch_size: int,
    show_progress: bool = False,
) -> list[EvalResult]:
    if not tasks:
        return []

    if batch_size <= 0:
        batch_size = 1

    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        results_iter = executor.map(_evaluate_task_batch, batches)
        if not show_progress:
            return [result for batch in results_iter for result in batch]

        from tqdm import tqdm

        results: list[EvalResult] = []
        progress = tqdm(total=len(tasks), desc="Evaluating")
        for batch in results_iter:
            results.extend(batch)
            progress.update(len(batch))
        progress.close()
        return results
