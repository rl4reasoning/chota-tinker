from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import re
import resource
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
import json
from dataclasses import dataclass
from typing import Any, Optional

from code_env.code_env.utils.deepcoder_utils import (
    extract_code_from_model,
    BASE_IMPORTS,
    process_input_output,
)


@dataclass(frozen=True)
class EvalTask:
    response: str
    tests: dict[str, Any]
    max_tests: int
    timeout_s: Optional[float]
    max_timeout_records: int = 0
    require_solution_class: bool = True


@dataclass(frozen=True)
class EvalResult:
    reward: float
    terminated: bool
    truncated: bool
    timeout_count: int = 0
    timeout_indices: tuple[int, ...] = ()
    invalidated: bool = False


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


def _exec_code_subprocess(code: str, stdin: Optional[str], timeout_s: Optional[float]) -> tuple[bool, str, str]:
    """Execute code in a subprocess with hard timeout (SIGKILL).
    
    Unlike _exec_code which uses SIGALRM, this can interrupt C extensions
    like itertools.permutations that don't release the GIL.
    """
    import subprocess
    import tempfile
    import os
    
    # Memory limit: 10GB virtual memory (matches code_env sandbox)
    MEMORY_LIMIT_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
    
    def _set_memory_limit():
        """Set memory limit for subprocess (Unix only)."""
        try:
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except (ValueError, OSError):
            pass  # Ignore if setting limit fails (e.g., on some systems)
    
    # Write code to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            input=stdin or "",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_set_memory_limit,
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out (subprocess killed)\n"
    except Exception as e:
        return False, "", f"Subprocess error: {e}\n"
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


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
    """Extract answer code from markdown code blocks using code_env utility."""
    code = extract_code_from_model(text)
    return code if code else None


def _select_tests(tests: dict[str, Any], max_tests: int) -> tuple[list[Any], list[Any]]:
    total_tests = len(tests["inputs"])
    if total_tests > max_tests:
        inputs = list(tests["inputs"][:max_tests])
        outputs = list(tests["outputs"][:max_tests])
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
    timeout_record_limit: int,
) -> str:
    safe_timeout = timeout_s if timeout_s and timeout_s > 0 else None
    safe_timeout_records = max(timeout_record_limit, 0)
    # Prepend BASE_IMPORTS to the code for better compatibility
    code_with_imports = BASE_IMPORTS + "\n" + code
    return f"""
import contextlib
import io
import json
import os
import signal
import sys
from decimal import Decimal, InvalidOperation

_code = {repr(code_with_imports)}
_inputs = {repr(inputs)}
_expected = {repr(outputs)}
_fn_name = {repr(fn_name)}
_timeout_s = {repr(safe_timeout)}
_timeout_record_limit = {repr(safe_timeout_records)}

_original_os_exit = os._exit
def _safe_os_exit(code=0):
    raise SystemExit(code)
os._exit = _safe_os_exit

class _TimeoutException(Exception):
    pass

def _run_with_timeout(func, timeout_s):
    if not timeout_s:
        return True, func(), False
    def _handler(signum, frame):
        raise _TimeoutException("Execution timed out")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return True, func(), False
    except _TimeoutException:
        return False, None, True
    except Exception:
        return False, None, False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

def _normalize_expected(val):
    expected_clean = val.strip()
    if expected_clean.startswith('"') and expected_clean.endswith('"'):
        expected_clean = expected_clean[1:-1]
    return expected_clean

# Tiered comparison strategies from code_env/deepcoder_utils
def _compare_trimmed_strings(a, b):
    return a.strip() == b.strip()

def _split_lines(value):
    return [line.strip() for line in value.strip().splitlines() if line.strip()]

def _compare_linewise(a, b):
    return _split_lines(a) == _split_lines(b)

def _tokenise(value):
    return [line.split() for line in _split_lines(value)]

def _compare_tokenwise(a, b):
    return _tokenise(a) == _tokenise(b)

def _flatten(tokens):
    flattened = []
    for line in tokens:
        flattened.extend(line)
    return flattened

def _to_decimals(tokens):
    decimals = []
    for token in tokens:
        try:
            decimals.append(Decimal(token))
        except (InvalidOperation, ValueError):
            return None
    return decimals

def _compare_numeric_tokens(a, b, tolerance=1e-3):
    tokens_a = _flatten(_tokenise(a))
    tokens_b = _flatten(_tokenise(b))
    if len(tokens_a) != len(tokens_b) or not tokens_a:
        return False
    decimals_a = _to_decimals(tokens_a)
    decimals_b = _to_decimals(tokens_b)
    if decimals_a is None or decimals_b is None:
        return False
    decimal_tol = Decimal(tolerance)
    for left, right in zip(decimals_a, decimals_b):
        if abs(left - right) > decimal_tol:
            return False
    return True

def _compare(actual, expected_clean):
    # Tiered comparison: try multiple strategies
    strategies = [
        lambda a, b: _compare_trimmed_strings(a, b),
        lambda a, b: _compare_linewise(a, b),
        lambda a, b: _compare_tokenwise(a, b),
        lambda a, b: _compare_numeric_tokens(a, b, 1e-3),
    ]
    for strategy in strategies:
        try:
            if strategy(actual, expected_clean):
                return True
        except Exception:
            continue
    # Also check boolean case-insensitivity
    if actual.lower() in ("true", "false") and expected_clean.lower() in ("true", "false"):
        return actual.lower() == expected_clean.lower()
    return False

def _parse_input_args(input_str):
    \"\"\"Parse stdin-format input string into function arguments (code_env style).
    
    Input format: newline-separated values, each line is a Python literal.
    Example: '[2, 2, 1]\\n4' -> [[2, 2, 1], 4]
    \"\"\"
    return list(map(eval, input_str.split("\\n")))

def _compare_func_result(exec_outputs, expected_str):
    \"\"\"Compare function call result with expected output (code_env parity).
    
    Handles tuple/list conversion and fallback checks.
    \"\"\"
    try:
        test_case_outputs = json.loads(expected_str)
    except (json.JSONDecodeError, TypeError):
        # Fall back to string comparison if not valid JSON
        return str(exec_outputs) == expected_str.strip()
    
    # Convert tuple to list
    if isinstance(exec_outputs, tuple):
        exec_outputs = list(exec_outputs)
    
    # Direct value comparison
    tmp_result = exec_outputs == test_case_outputs
    
    # Fallback: check against first element if expected is wrapped in list
    if not tmp_result and isinstance(test_case_outputs, list) and len(test_case_outputs) > 0:
        tmp_result = exec_outputs == test_case_outputs[0]
    
    # Handle nested tuples in output
    if not tmp_result:
        try:
            if isinstance(exec_outputs, list) and len(exec_outputs) > 0 and isinstance(exec_outputs[0], tuple):
                exec_outputs_converted = [list(x) for x in exec_outputs]
                tmp_result = exec_outputs_converted == test_case_outputs
                if not tmp_result and isinstance(test_case_outputs, list) and len(test_case_outputs) > 0:
                    tmp_result = exec_outputs_converted == test_case_outputs[0]
        except (TypeError, IndexError):
            pass
    
    return tmp_result

def _run_solution_case():
    # Previously we used "__main__" as the eval harness namespace, but this caused
    # `if __name__ == "__main__":` blocks to execute during class extraction. Models may
    # generate main blocks alongside Solution classes (especially gpt-oss), and these can crash
    # when run with empty stdin. Using "__solution__" prevents the main block from running
    # while still allowing the Solution class to be extracted.
    global_ns = {{"__name__": "__solution__"}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(_code, global_ns)
    
    # Support both Solution class and standalone function (code_env parity)
    sol_cls = global_ns.get("Solution")
    standalone_fn = global_ns.get(_fn_name) if _fn_name else None
    
    if sol_cls is None and standalone_fn is None:
        return None
    
    passed = 0
    total = len(_inputs)
    timeout_count = 0
    timeout_indices = []
    for idx, (input_str, expected) in enumerate(zip(_inputs, _expected)):
        def _call(inp=input_str):
            # Parse input string into arguments
            args = _parse_input_args(inp)
            try:
                if sol_cls is not None:
                    sol = sol_cls()
                    result = getattr(sol, _fn_name)(*args)
                else:
                    result = standalone_fn(*args)
                return {{"success": True, "result": result}}
            except SystemExit:
                return {{"success": False, "error": "SystemExit"}}
            except Exception as e:
                return {{"success": False, "error": repr(e)}}
        
        ok, result_data, timed_out = _run_with_timeout(_call, _timeout_s)
        if timed_out:
            timeout_count += 1
            if len(timeout_indices) < _timeout_record_limit:
                timeout_indices.append(idx)
            continue
        
        if ok and isinstance(result_data, dict) and result_data.get("success", False):
            exec_outputs = result_data["result"]
            if _compare_func_result(exec_outputs, expected):
                passed += 1
    
    return {{"passed": passed, "total": total, "timeouts": timeout_count, "timeout_indices": timeout_indices}}

def _run_script_case():
    compiled = compile(_code, "<solution>", "exec")
    passed = 0
    total = len(_inputs)
    timeout_count = 0
    timeout_indices = []
    for idx, (inp, expected) in enumerate(zip(_inputs, _expected)):
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
                try:
                    exec(compiled, {{"__name__": "__main__"}})
                except SystemExit:
                    pass
                return stdout_buffer.getvalue().strip()
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        ok, actual, timed_out = _run_with_timeout(_call, _timeout_s)
        if timed_out:
            timeout_count += 1
            if len(timeout_indices) < _timeout_record_limit:
                timeout_indices.append(idx)
        expected_clean = _normalize_expected(expected)
        if ok and _compare(actual, expected_clean):
            passed += 1
    return {{"passed": passed, "total": total, "timeouts": timeout_count, "timeout_indices": timeout_indices}}

_result = None
# Auto-detect: if fn_name exists, try Solution class or standalone function
# Otherwise, fall back to stdin-based execution
if _fn_name:
    _result = _run_solution_case()
if _result is None:
    # No fn_name or no Solution class/function, use stdin-based execution
    _result = _run_script_case()
print(json.dumps(_result))
"""


def _parse_harness_output(stdout: str) -> tuple[float, int, tuple[int, ...]]:
    if not stdout:
        return 0.0, 0, ()
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
        total = int(payload.get("total", 0))
        timeout_count = int(payload.get("timeouts", 0) or 0)
        raw_indices = payload.get("timeout_indices") or []
        timeout_indices: list[int] = []
        for idx in raw_indices:
            try:
                timeout_indices.append(int(idx))
            except (TypeError, ValueError):
                continue
        timeout_indices_tuple = tuple(timeout_indices)
        if total <= 0:
            return 0.0, timeout_count, timeout_indices_tuple
        passed = int(payload.get("passed", 0))
        return passed / total, timeout_count, timeout_indices_tuple
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0, 0, ()


def _evaluate_code(
    code: str,
    tests: dict[str, Any],
    max_tests: int,
    timeout_s: Optional[float],
    timeout_record_limit: int,
    require_solution_class: bool,
) -> tuple[float, int, tuple[int, ...]]:
    inputs, outputs = _select_tests(tests, max_tests)
    
    # Apply process_input_output for dict key normalization (code_env parity)
    # This converts JSON string keys back to integers for dict inputs/outputs
    processed_pairs = [process_input_output(inp, out) for inp, out in zip(inputs, outputs)]
    inputs = [p[0] for p in processed_pairs]
    outputs = [p[1] for p in processed_pairs]
    
    # Normalize inputs/outputs to strings (for both Solution class and stdin execution)
    normalized_inputs = [str(_normalize_io(inp)) if inp is not None else "" for inp in inputs]
    normalized_outputs = [str(_normalize_io(expected)) if expected is not None else "" for expected in outputs]

    harness = _build_multi_test_harness(
        code=code,
        inputs=normalized_inputs,
        outputs=normalized_outputs,
        fn_name=tests.get("fn_name", None),
        timeout_s=timeout_s,
        timeout_record_limit=timeout_record_limit,
    )
    # Calculate overall harness timeout: per-test timeout * num_tests + buffer for setup
    # Use subprocess execution which can kill C extensions that don't release GIL
    # (e.g., infinite loops in itertools.permutations)
    num_tests = len(inputs)
    if timeout_s and timeout_s > 0:
        # Give each test its timeout plus 5s buffer for compilation/setup
        overall_timeout = timeout_s * num_tests + 5.0
    else:
        # Default to 60s if no per-test timeout specified
        overall_timeout = 60.0
    success, stdout, _ = _exec_code_subprocess(harness, None, overall_timeout)
    if not success:
        return 0.0, 0, ()
    return _parse_harness_output(stdout)


def evaluate_task(task: EvalTask) -> EvalResult:
    if _extract_interact_code(task.response):
        return EvalResult(
            reward=0.0,
            terminated=True,
            truncated=True,
            timeout_count=0,
            timeout_indices=(),
        )

    answer_code = _extract_answer_code(task.response)
    if not answer_code:
        return EvalResult(
            reward=0.0,
            terminated=True,
            truncated=True,
            timeout_count=0,
            timeout_indices=(),
        )

    # If require_solution_class is True and problem has fn_name, enforce Solution class requirement
    if task.require_solution_class:
        fn_name = None
        if isinstance(task.tests, dict):
            fn_name = task.tests.get("fn_name")
        # Only invalidate if problem expects Solution class (has fn_name) but code doesn't have it
        if fn_name and "class Solution" not in answer_code:
            return EvalResult(
                reward=0.0,
                terminated=True,
                truncated=True,
                timeout_count=0,
                timeout_indices=(),
                invalidated=True,
            )
        # If no fn_name, allow stdin-based execution even with require_solution_class=True

    reward, timeout_count, timeout_indices = _evaluate_code(
        answer_code,
        task.tests,
        task.max_tests,
        task.timeout_s,
        task.max_timeout_records,
        task.require_solution_class,
    )
    return EvalResult(
        reward=reward,
        terminated=True,
        truncated=False,
        timeout_count=timeout_count,
        timeout_indices=timeout_indices,
    )


def _evaluate_task_batch(tasks: list[EvalTask]) -> list[EvalResult]:
    return [evaluate_task(task) for task in tasks]


def evaluate_tasks(
    tasks: list[EvalTask],
    max_workers: int,
    batch_size: int,
    show_progress: bool = False,
    executor: ProcessPoolExecutor | None = None,
) -> list[EvalResult]:
    if not tasks:
        return []

    if batch_size <= 0:
        batch_size = 1

    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    def _run_with_executor(pool: ProcessPoolExecutor) -> list[EvalResult]:
        results_iter = pool.map(_evaluate_task_batch, batches)
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

    if executor is not None:
        return _run_with_executor(executor)

    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
        return _run_with_executor(pool)
