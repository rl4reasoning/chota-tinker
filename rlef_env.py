"""
RLEF (Reinforcement Learning with Execution Feedback) environment.

Thin wrapper around IntellectCodeEnv implementing the RLEF paper's
multi-turn inference loop (arXiv:2410.02089):

  1. Model receives problem + public test cases
  2. Model generates code solution
  3. Code is evaluated against public tests → structured feedback
  4. Model refines solution (or episode ends)
  5. Final solution scored against all tests (public + private)

Usage:
    env = RLEFCodeEnv(num_public_tests=3, max_turns=3, ...)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import json as _json
from typing import Any, Optional, Tuple

from intellect_env import IntellectCodeEnv
from utils.fast_eval import _exec_code_subprocess, _evaluate_code, _normalize_io
from code_env.code_env.utils.deepcoder_utils import BASE_IMPORTS, process_input_output


# ======================================================================
# Standalone functions for batched public test execution.
# These are module-level (picklable) so they can run in a ProcessPoolExecutor.
# ======================================================================

def _build_public_test_harness(
    code: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: Optional[str],
    timeout_s: Optional[float],
) -> str:
    """Build a harness that runs code against multiple public tests in one process.

    Returns a Python script that outputs a JSON array of per-test results:
    ``[{"passed": bool, "actual": str|null, "error": str|null}, ...]``
    """
    code_with_imports = BASE_IMPORTS + "\n" + code
    safe_timeout = timeout_s if timeout_s and timeout_s > 0 else 10.0
    return f"""
import contextlib
import io
import json
import os
import signal
import sys

_code = {repr(code_with_imports)}
_inputs = {repr(inputs)}
_expected = {repr(outputs)}
_fn_name = {repr(fn_name)}
_timeout_s = {repr(safe_timeout)}

_original_os_exit = os._exit
def _safe_os_exit(code=0):
    raise SystemExit(code)
os._exit = _safe_os_exit

class _TimeoutException(Exception):
    pass

def _handler(signum, frame):
    raise _TimeoutException("Time Limit Exceeded")

_results = []

if _fn_name:
    # fn_name mode: extract Solution class once, call method per test
    _ns = {{}}
    _class_ok = True
    try:
        exec(_code, _ns)
    except Exception as _e:
        for _ in _inputs:
            _results.append({{"passed": False, "actual": None, "error": f"{{type(_e).__name__}}: {{_e}}"}})
        _class_ok = False

    if _class_ok:
        _sol_class = _ns.get("Solution")
        _fn = getattr(_sol_class, _fn_name, None) if _sol_class else None
        if not _sol_class:
            for _ in _inputs:
                _results.append({{"passed": False, "actual": None, "error": "No Solution class found"}})
        elif not _fn:
            for _ in _inputs:
                _results.append({{"passed": False, "actual": None, "error": f"Method '{{_fn_name}}' not found"}})
        else:
            _sol = _sol_class()
            for _inp, _exp in zip(_inputs, _expected):
                _r = {{"passed": False, "actual": None, "error": None}}
                try:
                    _old_h = signal.signal(signal.SIGALRM, _handler)
                    signal.setitimer(signal.ITIMER_REAL, _timeout_s)
                    _args = list(map(eval, _inp.split("\\n"))) if _inp.strip() else []
                    _actual = getattr(_sol, _fn_name)(*_args)
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, _old_h)
                    _actual_str = str(_actual)
                    _r["actual"] = _actual_str
                    try:
                        _ev = json.loads(_exp)
                        _cmp = list(_actual) if isinstance(_actual, tuple) else _actual
                        _passed = _cmp == _ev
                        if not _passed and isinstance(_ev, list) and len(_ev) > 0:
                            _passed = _cmp == _ev[0]
                    except (json.JSONDecodeError, TypeError):
                        _passed = _actual_str.strip() == _exp.strip()
                    _r["passed"] = _passed
                except _TimeoutException:
                    _r["error"] = "Time Limit Exceeded"
                except SystemExit as _se:
                    if _se.code not in (None, 0):
                        _r["error"] = f"SystemExit: {{_se}}"
                except Exception as _e:
                    _r["error"] = f"{{type(_e).__name__}}: {{_e}}"
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    try:
                        signal.signal(signal.SIGALRM, _old_h)
                    except Exception:
                        pass
                _results.append(_r)
else:
    # stdin mode: exec code per test with redirected I/O
    _compiled = compile(_code, "<solution>", "exec")
    for _inp, _exp in zip(_inputs, _expected):
        _r = {{"passed": False, "actual": None, "error": None}}
        _stdout_buf = io.StringIO()
        try:
            _old_h = signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, _timeout_s)
            _old_stdin = sys.stdin
            sys.stdin = io.StringIO(_inp)
            try:
                with contextlib.redirect_stdout(_stdout_buf):
                    exec(_compiled, {{"__name__": "__main__"}})
            finally:
                sys.stdin = _old_stdin
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, _old_h)
            _actual = _stdout_buf.getvalue()
            _r["actual"] = _actual.strip()
            _exp_clean = _exp.strip()
            if _exp_clean.startswith('"') and _exp_clean.endswith('"'):
                _exp_clean = _exp_clean[1:-1]
            _passed = _actual.strip() == _exp_clean
            if not _passed:
                _a_lines = [l.strip() for l in _actual.strip().splitlines() if l.strip()]
                _e_lines = [l.strip() for l in _exp_clean.splitlines() if l.strip()]
                _passed = _a_lines == _e_lines
            _r["passed"] = _passed
        except _TimeoutException:
            _r["error"] = "Time Limit Exceeded"
        except SystemExit as _se:
            if _se.code in (None, 0):
                _actual = _stdout_buf.getvalue()
                _r["actual"] = _actual.strip()
                _exp_clean = _exp.strip()
                if _exp_clean.startswith('"') and _exp_clean.endswith('"'):
                    _exp_clean = _exp_clean[1:-1]
                _r["passed"] = _actual.strip() == _exp_clean
            else:
                _r["error"] = f"SystemExit: {{_se}}"
        except Exception as _e:
            _r["error"] = f"{{type(_e).__name__}}: {{_e}}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                signal.signal(signal.SIGALRM, _old_h)
            except Exception:
                pass
        _results.append(_r)

os._exit = _original_os_exit
print(json.dumps(_results))
"""


def run_public_tests(
    code: str, public_tests: dict[str, Any], eval_timeout_s: float
) -> list[dict[str, Any]]:
    """Run code against public tests in a single subprocess.

    This is a module-level function so it can be submitted to a
    ``ProcessPoolExecutor``.  Returns a list of per-test result dicts with
    keys: index, passed, actual, error, input_display, expected_display.
    """
    fn_name = public_tests.get("fn_name", None)
    raw_inputs = list(public_tests.get("inputs", []))
    raw_outputs = list(public_tests.get("outputs", []))

    norm_inputs: list[str] = []
    norm_outputs: list[str] = []
    display_inputs: list[str] = []
    display_outputs: list[str] = []

    for raw_inp, raw_out in zip(raw_inputs, raw_outputs):
        inp, out = process_input_output(raw_inp, raw_out)
        inp_str = str(_normalize_io(inp)) if inp is not None else ""
        out_str = str(_normalize_io(out)) if out is not None else ""
        norm_inputs.append(inp_str)
        norm_outputs.append(out_str)
        display_inputs.append(inp_str)
        display_outputs.append(out_str)

    harness = _build_public_test_harness(
        code, norm_inputs, norm_outputs, fn_name, eval_timeout_s,
    )
    num_tests = len(norm_inputs)
    overall_timeout = (eval_timeout_s or 10.0) * num_tests + 5.0
    success, stdout, stderr = _exec_code_subprocess(harness, None, overall_timeout)

    if not success:
        error_msg = stderr.strip() if stderr else "Unknown execution error"
        if "timed out" in error_msg.lower():
            error_msg = "Time Limit Exceeded"
        return [
            {"index": i + 1, "passed": False, "actual": None, "error": error_msg,
             "input_display": display_inputs[i], "expected_display": display_outputs[i]}
            for i in range(num_tests)
        ]

    try:
        payload = _json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, _json.JSONDecodeError):
        return [
            {"index": i + 1, "passed": False, "actual": None,
             "error": "Could not parse test output",
             "input_display": display_inputs[i], "expected_display": display_outputs[i]}
            for i in range(num_tests)
        ]

    results: list[dict[str, Any]] = []
    for i, item in enumerate(payload):
        results.append({
            "index": i + 1,
            "passed": item.get("passed", False),
            "actual": item.get("actual"),
            "error": item.get("error"),
            "input_display": display_inputs[i],
            "expected_display": display_outputs[i],
        })
    return results


class RLEFCodeEnv(IntellectCodeEnv):
    """RLEF-style multi-turn code environment.

    Splits test cases into public (for feedback) and private (for reward).
    Each turn the model submits a code solution which is evaluated against
    public tests. Structured per-test feedback is returned on failure.
    When public tests all pass or the turn limit is reached, the solution
    is scored against ALL tests for the final reward.
    """

    def __init__(
        self,
        num_public_tests: int = 3,
        eval_timeout_s: Optional[float] = 10.0,
        **kwargs,
    ):
        kwargs.setdefault("interaction_mode", False)
        kwargs["eval_timeout_s"] = eval_timeout_s
        super().__init__(**kwargs)
        self.num_public_tests = num_public_tests
        self.public_tests: dict[str, Any] = {}
        self.private_tests: dict[str, Any] = {}
        self._last_valid_code: Optional[str] = None

    # ------------------------------------------------------------------
    # reset / step overrides
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        obs, info = super().reset(seed)
        self._split_tests()
        self._last_valid_code = None
        obs = self._build_initial_obs()
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Run one turn of the RLEF loop.

        When the episode ends (public tests all pass or turn limit reached),
        final evaluation is NOT performed here. Instead ``info["needs_eval"]``
        is set to ``True`` and ``info["code"]`` contains the solution to
        evaluate. The caller (e.g. ``collect_trajectories_rlef.py``) is
        responsible for batching final evaluations efficiently.

        For convenience in the demo script, call ``evaluate_final()`` to get
        the reward for a completed episode.
        """
        self.current_turn += 1

        code = self._extract_answer_code(action)
        if not code:
            return self._handle_no_code()

        self._last_valid_code = code
        test_results = self._run_public_tests(code)
        all_passed = all(r["passed"] for r in test_results)

        if all_passed:
            return "", 0.0, True, False, {
                "final": True,
                "public_all_passed": True,
                "needs_eval": True,
                "code": code,
            }

        at_turn_limit = self.current_turn >= self.max_turns
        if at_turn_limit:
            return "", 0.0, True, False, {
                "final": True,
                "public_all_passed": False,
                "needs_eval": True,
                "code": code,
            }

        feedback = self._format_feedback(test_results)
        return feedback, 0.0, False, False, {"public_all_passed": False}

    def evaluate_final(self, code: str) -> float:
        """Convenience method: evaluate code against private tests and return reward.

        Useful in the demo script where batched evaluation is not needed.
        """
        return self._evaluate_all_tests(code)

    # ------------------------------------------------------------------
    # Test splitting
    # ------------------------------------------------------------------

    def _split_tests(self) -> None:
        inputs = list(self.tests.get("inputs", []))
        outputs = list(self.tests.get("outputs", []))
        n = min(self.num_public_tests, len(inputs))

        base = {"fn_name": self.tests.get("fn_name", None)}
        self.public_tests = {**base, "inputs": inputs[:n], "outputs": outputs[:n]}
        self.private_tests = {**base, "inputs": inputs[n:], "outputs": outputs[n:]}

    # ------------------------------------------------------------------
    # Initial observation
    # ------------------------------------------------------------------

    ## test cases in the prompt or not
    def _build_initial_obs(self) -> str:
        parts = [self.question, "\nPublic test cases:"]
        for i, (inp, out) in enumerate(
            zip(self.public_tests["inputs"], self.public_tests["outputs"]), 1
        ):
            inp_str = str(_normalize_io(inp)).strip()
            out_str = str(_normalize_io(out)).strip()
            parts.append(f"\nTest {i}:")
            parts.append(f"Input:\n{inp_str}")
            parts.append(f"Expected Output:\n{out_str}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public test execution (per-test feedback)
    # ------------------------------------------------------------------

    def _run_public_tests(self, code: str) -> list[dict[str, Any]]:
        """Run code against public tests using the module-level function."""
        return run_public_tests(code, self.public_tests, self.eval_timeout_s or 10.0)

    # ------------------------------------------------------------------
    # Feedback formatting
    # ------------------------------------------------------------------

    def _format_feedback(self, test_results: list[dict[str, Any]]) -> str:
        lines = ["Your code was tested on the public test cases:\n"]
        for r in test_results:
            idx = r["index"]
            if r["passed"]:
                lines.append(f"Test {idx}: PASSED")
            elif r.get("error"):
                lines.append(f"Test {idx}: FAILED ({r['error']})")
                lines.append(f"  Input:\n  {r['input_display']}")
                lines.append(f"  Expected output:\n  {r['expected_display']}")
            else:
                lines.append(f"Test {idx}: FAILED")
                lines.append(f"  Input:\n  {r['input_display']}")
                lines.append(f"  Expected output:\n  {r['expected_display']}")
                if r.get("actual") is not None:
                    lines.append(f"  Your output:\n  {r['actual']}")
        lines.append("\nFix your solution and try again.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private (final) evaluation
    # ------------------------------------------------------------------

    def _evaluate_all_tests(self, code: str) -> float:
        """Evaluate against private tests only for final reward.

        Returns the fraction of private tests passed (0.0 to 1.0).
        Public tests are excluded since the model already received feedback
        on them during the conversation.
        """
        if not self.private_tests.get("inputs"):
            return 0.0
        reward, _, _ = _evaluate_code(
            code=code,
            tests=self.private_tests,
            max_tests=self.max_tests,
            timeout_s=self.eval_timeout_s,
            timeout_record_limit=0,
            require_solution_class=True,
        )
        return reward

    # ------------------------------------------------------------------
    # No-code fallback
    # ------------------------------------------------------------------

    def _handle_no_code(self) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        obs = "No valid Python code block found. Please provide your solution inside a ```python``` code block."
        at_turn_limit = self.current_turn >= self.max_turns
        if at_turn_limit:
            if self._last_valid_code:
                return "", 0.0, True, False, {
                    "final": True,
                    "no_code_this_turn": True,
                    "needs_eval": True,
                    "code": self._last_valid_code,
                }
            return "", 0.0, True, False, {"final": True, "no_code_ever": True}
        return obs, 0.0, False, False, {}
