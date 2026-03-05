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

from typing import Any, Optional, Tuple

from intellect_env import IntellectCodeEnv
from utils.fast_eval import _exec_code_subprocess, _evaluate_code, _normalize_io
from code_env.code_env.utils.deepcoder_utils import BASE_IMPORTS, process_input_output


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
        fn_name = self.public_tests.get("fn_name", None)
        results: list[dict[str, Any]] = []

        for i, (raw_inp, raw_out) in enumerate(
            zip(self.public_tests["inputs"], self.public_tests["outputs"])
        ):
            inp, out = process_input_output(raw_inp, raw_out)
            inp_str = str(_normalize_io(inp)) if inp is not None else ""
            out_str = str(_normalize_io(out)) if out is not None else ""

            result = self._run_single_test(code, inp_str, out_str, fn_name)
            result["index"] = i + 1
            result["input_display"] = inp_str
            result["expected_display"] = out_str
            results.append(result)

        return results

    def _run_single_test(
        self, code: str, inp: str, expected: str, fn_name: Optional[str]
    ) -> dict[str, Any]:
        harness = self._build_single_test_harness(code, inp, expected, fn_name)
        timeout = self.eval_timeout_s or 10.0
        overall_timeout = timeout + 5.0
        success, stdout, stderr = _exec_code_subprocess(harness, None, overall_timeout)

        if not success:
            error_msg = stderr.strip() if stderr else "Unknown execution error"
            if "timed out" in error_msg.lower():
                return {"passed": False, "error": "Time Limit Exceeded", "actual": None}
            return {"passed": False, "error": error_msg, "actual": None}

        import json as _json
        try:
            payload = _json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError, _json.JSONDecodeError):
            return {"passed": False, "error": "Could not parse test output", "actual": None}

        return {
            "passed": payload.get("passed", False),
            "actual": payload.get("actual"),
            "error": payload.get("error"),
        }

    @staticmethod
    def _build_single_test_harness(
        code: str, inp: str, expected: str, fn_name: Optional[str]
    ) -> str:
        code_with_imports = BASE_IMPORTS + "\n" + code
        safe_timeout = 10.0
        return f"""
import contextlib
import io
import json
import os
import signal
import sys

_code = {repr(code_with_imports)}
_input = {repr(inp)}
_expected = {repr(expected)}
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

def _run():
    result = {{"passed": False, "actual": None, "error": None}}
    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, _timeout_s)

        if _fn_name:
            ns = {{}}
            exec(_code, ns)
            sol_class = ns.get("Solution")
            if not sol_class:
                result["error"] = "No Solution class found"
                return result
            sol = sol_class()
            fn = getattr(sol, _fn_name, None)
            if not fn:
                result["error"] = f"Method '{{_fn_name}}' not found on Solution"
                return result
            args = list(map(eval, _input.split("\\n"))) if _input.strip() else []
            actual = fn(*args)
            actual_str = str(actual)
            try:
                expected_val = json.loads(_expected)
                if isinstance(actual, tuple):
                    actual = list(actual)
                passed = actual == expected_val
                if not passed and isinstance(expected_val, list) and len(expected_val) > 0:
                    passed = actual == expected_val[0]
            except (json.JSONDecodeError, TypeError):
                passed = actual_str.strip() == _expected.strip()
            result["passed"] = passed
            result["actual"] = actual_str
        else:
            stdin_buf = io.StringIO(_input)
            stdout_buf = io.StringIO()
            old_stdin = sys.stdin
            sys.stdin = stdin_buf
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(_code, {{"__name__": "__main__"}})
            finally:
                sys.stdin = old_stdin
            actual = stdout_buf.getvalue()
            result["actual"] = actual.strip()
            expected_clean = _expected.strip()
            if expected_clean.startswith('"') and expected_clean.endswith('"'):
                expected_clean = expected_clean[1:-1]
            passed = actual.strip() == expected_clean
            if not passed:
                a_lines = [l.strip() for l in actual.strip().splitlines() if l.strip()]
                e_lines = [l.strip() for l in expected_clean.splitlines() if l.strip()]
                passed = a_lines == e_lines
            result["passed"] = passed
    except _TimeoutException:
        result["error"] = "Time Limit Exceeded"
    except SystemExit as exc:
        if exc.code in (None, 0):
            actual = stdout_buf.getvalue() if 'stdout_buf' in dir() else ""
            result["actual"] = actual.strip()
            expected_clean = _expected.strip()
            if expected_clean.startswith('"') and expected_clean.endswith('"'):
                expected_clean = expected_clean[1:-1]
            result["passed"] = actual.strip() == expected_clean
        else:
            result["error"] = f"SystemExit: {{exc}}"
    except Exception as exc:
        result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass
    return result

r = _run()
os._exit = _original_os_exit
print(json.dumps(r))
"""

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
