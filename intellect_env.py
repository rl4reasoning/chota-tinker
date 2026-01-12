"""
Custom GEM environment for INTELLECT-3-RL dataset with multi-turn Python REPL.
"""

import json
import re
import random
from typing import Any, Optional, Tuple
from datasets import Dataset, load_dataset
from gem.core import Env
from gem.utils.sandbox import run_python


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
        sandbox_type: str = "none",
        seed: int = 0,
        dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
        problem_index: Optional[int] = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tests = max_tests
        self.sandbox_type = sandbox_type
        self.seed = seed
        self.dataset_name = dataset_name
        self.problem_index = problem_index
        
        # Custom datasets (bicycleman15/*) don't use config parameter
        if dataset_name.startswith("bicycleman15/"):
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
            try:
                success, stdout, stderr = run_python(python_code, self.sandbox_type)
                if success:
                    output = stdout if stdout else "(no output during interaction -- did you forget to print? did you enclose the code correctly in <interact></interact>?)"
                else:
                    output = f"Error:\n{stderr}" if stderr else "Error: execution failed."
            except Exception as e:
                success = False
                output = f"Error: failed to execute code - {str(e)}"
            
            self.history.append({"code": python_code, "output": output})
            self.has_interacted = True  # Mark that interaction occurred
            obs = f"<output>\n{output}</output>"
            
            if self.current_turn >= self.max_turns:
                return obs, 0.0, True, True, {"truncated": True}
            
            return obs, 0.0, False, False, {}
        
        # check for final answer (only if no <interact> tag)
        answer_code = self._extract_answer_code(action)
        if answer_code:
            # Require at least one interaction before final answer
            if not self.has_interacted:
                obs = "<output>You must interact at least once before submitting your final answer. Use <interact> ... your code here ... </interact> to test your code first. Remember to pass in the inputs yourself.</output>"
                if self.current_turn >= self.max_turns:
                    return obs, 0.0, True, True, {"truncated": True}
                return obs, 0.0, False, False, {}
            
            reward = self._evaluate(answer_code)
            return "", reward, True, False, {"final": True}
        
        # no valid code found
        obs = "<output>No valid code block found. Use <interact></interact> or ```python```.</output>"
        
        if self.current_turn >= self.max_turns:
            return obs, 0.0, True, True, {"truncated": True}
        
        return obs, 0.0, False, False, {}

    def _build_observation(self, content: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{content}"
        return content

    def _extract_interact_code(self, text: str) -> Optional[str]:
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

    def _extract_answer_code(self, text: str) -> Optional[str]:
        # match ```python for final answer
        pattern = r"```python\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    def _evaluate(self, code: str) -> float:
        tests = self.tests
        
        total_tests = len(tests["inputs"])
        if total_tests > self.max_tests:
            indices = sorted(
                range(total_tests),
                key=lambda i: len(tests["inputs"][i]),
                reverse=True,
            )[:self.max_tests]
            tests = {
                "inputs": [tests["inputs"][i] for i in indices],
                "outputs": [tests["outputs"][i] for i in indices],
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

