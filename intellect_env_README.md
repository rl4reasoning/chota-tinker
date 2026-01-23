# IntellectCodeEnv - Multi-turn Code Environment

This document describes `intellect_env.py`, a multi-turn Python REPL environment for the INTELLECT-3-RL dataset.

## Overview

`IntellectCodeEnv` extends the GEM `Env` class to provide:
- **Multi-turn interactions**: Model can test code before submitting final answer
- **Interactive execution**: `<interact>` tags for exploratory code execution
- **Final evaluation**: `python` code blocks for graded submissions
- **Parallel batch processing**: Efficient evaluation of multiple environments

---

## Action Types

The model can submit two types of actions:

### 1. Interaction (`<interact>`)

```
<interact>
# Test some code
print(sum([1, 2, 3]))
</interact>
```

- Executed immediately, output returned in `<output>...</output>`
- Does NOT count as final answer
- Model MUST interact at least once before submitting final answer
- Uses `BASE_IMPORTS` (same environment as evaluation)

### 2. Final Answer (```python```)

````
```python
class Solution:
    def twoSum(self, nums, target):
        # Final implementation
        ...
```
````

- Evaluated against test cases
- Returns reward (0.0 to 1.0)
- Episode terminates

---

## Environment Flow

```
reset()
│
├─ Load problem from dataset
├─ Parse tests from info JSON
└─ Return initial observation

step(action)
│
├─ Check for <interact> tag
│  └─ Yes: Execute code → Return output → Continue
│
├─ Check for ```python``` block
│  ├─ Has interacted? 
│  │  ├─ No & not at turn limit: Require interaction first
│  │  └─ Yes or at limit: Evaluate against tests
│  └─ Return reward, terminate
│
└─ Neither found: Return "invalid" message
```

---

## Key Components

### Code Extraction

```python
# Interaction code
_extract_interact_code(text)  # Extracts from <interact>...</interact>

# Final answer code  
_extract_answer_code(text)    # Uses code_env's extract_code_from_model
```

### Execution Functions

```python
# Subprocess execution (used by default)
_exec_interaction_code_subprocess(code, timeout_s)
# - Prepends BASE_IMPORTS
# - Hard timeout via subprocess
# - Can kill hung C extensions

# In-process execution (alternative, not used)
_exec_interaction_code(code, timeout_s)
# - Prepends BASE_IMPORTS
# - Uses SIGALRM for timeout
# - Faster but can't interrupt GIL-holding code
```

### Evaluation

```python
_evaluate(code)
# - Uses _evaluate_code from fast_eval.py
# - Tiered comparison strategies
# - Tuple/list normalization
# - Memory limits (10GB)
```

---

## ExecutorPool

Manages persistent `ProcessPoolExecutor` instances for efficient parallel execution:

```python
class ExecutorPool:
    def get(max_workers) -> ProcessPoolExecutor
    def shutdown() -> None
```

Two module-level pools:
- `_interaction_pool` - For parallel interaction execution
- `_eval_pool` - For parallel evaluation

Pools are automatically shutdown on exit via `atexit`.

---

## Batch Processing

### `step_batch()`

Efficiently processes multiple environments in parallel:

```python
results = step_batch(
    envs=list_of_envs,
    actions=list_of_actions,
    eval_workers=8,
    eval_batch_size=8,
    eval_timeout_s=5.0,
    show_progress=True,
    use_persistent_pool=True,
)
```

**Flow:**

```
step_batch(envs, actions)
│
├─ Categorize actions:
│  ├─ Interactions → interact_tasks
│  ├─ Final answers → eval_tasks
│  └─ Invalid → immediate results
│
├─ Batch execute interactions (parallel)
│  └─ _run_python_batch() with interaction_pool
│
├─ Batch evaluate finals (parallel)
│  └─ evaluate_tasks() with eval_pool
│
└─ Combine and return all results
```

---

## Configuration

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `system_prompt` | `""` | Prepended to observations |
| `config` | `"code"` | Dataset config |
| `split` | `"train"` | Dataset split |
| `max_turns` | `5` | Maximum turns before truncation |
| `max_tests` | `15` | Max test cases for evaluation |
| `interaction_timeout_s` | `None` | Timeout for interactions |
| `sandbox_type` | `"none"` | Sandbox type (unused) |
| `dataset_name` | `"PrimeIntellect/INTELLECT-3-RL"` | HuggingFace dataset |
| `problem_index` | `None` | Fixed problem index (or iterate) |
| `dataset` | `None` | Pre-loaded dataset |
| `interaction_mode` | `False` | Add interaction reminder |

---

## Integration with code_env

Uses utilities from `code_env/code_env/utils/deepcoder_utils.py`:

```python
from code_env.code_env.utils.deepcoder_utils import (
    extract_code_from_model,  # For final answer extraction
    BASE_IMPORTS,             # Prepended to all code execution
)
```

Uses evaluation from `utils/fast_eval.py`:

```python
from utils.fast_eval import (
    EvalTask,
    evaluate_task,
    evaluate_tasks,
    _evaluate_code,  # Direct evaluation function
)
```

---

## Usage Example

### Single Environment

```python
env = IntellectCodeEnv(
    max_turns=5,
    max_tests=15,
    interaction_mode=True,
)

obs, info = env.reset()
print(obs)  # Problem description

# Interaction
action = "<interact>\nprint(2 + 2)\n</interact>"
obs, reward, terminated, truncated, info = env.step(action)
print(obs)  # <output>4</output>

# Final answer
action = "```python\nclass Solution:\n    def solve(self, x): return x * 2\n```"
obs, reward, terminated, truncated, info = env.step(action)
print(reward)  # 0.0 - 1.0
```

### Batch Processing

```python
envs = [IntellectCodeEnv() for _ in range(32)]
for env in envs:
    env.reset()

actions = ["<interact>\nprint('test')\n</interact>"] * 32

results = step_batch(
    envs=envs,
    actions=actions,
    eval_workers=8,
    eval_batch_size=8,
    eval_timeout_s=5.0,
)

for obs, reward, terminated, truncated, info in results:
    print(obs)
```

---

## Interaction Requirement

By default, the model must interact at least once before submitting a final answer:

1. **First turn with final answer** → "You must interact first" message
2. **After interaction** → Final answer is evaluated
3. **At turn limit** → Final answer evaluated even without interaction

This encourages the model to test its code before committing to a solution.

Disable by setting `interaction_mode=False` (but interaction tracking still applies).
