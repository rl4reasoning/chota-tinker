# Utils - Evaluation Infrastructure

This document covers the evaluation infrastructure in the `utils/` folder, including parallel execution flow and integration with `code_env`.

---

## Parallel Evaluation Flow

This section explains how batching and workers interact in the fast evaluator.

### High-level hierarchy

```
Main process
└─ ProcessPoolExecutor (workers)
   ├─ Worker process #1
   │  └─ Batch (size = eval-batch-size)
   │     ├─ EvalTask #1
   │     │  └─ Harness exec (subprocess with 10GB memory limit)
   │     │     ├─ BASE_IMPORTS prepended
   │     │     ├─ Solution-case: exec code once → Solution instance → loop tests
   │     │     └─ Stdin-case: compile once → exec per test (fresh globals/stdin)
   │     ├─ EvalTask #2
   │     └─ ...
   ├─ Worker process #2
   │  └─ Batch ...
   └─ ...
```

### Batches over time

Workers persist and pull multiple batches; a batch does not spawn a new process.

```
Main process
└─ ProcessPoolExecutor (workers)
   ├─ Worker #1
   │  ├─ Batch A
   │  ├─ Batch D
   │  └─ Batch G
   ├─ Worker #2
   │  ├─ Batch B
   │  ├─ Batch E
   │  └─ Batch H
   └─ Worker #3
      ├─ Batch C
      ├─ Batch F
      └─ ...
```

### Batch size effect

`eval-batch-size` only affects parallel mode:
- Larger batches reduce cross-process scheduling/IPC overhead.
- Smaller batches improve load balancing and progress granularity.

### Interaction execution

Interaction code now executes in subprocess with:
- `BASE_IMPORTS` prepended for consistency with evaluation
- Hard timeout via `subprocess.run(timeout=...)` that can kill hung processes
- Memory limit matching evaluation (10GB)

---

## Integration with code_env

Our local evaluation (`fast_eval.py`, `intellect_env.py`) achieves functional parity with the `code_env` package used for remote sandbox evaluation.

### What we import from code_env

We import utilities from `code_env/code_env/utils/deepcoder_utils.py`:

```python
from code_env.code_env.utils.deepcoder_utils import (
    extract_code_from_model,  # Extract Python code from markdown blocks
    BASE_IMPORTS,             # Common imports (itertools, collections, etc.)
    process_input_output,     # Normalize dict keys in test inputs/outputs
)
```

#### `extract_code_from_model`
Extracts Python code from model responses, handling various markdown formats:
```python
response = "Here's my solution:\n```python\nprint('hello')\n```"
code = extract_code_from_model(response)  # Returns: "print('hello')"
```

#### `BASE_IMPORTS`
Standard imports prepended to all code execution (both interactions and evaluation):
```python
from itertools import accumulate, chain, combinations, permutations, ...
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop, heapify, ...
from math import floor, log2, sqrt, gcd, ceil, inf, ...
# ... and more
```

#### `process_input_output`
Normalizes test case inputs/outputs, particularly converting JSON string keys back to integers:
```python
inputs = {"1": "value"}  # JSON serialization converts int keys to strings
inputs, outputs = process_input_output(inputs, outputs)
# inputs is now {1: "value"}
```

### Evaluation comparison table

| Aspect | code_env | Our Implementation |
|--------|----------|-------------------|
| **Execution** | Remote sandbox (`AsyncSandboxClient`) | Local subprocess |
| **Parallelism** | 32 concurrent tests per sandbox | Sequential in harness, parallel across tasks |
| **Memory limit** | `ulimit -v 10485760` (10GB) | Same |
| **BASE_IMPORTS** | Prepended to code | Same |
| **Code extraction** | `extract_code_from_model` | Same (imported) |
| **Input processing** | `process_input_output` | Same (imported) |
| **Input parsing** | `eval(inputs.split("\n"))` | Same |
| **Stdout comparison** | Tiered (trimmed → linewise → tokenwise → numeric) | Same (ported into harness) |
| **Function result comparison** | tuple→list, fallback to `[0]` | Same |
| **max_tests default** | 15 | Same |

### What we DON'T use from code_env

- **`verification_utils.py`** - Remote sandbox execution (`run_test_cases`, `run_standard_input`, `run_func_call`)
- **`sandbox_pool.py`** - Remote sandbox lifecycle management
- **`clean_code_main_block`** - We exec directly, don't need to strip `__main__` blocks
- **`generate_cb_wrapper_script`** - Generates per-test scripts for remote upload

### Architecture comparison

```
code_env (remote sandbox):
┌─────────────┐     ┌─────────────────────────┐
│ Main Process│────▶│ Remote Sandbox Pool     │
│             │     │ (AsyncSandboxClient)    │
│             │     │                         │
│             │     │ Per sandbox:            │
│             │     │ - Upload code bundle    │
│             │     │ - 32 concurrent tests   │
│             │     │ - ulimit -v 10485760    │
└─────────────┘     └─────────────────────────┘

Our implementation (local subprocess):
┌─────────────┐     ┌─────────────────────────┐
│ Main Process│────▶│ ProcessPoolExecutor     │
│             │     │ (N worker processes)    │
│             │     │                         │
│             │     │ Per worker:             │
│             │     │ - Batch of EvalTasks    │
│             │     │ - Subprocess per task   │
│             │     │ - 10GB memory limit     │
│             │     │ - Sequential tests      │
└─────────────┘     └─────────────────────────┘
```

### Test harness flow

```
EvalTask
│
├─ Extract code (extract_code_from_model)
├─ Process tests (process_input_output)
│
└─ Build harness
   │
   ├─ Prepend BASE_IMPORTS
   ├─ Embed code, inputs, expected outputs
   │
   └─ Execute in subprocess
      │
      ├─ If fn_name exists:
      │  └─ _run_solution_case()
      │     ├─ exec code → get Solution class or standalone function
      │     ├─ For each test: parse args → call function → compare result
      │     └─ JSON comparison with tuple→list normalization
      │
      └─ Else (stdin-based):
         └─ _run_script_case()
            ├─ compile code once
            ├─ For each test: exec with stdin → capture stdout
            └─ Tiered stdout comparison:
               1. Trimmed string match
               2. Line-wise match
               3. Token-wise match
               4. Numeric tolerance (1e-3)
```

### Functional equivalence

Our evaluation produces **identical results** to code_env for:
- Code extraction from model responses
- Test case input/output processing  
- Stdout comparison (all 4 tiers)
- Function call result comparison (tuple/list normalization)
- Memory limits (10GB)

The only differences are architectural (local vs remote execution).

---

## Testing parity

Use `tests/test_eval_parity.py` to verify evaluation matches expected results:

```bash
# Basic test
python tests/test_eval_parity.py \
  --dataset bicycleman15/1k_32_s1 \
  --max-rows 100 \
  --compare-original

# Debug specific cases
python tests/test_eval_parity.py \
  --dataset bicycleman15/1k_32_s1 \
  --max-rows 100 \
  --compare-original \
  --debug-degraded 3
```
