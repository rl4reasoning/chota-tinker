# Parallel Evaluation Flow

This note explains how batching and workers interact in the fast evaluator.

## High-level hierarchy

```
Main process
└─ ProcessPoolExecutor (workers)
   ├─ Worker process #1
   │  └─ Batch (size = eval-batch-size)
   │     ├─ EvalTask #1
   │     │  └─ Harness exec
   │     │     ├─ Solution-case: exec code once → Solution instance → loop tests
   │     │     └─ Stdin-case: compile once → exec per test (fresh globals/stdin)
   │     ├─ EvalTask #2
   │     └─ ...
   ├─ Worker process #2
   │  └─ Batch ...
   └─ ...
```

## Batches over time

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

## Batch size effect

`eval-batch-size` only affects parallel mode:
- Larger batches reduce cross-process scheduling/IPC overhead.
- Smaller batches improve load balancing and progress granularity.

## Fast interactions (future work)

We now execute interaction code in-process for speed. This removes OS-level sandboxing.
If safety becomes a concern, add a guarded mode that falls back to `run_python`
for interactions that match risky patterns (filesystem, network, dynamic exec).
