"""Collect trajectories with RSA (Recursive Self-Aggregation) and save as HuggingFace dataset.

Implements RSA from:
    Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models
    (https://arxiv.org/abs/2509.26626)

RSA maintains a population of N candidates per problem and iteratively refines
them by aggregating K random candidates into improved solutions over T steps.

Usage:
    python collect_trajectories_rsa.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --num-problems 10 \
        --population 4 \
        --k 2 \
        --steps 3 \
        \
        --fast-eval \
        --eval-workers 8 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/rsa_trajectories

Multi-GPU (launches one vLLM server per GPU, shards prompts across them):
    python collect_trajectories_rsa.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --vllm-multi-gpu \
        --vllm-gpu-ids 0,1,2,3 \
        --num-problems 50 \
        --population 16 \
        --k 4 \
        --steps 10 \
        \
        --fast-eval \
        --eval-workers 16 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/qwen3_4b_rsa_trajectories

Resume from checkpoint (if previous run failed):
    python collect_trajectories_rsa.py \
        --resume-from checkpoints/20260117_143052 \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        ... (same args as original run)

Checkpoints are automatically saved after each RSA step to:
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint.pkl
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint_info.json

Output format: DatasetDict with splits step_1, step_2, ..., step_T. Each split has the same
columns (problem_id, candidate_id, question, response, final_reward, is_successful, rsa_step,
aggregated_from, tests, rendered). Evaluation is performed at each RSA step.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

from checkpoint import CheckpointManager, get_checkpoint_dir
from intellect_env import IntellectCodeEnv, step_batch
from utils.pass_at_k import compute_pass_at_k
from utils.vllm_multi_gpu import (
    resolve_vllm_gpu_ids,
    build_vllm_server_urls,
    launch_vllm_servers,
    wait_for_vllm_servers,
    register_vllm_shutdown,
)

# Backend imports (conditional)
try:
    import tinker
    from tinker import types as tinker_types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

try:
    from chota_tinker import (
        SamplingClient,
        ServerSamplingClient,
        MultiServerSamplingClient,
        SamplingParams,
        ModelInput,
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""


def aggregate_prompt(question: str, candidates: list[str]) -> str:
    """
    Build aggregation prompt from question and K candidate solutions.
    Adapted from RSA paper Appendix F for code problems.
    """
    if len(candidates) == 1:
        # K=1: Self-refinement mode
        return f"""You are given a coding problem and a candidate solution.
The candidate may be incomplete or contain errors.
Refine this trajectory and produce an improved, higher-quality solution.
If it is entirely wrong, attempt a new strategy.
Provide your final solution wrapped in ```python``` code blocks.

Problem:
{question.strip()}

Candidate solution (may contain mistakes):
---- Candidate ----
{candidates[0].strip()}

Now refine the candidate into an improved solution. Provide clear reasoning and end with your final code in ```python``` blocks."""
    else:
        # K>1: Multi-trajectory aggregation
        parts = [f"""You are given a coding problem and several candidate solutions.
Some candidates may be incorrect or contain errors.
Aggregate the useful ideas and produce a single, high-quality solution.
Reason carefully; if candidates disagree, choose the correct path.
If all are incorrect, then attempt a different strategy.
Provide your final solution wrapped in ```python``` code blocks.

Problem:
{question.strip()}

Candidate solutions (may contain mistakes):"""]
        
        for i, cand in enumerate(candidates, 1):
            parts.append(f"---- Solution {i} ----\n{cand.strip()}")
        
        parts.append("\nNow write a single improved solution. Provide clear reasoning and end with your final code in ```python``` blocks.")
        return "\n".join(parts)


def render_trajectory(question: str, candidate: str, reward: float, 
                      rsa_step: int, total_steps: int, candidate_idx: int,
                      aggregated_from: list[int]) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.2f} | RSA Step: {rsa_step}/{total_steps} | Candidate: {candidate_idx}")
    if aggregated_from:
        lines.append(f"Aggregated from indices: {aggregated_from}")
    lines.append("=" * 80)
    lines.append(f"\n[RESPONSE]\n{candidate}")
    return "\n".join(lines)


@dataclass
class RSAProblemState:
    """Track state of RSA for a single problem."""
    problem_index: int
    question: str
    tests: dict
    # Current population of candidates (list of response strings)
    candidates: list[str] = field(default_factory=list)
    # Population at each step: candidates_by_step[step_idx] = list of N responses at that step (1-indexed step = step_idx + 1)
    candidates_by_step: list[list[str]] = field(default_factory=list)
    # Track which candidates were aggregated to produce each new candidate (current step only)
    # List of lists: aggregation_history[candidate_idx] = [source_indices]
    aggregation_history: list[list[int]] = field(default_factory=list)
    # Per-step aggregation: aggregation_history_by_step[step_idx] = list of N source-index lists
    aggregation_history_by_step: list[list[list[int]]] = field(default_factory=list)
    # Current RSA step (1 = initial, 2+ = aggregation steps)
    current_step: int = 0
    # Rewards per step: rewards_by_step[step_idx] = list of N rewards for that step
    rewards_by_step: list[list[float]] = field(default_factory=list)


def serialize_rsa_state(state: RSAProblemState) -> dict:
    """Serialize a RSAProblemState to a dictionary for checkpointing."""
    return {
        "problem_index": state.problem_index,
        "question": state.question,
        "tests": state.tests,
        "candidates": state.candidates.copy(),
        "candidates_by_step": [list(step) for step in state.candidates_by_step],
        "aggregation_history": [h.copy() for h in state.aggregation_history],
        "aggregation_history_by_step": [[list(h) for h in step] for step in state.aggregation_history_by_step],
        "current_step": state.current_step,
        "rewards_by_step": [list(r) for r in state.rewards_by_step],
    }


def deserialize_rsa_state(data: dict) -> RSAProblemState:
    """Deserialize a dictionary back to a RSAProblemState."""
    return RSAProblemState(
        problem_index=data["problem_index"],
        question=data["question"],
        tests=data["tests"],
        candidates=data["candidates"],
        candidates_by_step=data.get("candidates_by_step", []),
        aggregation_history=data["aggregation_history"],
        aggregation_history_by_step=data.get("aggregation_history_by_step", []),
        current_step=data["current_step"],
        rewards_by_step=data.get("rewards_by_step", []),
    )


def create_sampling_client(args):
    """Create sampling client based on backend choice."""
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        if args.vllm_multi_gpu:
            raise ValueError("--vllm-multi-gpu requires --backend vllm")
        service_client = tinker.ServiceClient()
        return service_client.create_sampling_client(base_model=args.model)
    else:  # vllm
        if not VLLM_AVAILABLE:
            raise ImportError("chota_tinker not installed. Install it or use --backend tinker")
        if args.vllm_multi_gpu:
            if args.vllm_server_url:
                raise ValueError("--vllm-server-url cannot be used with --vllm-multi-gpu")
            gpu_ids = resolve_vllm_gpu_ids(args)
            urls = build_vllm_server_urls(args, gpu_ids)
            print(f"Launching vLLM servers for GPUs: {', '.join(gpu_ids)}")
            processes = launch_vllm_servers(args, gpu_ids)
            register_vllm_shutdown(processes)
            wait_for_vllm_servers(urls, args.vllm_server_startup_timeout_s)
            return MultiServerSamplingClient(urls)
        if args.vllm_server_url:
            return ServerSamplingClient(args.vllm_server_url)
        else:
            return SamplingClient(
                args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )


def create_sampling_params(args, backend: str):
    """Create sampling params for the chosen backend."""
    if backend == "tinker":
        return tinker_types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params, tokenizer) -> list[str]:
    """Batch sample using tinker (via async gather)."""
    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        tokens = result.sequences[0].tokens
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params, show_progress: bool = False) -> list[str]:
    """Batch sample using vLLM."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return [r.sequences[0].text for r in results]


def run_batched_rsa(
    args,
    client,
    tokenizer,
    shared_dataset,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> list[RSAProblemState]:
    """Run batched RSA across all problems."""
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
    else:
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, "code", split="train")
    
    # Select slice
    end_problem = min(args.start_problem + args.num_problems, len(full_dataset))
    shared_dataset = full_dataset.select(range(args.start_problem, end_problem))
    actual_num_problems = len(shared_dataset)
    print(f"Dataset loaded with {len(full_dataset)} total problems.")
    print(f"Selected slice: problems {args.start_problem} to {end_problem - 1} ({actual_num_problems} problems)")
    
    if actual_num_problems < args.num_problems:
        print(f"Warning: Requested {args.num_problems} problems but only {actual_num_problems} available in slice.")
        args.num_problems = actual_num_problems
    
    population = args.population
    k = args.k
    steps = args.steps
    start_step = 1
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
    
    problem_states: list[RSAProblemState] = []
    
    # Resume from checkpoint if available
    if checkpoint_manager and checkpoint_manager.has_checkpoint():
        print(f"\nResuming from checkpoint: {checkpoint_manager.checkpoint_dir}")
        checkpoint_data = checkpoint_manager.load()
        
        for warning in checkpoint_manager.verify_args({
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "population": args.population,
            "k": args.k,
            "steps": args.steps,
            "dataset": args.dataset,
            "model": args.model,
        }):
            print(warning)
        
        start_step = checkpoint_data.current_round
        print(f"  Resuming from step {start_step}/{steps}")
        print(f"  Restoring {len(checkpoint_data.active_states_data)} problem states...")
        
        for state_data in checkpoint_data.active_states_data:
            state = deserialize_rsa_state(state_data)
            problem_states.append(state)
        
        print(f"  Successfully restored {len(problem_states)} problem states.")
    else:
        # Initialize problem states
        print(f"Initializing {args.num_problems} problems with population={population}...")
        for problem_idx in range(args.num_problems):
            env = IntellectCodeEnv(
                system_prompt="",
                dataset_name=args.dataset,
                problem_index=problem_idx,
                max_turns=1,
                dataset=shared_dataset,
                interaction_mode=False,
            )
            obs, info = env.reset()
            
            state = RSAProblemState(
                problem_index=problem_idx,
                question=obs,
                tests=env.tests,
            )
            problem_states.append(state)
    
    sampling_params = create_sampling_params(args, args.backend)
    
    # RSA Steps
    for step in range(start_step, steps + 1):
        is_initial = (step == 1)
        
        if is_initial:
            print(f"\n[RSA Step 1/{steps}] Generating initial population of {population} candidates per problem...")
            
            # Build initial prompts for all problems
            all_prompts = []
            prompt_to_problem = []  # Track which problem each prompt belongs to
            
            for state in problem_states:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": state.question}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer(prompt_text)["input_ids"]
                
                # Add N copies for population
                for _ in range(population):
                    all_prompts.append(input_ids)
                    prompt_to_problem.append(state.problem_index)
            
            print(f"  Total prompts to generate: {len(all_prompts)}")
            
            # Batch sample
            if args.backend == "tinker":
                all_responses = asyncio.run(sample_batch_tinker(client, all_prompts, sampling_params, tokenizer))
            else:
                all_responses = sample_batch_vllm(client, all_prompts, sampling_params, show_progress=args.vllm_multi_gpu)
            
            # Distribute responses to problem states
            response_idx = 0
            for state in problem_states:
                state.candidates = all_responses[response_idx:response_idx + population]
                state.candidates_by_step.append(list(state.candidates))
                state.aggregation_history = [[] for _ in range(population)]  # No aggregation for initial
                state.aggregation_history_by_step.append([[] for _ in range(population)])
                state.current_step = 1
                response_idx += population
            
            print(f"  Generated {len(all_responses)} initial candidates.")
        
        else:
            print(f"\n[RSA Step {step}/{steps}] Aggregating with K={k}...")
            
            # Build aggregation prompts
            all_prompts = []
            all_subsets = []  # Track which candidates were aggregated
            prompt_to_problem = []
            
            for state in problem_states:
                new_subsets = []
                for _ in range(population):
                    # Subsample K candidates without replacement
                    subset_indices = random.sample(range(len(state.candidates)), k)
                    subset = [state.candidates[idx] for idx in subset_indices]
                    new_subsets.append(subset_indices)
                    
                    # Build aggregation prompt
                    agg_prompt_text = aggregate_prompt(state.question, subset)
                    messages = [{"role": "user", "content": agg_prompt_text}]
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    input_ids = tokenizer(prompt_text)["input_ids"]
                    
                    all_prompts.append(input_ids)
                    prompt_to_problem.append(state.problem_index)
                
                all_subsets.append(new_subsets)
            
            print(f"  Total prompts to generate: {len(all_prompts)}")
            
            # Batch sample
            if args.backend == "tinker":
                all_responses = asyncio.run(sample_batch_tinker(client, all_prompts, sampling_params, tokenizer))
            else:
                all_responses = sample_batch_vllm(client, all_prompts, sampling_params, show_progress=args.vllm_multi_gpu)
            
            # Distribute responses to problem states
            response_idx = 0
            for state, subsets in zip(problem_states, all_subsets):
                state.candidates = all_responses[response_idx:response_idx + population]
                state.candidates_by_step.append(list(state.candidates))
                state.aggregation_history = subsets
                state.aggregation_history_by_step.append([list(s) for s in subsets])
                state.current_step = step
                response_idx += population
            
            print(f"  Generated {len(all_responses)} aggregated candidates.")
        
        # Evaluate current step candidates (required for per-step splits)
        print(f"\n  Evaluating step {step} candidates...")
        problem_states = evaluate_rsa_results(
            problem_states, args, shared_dataset, step_idx=step - 1
        )
        
        # Save checkpoint after each step
        if checkpoint_manager:
            checkpoint_manager.save(
                active_states_data=[serialize_rsa_state(s) for s in problem_states],
                completed_states_data=[],
                current_round=step + 1,
                total_rounds=steps,
            )
    
    return problem_states


def evaluate_rsa_results(
    problem_states: list[RSAProblemState],
    args,
    shared_dataset,
    step_idx: Optional[int] = None,
) -> list[RSAProblemState]:
    """Evaluate candidates in each problem's population.
    
    If step_idx is provided, evaluates state.candidates and stores rewards in
    state.rewards_by_step[step_idx]. Otherwise uses legacy state.rewards (deprecated).
    """
    print(f"  Evaluating {sum(len(s.candidates) for s in problem_states)} candidates...")
    
    # Build environments and responses for batch evaluation
    all_envs = []
    all_responses = []
    env_to_state_idx = []
    
    for state_idx, state in enumerate(problem_states):
        for cand_idx, candidate in enumerate(state.candidates):
            env = IntellectCodeEnv(
                system_prompt="",
                dataset_name=args.dataset,
                problem_index=state.problem_index,
                max_turns=1,
                dataset=shared_dataset,
                interaction_mode=False,
            )
            env.reset()
            env.has_interacted = True
            
            all_envs.append(env)
            all_responses.append(candidate)
            env_to_state_idx.append((state_idx, cand_idx))
    
    # Batch evaluate
    if args.fast_eval:
        results = step_batch(
            all_envs,
            all_responses,
            eval_workers=args.eval_workers,
            eval_batch_size=args.eval_batch_size,
            eval_timeout_s=args.eval_timeout_s,
            show_progress=True,
        )
    else:
        results = []
        for env, response in tqdm(zip(all_envs, all_responses), total=len(all_envs), desc="Evaluating"):
            _, reward, terminated, truncated, info = env.step(response)
            results.append((None, reward, terminated, truncated, info))
    
    # Distribute rewards back to states
    for (state_idx, cand_idx), (_, reward, _, _, _) in zip(env_to_state_idx, results):
        state = problem_states[state_idx]
        if step_idx is not None:
            # Store in rewards_by_step
            while len(state.rewards_by_step) <= step_idx:
                state.rewards_by_step.append([])
            if len(state.rewards_by_step[step_idx]) <= cand_idx:
                state.rewards_by_step[step_idx].extend(
                    [0.0] * (cand_idx + 1 - len(state.rewards_by_step[step_idx]))
                )
            state.rewards_by_step[step_idx][cand_idx] = reward
        else:
            # Legacy: store in rewards
            if len(state.rewards) <= cand_idx:
                state.rewards.extend([0.0] * (cand_idx + 1 - len(state.rewards)))
            state.rewards[cand_idx] = reward
    
    return problem_states


def backfill_missing_step_rewards(
    problem_states: list[RSAProblemState],
    args,
    shared_dataset,
) -> list[RSAProblemState]:
    """Re-evaluate any steps that have candidates but missing rewards (e.g. after resume from old checkpoint)."""
    for step_idx in range(max(len(s.candidates_by_step) for s in problem_states) if problem_states else 0):
        needs_eval = False
        for state in problem_states:
            if step_idx < len(state.candidates_by_step) and (
                step_idx >= len(state.rewards_by_step) or len(state.rewards_by_step[step_idx]) < len(state.candidates_by_step[step_idx])
            ):
                needs_eval = True
                break
        if not needs_eval:
            continue
        # Temporarily set candidates to this step's candidates, evaluate, store, restore
        orig_candidates = []
        for state in problem_states:
            orig_candidates.append(state.candidates)
            state.candidates = state.candidates_by_step[step_idx]
        print(f"\n  Backfilling evaluation for step {step_idx + 1}...")
        evaluate_rsa_results(problem_states, args, shared_dataset, step_idx=step_idx)
        for state, orig in zip(problem_states, orig_candidates):
            state.candidates = orig
    return problem_states


def main(args):
    print(f"=" * 60)
    print(f"Collecting trajectories with RSA (Recursive Self-Aggregation)")
    print(f"  Paper: https://arxiv.org/abs/2509.26626")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Problem range: {args.start_problem} to {args.start_problem + args.num_problems - 1} ({args.num_problems} problems)")
    print(f"  RSA params: N={args.population}, K={args.k}, T={args.steps}")
    print(f"  Output: {args.output_dir}")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print(f"=" * 60)
    
    # Validate K <= N
    if args.k > args.population:
        raise ValueError(f"K ({args.k}) cannot be greater than N ({args.population})")
    
    # Setup checkpoint manager
    if args.resume_from:
        checkpoint_dir = args.resume_from
        print(f"\nResuming from checkpoint directory: {checkpoint_dir}")
    else:
        checkpoint_dir = get_checkpoint_dir()
        print(f"\nCheckpoint directory: {checkpoint_dir}")
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        args_dict={
            "dataset": args.dataset,
            "model": args.model,
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "population": args.population,
            "k": args.k,
            "steps": args.steps,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    )
    
    # Initialize client
    print(f"\nInitializing {args.backend} client...")
    sampling_client = create_sampling_client(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load dataset for evaluation
    print(f"Loading dataset for evaluation...")
    if args.dataset.startswith("bicycleman15/"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
    else:
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, "code", split="train")
    
    end_problem = min(args.start_problem + args.num_problems, len(full_dataset))
    shared_dataset = full_dataset.select(range(args.start_problem, end_problem))
    
    # Run RSA (evaluation happens inside the loop after each step)
    problem_states = run_batched_rsa(
        args=args,
        client=sampling_client,
        tokenizer=tokenizer,
        shared_dataset=shared_dataset,
        checkpoint_manager=checkpoint_manager,
    )
    
    # Backfill any missing per-step rewards (e.g. when resuming from old checkpoint)
    problem_states = backfill_missing_step_rewards(problem_states, args, shared_dataset)
    
    # Build DatasetDict with one split per step (step_1, step_2, ..., step_T)
    splits_dict = {}
    metrics_per_step = {}
    
    for step_idx in range(args.steps):
        split_name = f"step_{step_idx + 1}"
        rows = []
        all_results = []
        
        for state in problem_states:
            if step_idx >= len(state.candidates_by_step):
                continue
            candidates = state.candidates_by_step[step_idx]
            rewards = state.rewards_by_step[step_idx] if step_idx < len(state.rewards_by_step) else [0.0] * len(candidates)
            agg_history = state.aggregation_history_by_step[step_idx] if step_idx < len(state.aggregation_history_by_step) else [[]] * len(candidates)
            
            problem_results = []
            for cand_idx, candidate in enumerate(candidates):
                reward = rewards[cand_idx] if cand_idx < len(rewards) else 0.0
                is_successful = reward > 0
                problem_results.append(is_successful)
                
                agg_from = agg_history[cand_idx] if cand_idx < len(agg_history) else []
                
                row = {
                    "problem_id": state.problem_index,
                    "candidate_id": cand_idx,
                    "question": state.question,
                    "response": candidate,
                    "final_reward": reward,
                    "is_successful": is_successful,
                    "rsa_step": step_idx + 1,
                    "aggregated_from": json.dumps(agg_from),
                    "tests": json.dumps(state.tests),
                    "rendered": render_trajectory(
                        state.question, candidate, reward,
                        step_idx + 1, args.steps, cand_idx, agg_from
                    ),
                }
                rows.append(row)
            all_results.append(problem_results)
        
        if rows:
            splits_dict[split_name] = Dataset.from_list(rows)
            # Compute metrics for this step
            pass_at_1 = compute_pass_at_k(all_results, k=1)
            pass_at_n = compute_pass_at_k(all_results, k=args.population)
            mean_acc = sum(r["final_reward"] for r in rows) / len(rows)
            metrics_per_step[split_name] = {
                "mean_accuracy": mean_acc,
                "pass_at_1": pass_at_1,
                f"pass_at_{args.population}": pass_at_n,
                "num_successful": sum(1 for r in rows if r["is_successful"]),
                "problems_solved": sum(1 for pr in all_results if any(pr)),
            }
    
    dataset = DatasetDict(splits_dict)
    
    # Print results per step
    print(f"\n{'=' * 60}")
    print(f"Results per step:")
    for split_name, m in metrics_per_step.items():
        print(f"  {split_name}: mean_acc={m['mean_accuracy']:.4f}, pass@1={m['pass_at_1']:.4f}, pass@{args.population}={m[f'pass_at_{args.population}']:.4f}")
    print(f"{'=' * 60}")
    
    # Use final step metrics for overall summary
    final_split = f"step_{args.steps}"
    final_metrics = metrics_per_step.get(final_split, {})
    if not final_metrics and metrics_per_step:
        final_split = list(metrics_per_step.keys())[-1]
        final_metrics = metrics_per_step[final_split]
    mean_acc = final_metrics.get("mean_accuracy", 0.0)
    pass_at_1 = final_metrics.get("pass_at_1", 0.0)
    pass_at_n = final_metrics.get(f"pass_at_{args.population}", 0.0)
    
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "start_problem": args.start_problem,
        "num_problems": args.num_problems,
        "population": args.population,
        "k": args.k,
        "steps": args.steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "splits": list(splits_dict.keys()),
        "metrics_per_step": metrics_per_step,
        "mean_accuracy_final_step": mean_acc,
        "pass_at_1_final_step": pass_at_1,
        f"pass_at_{args.population}_final_step": pass_at_n,
        "mode": "rsa",
        "method": "recursive_self_aggregation",
        "splits_note": f"Each split (step_1, ..., step_{args.steps}) contains N={args.population} candidates per problem for that RSA step, with evaluation done at that step.",
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved dataset (splits: {list(splits_dict.keys())}) to: {args.output_dir}")
    print(f"Saved metadata to: {metadata_path}")
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    total_rows = sum(len(splits_dict[s]) for s in splits_dict)
    summary = {
        **metadata,
        "total_rows_across_splits": total_rows,
        "num_successful_final_step": final_metrics.get("num_successful", 0),
        "problems_solved_final_step": final_metrics.get("problems_solved", 0),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_path}")
    
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return dataset, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect trajectories with RSA (Recursive Self-Aggregation)")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL"])
    parser.add_argument("--start-problem", type=int, default=0,
                        help="Starting problem index for dataset slicing (default: 0)")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (paper uses 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling (paper uses 1.0)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories_rsa")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    # RSA-specific parameters
    parser.add_argument("--population", type=int, default=4,
                        help="N: Population size per problem (default: 4, paper uses 16)")
    parser.add_argument("--k", type=int, default=2,
                        help="K: Aggregation set size (default: 2, paper uses 4)")
    parser.add_argument("--steps", type=int, default=3,
                        help="T: Number of RSA steps (default: 3, paper uses 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Checkpointing options
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. checkpoints/20260117_143052)")
    
    # Backend options
    parser.add_argument("--backend", type=str, default="vllm", choices=["tinker", "vllm"],
                        help="Inference backend: 'tinker' or 'vllm' (default: vllm)")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                        help="URL for vLLM server (e.g. http://localhost:8000). If not set, uses local vLLM.")
    parser.add_argument("--vllm-multi-gpu", action="store_true",
                        help="Launch one local vLLM server per GPU and shard prompts across them.")
    parser.add_argument("--vllm-gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for vLLM servers (default: all visible GPUs).")
    parser.add_argument("--vllm-server-base-port", type=int, default=8000,
                        help="Base port for vLLM servers; ports increment per GPU.")
    parser.add_argument("--vllm-server-host", type=str, default="127.0.0.1",
                        help="Host to bind vLLM servers (default: 127.0.0.1).")
    parser.add_argument("--vllm-server-startup-timeout-s", type=float, default=300.0,
                        help="Seconds to wait for vLLM servers to be ready.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for local vLLM or vLLM servers (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model context length for vLLM (default: None, uses model default)")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)),
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 1.0)")
    
    args = parser.parse_args()
    main(args)
