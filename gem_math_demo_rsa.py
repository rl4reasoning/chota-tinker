"""
RSA (Recursive Self-Aggregation) demo implementing test-time scaling.

This implements the RSA algorithm from:
    Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models
    (https://arxiv.org/abs/2509.26626)

RSA is a hybrid test-time scaling method that combines parallel and sequential
scaling by iteratively refining a population of candidate solutions through
aggregation. Each step samples K candidates from the population and asks the
model to combine/refine them into an improved solution.

Algorithm:
    1. Initialize: Generate N candidate solutions for the query
    2. For T-1 iterations:
       - For each of N new candidates, sample K solutions from current population
       - Generate improved solution conditioned on the K candidates + query
    3. Return: Final population (sample uniformly for Pass@1)

Usage:
    # With vLLM backend (default)
    python gem_math_demo_rsa.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --population 4 \
        --k 2 \
        --steps 3 \
        --difficulty easy_medium \
        --problem_index 0

    # With tinker backend
    python gem_math_demo_rsa.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend tinker \
        --population 4 \
        --k 2 \
        --steps 3 \
        --difficulty easy_medium \
        --problem_index 0
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import random
from typing import Optional

from transformers import AutoTokenizer

from intellect_env import IntellectCodeEnv
from utils.fast_eval import EvalTask, evaluate_tasks

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
        SamplingParams,
        ModelInput,
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# System prompt for initial generation
SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""


def aggregate_prompt(question: str, candidates: list[str]) -> str:
    """
    Build aggregation prompt from question and K candidate solutions.
    
    Adapted from RSA paper Appendix F for code problems.
    
    Args:
        question: The original coding problem
        candidates: List of K candidate solutions to aggregate
        
    Returns:
        Formatted prompt asking model to aggregate/refine solutions
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


def build_initial_prompt(question: str) -> str:
    """Build prompt for initial generation (no candidates yet)."""
    return question


def build_chat_messages(prompt: str, use_system: bool = True) -> list[dict]:
    """Build chat messages for the model."""
    messages = []
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    return messages


def tokenize_messages(messages: list[dict], tokenizer) -> list[int]:
    """Tokenize chat messages using the tokenizer's chat template."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


# -----------------------------------------------------------------------------
# Tinker backend functions
# -----------------------------------------------------------------------------

async def sample_one_tinker(
    input_ids: list[int],
    client,
    sampling_params,
    tokenizer,
) -> str:
    """Sample a single response using tinker backend."""
    result = await client.sample_async(
        prompt=tinker_types.ModelInput.from_ints(input_ids),
        sampling_params=sampling_params,
        num_samples=1,
    )
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


async def sample_batch_tinker(
    prompts: list[list[int]],
    client,
    sampling_params,
    tokenizer,
) -> list[str]:
    """Sample multiple responses in parallel using tinker backend."""
    tasks = [
        sample_one_tinker(p, client, sampling_params, tokenizer)
        for p in prompts
    ]
    return await asyncio.gather(*tasks)


# -----------------------------------------------------------------------------
# vLLM backend functions
# -----------------------------------------------------------------------------

def sample_batch_vllm(
    prompts: list[list[int]],
    client: SamplingClient,
    sampling_params: SamplingParams,
) -> list[str]:
    """Sample multiple responses using vLLM backend."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return [r.sequences[0].text for r in results]


# -----------------------------------------------------------------------------
# RSA Algorithm
# -----------------------------------------------------------------------------

async def run_rsa_tinker(
    question: str,
    tokenizer,
    client,
    sampling_params,
    population: int,
    k: int,
    steps: int,
    verbose: bool = True,
    show_raw_outputs: bool = True,
) -> list[str]:
    """
    Run RSA algorithm using tinker backend.
    
    Args:
        question: The coding problem
        tokenizer: Tokenizer for the model
        client: Tinker sampling client
        sampling_params: Sampling parameters
        population: N - number of candidates in population
        k: K - number of candidates to aggregate per generation
        steps: T - number of RSA steps
        verbose: Whether to print progress
        show_raw_outputs: Whether to print raw inputs/outputs at every step
        
    Returns:
        Final population of candidate solutions
    """
    if verbose:
        print(f"[RSA] Starting with N={population}, K={k}, T={steps}")
    
    # Step 1: Initialize population P_1
    if verbose:
        print(f"[RSA] Step 1/{steps}: Initializing population with {population} candidates...")
    
    initial_prompt = build_initial_prompt(question)
    messages = build_chat_messages(initial_prompt, use_system=True)
    input_ids = tokenize_messages(messages, tokenizer)
    full_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate N initial candidates
    prompts = [input_ids] * population
    candidates = await sample_batch_tinker(prompts, client, sampling_params, tokenizer)
    
    if verbose:
        print(f"[RSA] Initialized {len(candidates)} candidates")
    
    # Show input-output pairs for initial generation
    if show_raw_outputs:
        print("\n" + "=" * 80)
        print(f"STEP 1/{steps} - Initial Generation")
        print("=" * 80)
        for i, cand in enumerate(candidates):
            print(f"\n{'#' * 80}")
            print(f"# CANDIDATE {i+1}/{population}")
            print(f"{'#' * 80}")
            print("\n[INPUT PROMPT]")
            print(full_prompt_text)
            print("\n[OUTPUT]")
            print(cand)
            print("-" * 80)
    
    # Steps 2 to T: Iterative aggregation
    for t in range(2, steps + 1):
        if verbose:
            print(f"\n[RSA] Step {t}/{steps}: Aggregating with K={k}...")
        
        agg_prompts = []
        agg_prompts_text = []  # Store text versions for display
        subsets_used = []  # Track which candidates were aggregated
        
        for i in range(population):
            # Subsample K candidates without replacement
            subset_indices = random.sample(range(len(candidates)), k)
            subset = [candidates[idx] for idx in subset_indices]
            subsets_used.append(subset_indices)
            # Build aggregation prompt
            agg_prompt_text = aggregate_prompt(question, subset)
            messages = build_chat_messages(agg_prompt_text, use_system=False)
            input_ids = tokenize_messages(messages, tokenizer)
            agg_prompts.append(input_ids)
            # Store full prompt text for display
            full_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            agg_prompts_text.append(full_prompt_text)
        
        # Generate all N new candidates in parallel
        new_candidates = await sample_batch_tinker(agg_prompts, client, sampling_params, tokenizer)
        candidates = new_candidates
        
        if verbose:
            print(f"[RSA] Generated {len(candidates)} aggregated candidates")
        
        # Show input-output pairs for this aggregation step
        if show_raw_outputs:
            print("\n" + "=" * 80)
            print(f"STEP {t}/{steps} - Aggregation")
            print("=" * 80)
            for i, (prompt_text, cand) in enumerate(zip(agg_prompts_text, candidates)):
                print(f"\n{'#' * 80}")
                print(f"# CANDIDATE {i+1}/{population} (aggregated from indices {subsets_used[i]})")
                print(f"{'#' * 80}")
                print("\n[INPUT PROMPT]")
                print(prompt_text)
                print("\n[OUTPUT]")
                print(cand)
                print("-" * 80)
    
    return candidates


def run_rsa_vllm(
    question: str,
    tokenizer,
    client: SamplingClient,
    sampling_params: SamplingParams,
    population: int,
    k: int,
    steps: int,
    verbose: bool = True,
    show_raw_outputs: bool = True,
) -> list[str]:
    """
    Run RSA algorithm using vLLM backend.
    
    Args:
        question: The coding problem
        tokenizer: Tokenizer for the model
        client: vLLM sampling client
        sampling_params: Sampling parameters
        population: N - number of candidates in population
        k: K - number of candidates to aggregate per generation
        steps: T - number of RSA steps
        verbose: Whether to print progress
        show_raw_outputs: Whether to print raw inputs/outputs at every step
        
    Returns:
        Final population of candidate solutions
    """
    if verbose:
        print(f"[RSA] Starting with N={population}, K={k}, T={steps}")
    
    # Step 1: Initialize population P_1
    if verbose:
        print(f"[RSA] Step 1/{steps}: Initializing population with {population} candidates...")
    
    initial_prompt = build_initial_prompt(question)
    messages = build_chat_messages(initial_prompt, use_system=True)
    input_ids = tokenize_messages(messages, tokenizer)
    full_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate N initial candidates
    prompts = [input_ids] * population
    candidates = sample_batch_vllm(prompts, client, sampling_params)
    
    if verbose:
        print(f"[RSA] Initialized {len(candidates)} candidates")
    
    # Show input-output pairs for initial generation
    if show_raw_outputs:
        print("\n" + "=" * 80)
        print(f"STEP 1/{steps} - Initial Generation")
        print("=" * 80)
        for i, cand in enumerate(candidates):
            print(f"\n{'#' * 80}")
            print(f"# CANDIDATE {i+1}/{population}")
            print(f"{'#' * 80}")
            print("\n[INPUT PROMPT]")
            print(full_prompt_text)
            print("\n[OUTPUT]")
            print(cand)
            print("-" * 80)
    
    # Steps 2 to T: Iterative aggregation
    for t in range(2, steps + 1):
        if verbose:
            print(f"\n[RSA] Step {t}/{steps}: Aggregating with K={k}...")
        
        agg_prompts = []
        agg_prompts_text = []  # Store text versions for display
        subsets_used = []  # Track which candidates were aggregated
        
        for i in range(population):
            # Subsample K candidates without replacement
            subset_indices = random.sample(range(len(candidates)), k)
            subset = [candidates[idx] for idx in subset_indices]
            subsets_used.append(subset_indices)
            # Build aggregation prompt
            agg_prompt_text = aggregate_prompt(question, subset)
            messages = build_chat_messages(agg_prompt_text, use_system=False)
            input_ids = tokenize_messages(messages, tokenizer)
            agg_prompts.append(input_ids)
            # Store full prompt text for display
            full_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            agg_prompts_text.append(full_prompt_text)
        
        # Generate all N new candidates in batch
        candidates = sample_batch_vllm(agg_prompts, client, sampling_params)
        
        if verbose:
            print(f"[RSA] Generated {len(candidates)} aggregated candidates")
        
        # Show input-output pairs for this aggregation step
        if show_raw_outputs:
            print("\n" + "=" * 80)
            print(f"STEP {t}/{steps} - Aggregation")
            print("=" * 80)
            for i, (prompt_text, cand) in enumerate(zip(agg_prompts_text, candidates)):
                print(f"\n{'#' * 80}")
                print(f"# CANDIDATE {i+1}/{population} (aggregated from indices {subsets_used[i]})")
                print(f"{'#' * 80}")
                print("\n[INPUT PROMPT]")
                print(prompt_text)
                print("\n[OUTPUT]")
                print(cand)
                print("-" * 80)
    
    return candidates


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_candidates(
    candidates: list[str],
    tests: dict,
    max_tests: int = 15,
    timeout_s: float = 1.0,
) -> tuple[float, float, list[float]]:
    """
    Evaluate all candidates and compute metrics.
    
    Args:
        candidates: List of candidate solutions
        tests: Test cases from the environment
        max_tests: Maximum number of tests to run
        timeout_s: Timeout per test in seconds
        
    Returns:
        (mean_accuracy, pass_at_n, individual_rewards)
    """
    eval_tasks = [
        EvalTask(
            response=cand,
            tests=tests,
            max_tests=max_tests,
            timeout_s=timeout_s,
            require_solution_class=True,
        )
        for cand in candidates
    ]
    
    results = evaluate_tasks(
        eval_tasks,
        max_workers=min(len(eval_tasks), 8),
        batch_size=1,
        show_progress=False,
    )
    
    rewards = [r.reward for r in results]
    mean_acc = sum(rewards) / len(rewards) if rewards else 0.0
    pass_at_n = 1.0 if any(r > 0 for r in rewards) else 0.0
    
    return mean_acc, pass_at_n, rewards


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# Dataset mapping for difficulty levels
DATASET_MAP = {
    "original": "PrimeIntellect/INTELLECT-3-RL",
    "easy_medium": "bicycleman15/intellect_3_code_easy_medium",
    "hard": "bicycleman15/intellect_3_code_hard",
    "very_hard": "bicycleman15/intellect_3_code_very_hard",
}


def create_sampling_client(args):
    """Create sampling client based on backend choice."""
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        service_client = tinker.ServiceClient()
        return service_client.create_sampling_client(base_model=args.model)
    else:  # vllm
        if not VLLM_AVAILABLE:
            raise ImportError("chota_tinker not installed. Install it or use --backend tinker")
        return SamplingClient(args.model, gpu_memory_utilization=args.gpu_memory_utilization)


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


async def main_tinker(args):
    """Main function for tinker backend."""
    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_params = tinker_types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    dataset_name = DATASET_MAP[args.difficulty]
    print(f"Using dataset: {dataset_name} (difficulty: {args.difficulty})")
    print(f"RSA parameters: N={args.population}, K={args.k}, T={args.steps}")
    if args.problem_index is not None:
        print(f"Using problem index: {args.problem_index}")
    print()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
    
    env = IntellectCodeEnv(
        system_prompt="",
        max_turns=1,
        dataset_name=dataset_name,
        problem_index=args.problem_index,
        interaction_mode=False,
    )
    
    obs, info = env.reset()
    env.has_interacted = True  # Single-turn mode
    
    print(f"[Problem]\n{obs[:500]}..." if len(obs) > 500 else f"[Problem]\n{obs}")
    print()
    
    # Run RSA
    show_raw = args.show_raw_outputs and not args.no_raw_outputs
    candidates = await run_rsa_tinker(
        question=obs,
        tokenizer=tokenizer,
        client=client,
        sampling_params=sampling_params,
        population=args.population,
        k=args.k,
        steps=args.steps,
        verbose=True,
        show_raw_outputs=show_raw,
    )
    
    # Evaluate candidates
    print("\n[Evaluating candidates...]")
    mean_acc, pass_at_n, rewards = evaluate_candidates(
        candidates=candidates,
        tests=env.tests,
        max_tests=env.max_tests,
        timeout_s=1.0,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RSA Results:")
    print(f"  Population size (N): {args.population}")
    print(f"  Aggregation size (K): {args.k}")
    print(f"  Steps (T): {args.steps}")
    print(f"  Mean accuracy: {mean_acc:.3f}")
    print(f"  Pass@{args.population}: {pass_at_n:.3f}")
    print(f"  Individual rewards: {rewards}")
    print("=" * 60)
    
    # Print a sample solution (the first successful one, or just the first)
    best_idx = next((i for i, r in enumerate(rewards) if r > 0), 0)
    print(f"\n[Best candidate (idx={best_idx}, reward={rewards[best_idx]:.3f})]")
    print(candidates[best_idx][:2000] if len(candidates[best_idx]) > 2000 else candidates[best_idx])
    
    return mean_acc, pass_at_n


def main_vllm(args):
    """Main function for vLLM backend."""
    client = SamplingClient(args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    dataset_name = DATASET_MAP[args.difficulty]
    print(f"Using dataset: {dataset_name} (difficulty: {args.difficulty})")
    print(f"RSA parameters: N={args.population}, K={args.k}, T={args.steps}")
    if args.problem_index is not None:
        print(f"Using problem index: {args.problem_index}")
    print()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
    
    env = IntellectCodeEnv(
        system_prompt="",
        max_turns=1,
        dataset_name=dataset_name,
        problem_index=args.problem_index,
        interaction_mode=False,
    )
    
    obs, info = env.reset()
    env.has_interacted = True  # Single-turn mode
    
    print(f"[Problem]\n{obs[:500]}..." if len(obs) > 500 else f"[Problem]\n{obs}")
    print()
    
    # Run RSA
    show_raw = args.show_raw_outputs and not args.no_raw_outputs
    candidates = run_rsa_vllm(
        question=obs,
        tokenizer=tokenizer,
        client=client,
        sampling_params=sampling_params,
        population=args.population,
        k=args.k,
        steps=args.steps,
        verbose=True,
        show_raw_outputs=show_raw,
    )
    
    # Evaluate candidates
    print("\n[Evaluating candidates...]")
    mean_acc, pass_at_n, rewards = evaluate_candidates(
        candidates=candidates,
        tests=env.tests,
        max_tests=env.max_tests,
        timeout_s=1.0,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RSA Results:")
    print(f"  Population size (N): {args.population}")
    print(f"  Aggregation size (K): {args.k}")
    print(f"  Steps (T): {args.steps}")
    print(f"  Mean accuracy: {mean_acc:.3f}")
    print(f"  Pass@{args.population}: {pass_at_n:.3f}")
    print(f"  Individual rewards: {rewards}")
    print("=" * 60)
    
    # Print a sample solution (the first successful one, or just the first)
    best_idx = next((i for i, r in enumerate(rewards) if r > 0), 0)
    print(f"\n[Best candidate (idx={best_idx}, reward={rewards[best_idx]:.3f})]")
    print(candidates[best_idx][:2000] if len(candidates[best_idx]) > 2000 else candidates[best_idx])
    
    return mean_acc, pass_at_n


def main():
    parser = argparse.ArgumentParser(
        description="Run coding problems with RSA (Recursive Self-Aggregation) test-time scaling"
    )
    
    # Model and backend
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name or path")
    parser.add_argument("--backend", type=str, default="vllm", choices=["tinker", "vllm"],
                        help="Inference backend: 'tinker' or 'vllm' (default: vllm)")
    
    # RSA hyperparameters (from paper)
    parser.add_argument("--population", type=int, default=4,
                        help="N: Population size (default: 4, paper uses 16)")
    parser.add_argument("--k", type=int, default=2,
                        help="K: Aggregation set size (default: 2, paper uses 4)")
    parser.add_argument("--steps", type=int, default=3,
                        help="T: Number of RSA steps (default: 3, paper uses 10)")
    
    # Sampling parameters (paper uses temp=1.0, top_p=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (paper uses 1.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling (paper uses 1.0)")
    
    # Problem selection
    parser.add_argument("--difficulty", type=str, default="easy_medium",
                        choices=["original", "easy_medium", "hard", "very_hard"],
                        help="Problem difficulty level")
    parser.add_argument("--problem_index", type=int, default=None,
                        help="Specific problem index to use")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU memory utilization for vLLM (default: 0.8)")
    parser.add_argument("--show_raw_outputs", action="store_true", default=True,
                        help="Show raw outputs at every RSA step (default: True)")
    parser.add_argument("--no_raw_outputs", action="store_true",
                        help="Disable showing raw outputs at every step")
    
    args = parser.parse_args()
    
    # Validate K <= N
    if args.k > args.population:
        parser.error(f"K ({args.k}) cannot be greater than N ({args.population})")
    
    print("=" * 60)
    print("RSA (Recursive Self-Aggregation) Demo")
    print("Paper: https://arxiv.org/abs/2509.26626")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"RSA params: N={args.population}, K={args.k}, T={args.steps}")
    print("=" * 60)
    print()
    
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        asyncio.run(main_tinker(args))
    else:
        if not VLLM_AVAILABLE:
            raise ImportError("chota_tinker not installed. Install it or use --backend tinker")
        main_vllm(args)


if __name__ == "__main__":
    main()
