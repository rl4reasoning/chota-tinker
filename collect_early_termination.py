"""
Batched early termination inference on trajectory datasets.

Takes a collected trajectory dataset, truncates at a specific turn index,
adds a prompt asking the model to write final code, and evaluates the results.

Usage:
    python collect_early_termination.py \
        --trajectory-dataset anirudhb11/qwen3_4b_instruct_start_425_end_450_interations_10_turns \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --turn-index 3 \
        --fast-eval \
        --eval-workers 8 \
        --eval-batch-size 8 \
        --push-to-hub anirudhb11/early_term_turn_3

Multi-GPU:
    python collect_early_termination.py \
        --trajectory-dataset anirudhb11/qwen3_4b_instruct_start_425_end_450_interations_10_turns \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --vllm-multi-gpu \
        --vllm-gpu-ids 0,1 \
        --turn-index 3 \
        --fast-eval \
        --push-to-hub anirudhb11/early_term_turn_3
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Dict, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.fast_eval import _evaluate_code, EvalTask, evaluate_tasks
from utils.pass_at_k import compute_pass_at_k
from code_env.code_env.utils.deepcoder_utils import extract_code_from_model

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

from utils.vllm_multi_gpu import (
    resolve_vllm_gpu_ids,
    build_vllm_server_urls,
    launch_vllm_servers,
    wait_for_vllm_servers,
    register_vllm_shutdown,
)


FINAL_PROMPT = """STOP. Do NOT use <interact> anymore. Your interaction budget is exhausted.

You MUST now output your final solution code wrapped in ```python``` code blocks.

Based on all the information and debugging you have done so far, write your best solution now. The code must:
- Read inputs from stdin
- NOT hardcode any inputs
- Be wrapped in ```python``` delimiters

Output ONLY the final ```python``` code block. No more <interact> blocks allowed."""


def render_result(
    messages: List[Dict[str, str]],
    response: str,
    question: str,
    original_reward: float,
    new_reward: float,
    turn_index: int,
) -> str:
    """Render a result as a formatted string for debugging.
    
    Shows the full trajectory including:
    - All truncated messages (system, user question, assistant turns, observations)
    - The final prompt asking for code
    - The assistant's response
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
    lines.append(f"Turn Index: {turn_index} | Original Reward: {original_reward:.2f} | New Reward: {new_reward:.2f}")
    lines.append("=" * 80)
    
    # Show full messages (including the final prompt)
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")
    
    # Show the assistant's response
    lines.append(f"\n[ASSISTANT RESPONSE]\n{response}")
    
    return "\n".join(lines)


@dataclass
class EarlyTermState:
    """State for a single early termination inference."""
    row_idx: int
    problem_id: int
    trajectory_id: int
    question: str
    tests: dict
    original_reward: float
    truncated_messages: List[Dict[str, str]] = field(default_factory=list)
    response: Optional[str] = None
    new_reward: float = 0.0


def _parse_json_field(value: Any) -> Any:
    """Parse JSON field if it's a string, otherwise return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def truncate_messages_at_turn(messages: List[Dict[str, str]], turn_index: int) -> List[Dict[str, str]]:
    """
    Truncate messages up to a specific turn index (including the user observation).
    
    Messages structure:
    - [0]: system message
    - [1]: initial user message (problem/question)
    - [2]: assistant turn 1
    - [3]: user observation turn 1 (feedback from environment)
    - [4]: assistant turn 2
    - [5]: user observation turn 2 (feedback from environment)
    - ...
    
    Turn index 1 means keep up to and including user observation 1 (indices 0-3)
    Turn index 2 means keep up to and including user observation 2 (indices 0-5)
    """
    if turn_index < 1:
        raise ValueError("turn_index must be >= 1")
    
    # For turn_index N: keep 2*N + 2 messages
    num_messages_to_keep = 2 * turn_index + 2
    
    if len(messages) <= num_messages_to_keep:
        return messages
    
    return messages[:num_messages_to_keep]


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
            kwargs = {"gpu_memory_utilization": args.gpu_memory_utilization}
            if getattr(args, "max_model_len", None) is not None:
                kwargs["max_model_len"] = args.max_model_len
            return SamplingClient(args.model, **kwargs)


def create_sampling_params(args, backend: str):
    """Create sampling params for the chosen backend."""
    if backend == "tinker":
        return tinker_types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=[],  # No stop - we want full response
        )
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=[],
        )


async def sample_batch_tinker(client, prompts: List[List[int]], sampling_params, tokenizer) -> List[str]:
    """Batch sample using tinker (via async gather)."""
    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    
    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


def sample_batch_vllm(client, prompts: List[List[int]], sampling_params, show_progress: bool = False) -> List[str]:
    """Batch sample using vLLM."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return [r.sequences[0].text for r in results]


def run_early_termination_batch(
    args,
    client,
    tokenizer,
    sampling_params,
    dataset,
) -> List[EarlyTermState]:
    """
    Run early termination on all trajectories in the dataset.
    
    This is the main processing function that:
    1. Prepares states by truncating messages
    2. Runs batched inference
    3. Runs batched evaluation
    
    Returns list of EarlyTermState with results.
    """
    turn_index = args.turn_index
    
    # =========================================================================
    # Step 1: Prepare states (truncate messages)
    # =========================================================================
    print(f"\n[Step 1] Preparing states from {len(dataset)} trajectories...")
    
    states = []
    skipped = 0
    
    for idx, row in enumerate(tqdm(dataset, desc="Loading trajectories")):
        messages = _parse_json_field(row.get("messages", []))
        tests = _parse_json_field(row.get("tests", {}))
        
        # Check if we have enough turns
        max_turn = (len(messages) - 2) // 2
        if turn_index > max_turn:
            skipped += 1
            continue
        
        # Truncate messages
        truncated = truncate_messages_at_turn(messages, turn_index)
        
        # Add final prompt
        truncated.append({"role": "user", "content": FINAL_PROMPT})
        
        states.append(EarlyTermState(
            row_idx=idx,
            problem_id=row.get("problem_id", idx),
            trajectory_id=row.get("trajectory_id", 0),
            question=row.get("question", ""),
            tests=tests,
            original_reward=row.get("final_reward", 0.0),
            truncated_messages=truncated,
        ))
    
    if skipped > 0:
        print(f"  Skipped {skipped} rows with fewer than {turn_index} turns")
    print(f"  Prepared {len(states)} states for inference")
    
    if not states:
        print("No states to process!")
        return []
    
    # =========================================================================
    # Step 2: Build prompts and run inference
    # =========================================================================
    print(f"\n[Step 2] Running inference on {len(states)} states...")
    
    # Build prompts
    prompts = []
    for state in tqdm(states, desc="Building prompts"):
        prompt_text = tokenizer.apply_chat_template(
            state.truncated_messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(tokenizer(prompt_text)["input_ids"])
    
    # Run batched inference
    print(f"  Sampling responses...")
    if args.backend == "vllm":
        responses = sample_batch_vllm(
            client, prompts, sampling_params,
            show_progress=args.vllm_multi_gpu
        )
    else:
        responses = asyncio.run(sample_batch_tinker(client, prompts, sampling_params, tokenizer))
    
    # Store responses in states
    for state, response in zip(states, responses):
        state.response = response
    
    print(f"  Completed inference for {len(states)} states")
    
    # =========================================================================
    # Step 3: Run evaluation
    # =========================================================================
    print(f"\n[Step 3] Evaluating {len(states)} responses...")
    
    if args.fast_eval:
        # Build eval tasks
        eval_tasks = []
        for state in states:
            eval_tasks.append(EvalTask(
                response=state.response,
                tests=state.tests,
                max_tests=15,
                timeout_s=args.eval_timeout_s,
                require_solution_class=True,
            ))
        
        # Run parallel evaluation
        results = evaluate_tasks(
            eval_tasks,
            max_workers=args.eval_workers,
            batch_size=args.eval_batch_size,
            show_progress=True,
        )
        
        for state, result in zip(states, results):
            state.new_reward = result.reward
    else:
        # Sequential evaluation
        for state in tqdm(states, desc="Evaluating"):
            code = extract_code_from_model(state.response)
            if not code:
                state.new_reward = 0.0
            else:
                reward, _, _ = _evaluate_code(
                    code=code,
                    tests=state.tests,
                    max_tests=15,
                    timeout_s=args.eval_timeout_s,
                    timeout_record_limit=0,
                    require_solution_class=True,
                )
                state.new_reward = reward
    
    print(f"  Completed evaluation for {len(states)} states")
    
    return states


def main(args):
    print(f"=" * 60)
    print(f"Early Termination Batch Inference")
    print(f"=" * 60)
    print(f"  Trajectory Dataset: {args.trajectory_dataset}")
    print(f"  Split: {args.split}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Turn Index: {args.turn_index}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Fast Eval: {args.fast_eval}")
    if args.fast_eval:
        print(f"    Eval Workers: {args.eval_workers}")
        print(f"    Eval Batch Size: {args.eval_batch_size}")
        print(f"    Eval Timeout: {args.eval_timeout_s}s")
    print(f"  Output Dir: {args.output_dir}")
    if args.push_to_hub:
        print(f"  Push to Hub: {args.push_to_hub}")
    print(f"=" * 60)
    
    # =========================================================================
    # Load dataset
    # =========================================================================
    print(f"\nLoading trajectory dataset: {args.trajectory_dataset}")
    dataset = load_dataset(args.trajectory_dataset, split=args.split)
    print(f"  Loaded {len(dataset)} trajectories")
    
    # =========================================================================
    # Initialize client and tokenizer
    # =========================================================================
    print(f"\nInitializing {args.backend} client...")
    client = create_sampling_client(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = create_sampling_params(args, args.backend)
    
    # =========================================================================
    # Run batched processing
    # =========================================================================
    states = run_early_termination_batch(
        args=args,
        client=client,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        dataset=dataset,
    )
    
    if not states:
        print("\nNo results to save.")
        return None, None
    
    # =========================================================================
    # Compute metrics
    # =========================================================================
    # Group by problem_id for pass@k
    problem_results: Dict[int, List[bool]] = {}
    for state in states:
        pid = state.problem_id
        if pid not in problem_results:
            problem_results[pid] = []
        problem_results[pid].append(state.new_reward > 0)
    
    all_results = list(problem_results.values())
    pass_at_1 = compute_pass_at_k(all_results, k=1) if all_results else 0.0
    
    # Compute success rates
    total_successful = sum(1 for s in states if s.new_reward > 0)
    success_rate = total_successful / len(states) if states else 0
    
    original_successes = sum(1 for s in states if s.original_reward > 0)
    original_rate = original_successes / len(states) if states else 0
    
    # Count problems solved
    problems_solved_new = sum(1 for results in all_results if any(results))
    problems_solved_original = len(set(s.problem_id for s in states if s.original_reward > 0))
    
    print(f"\n{'=' * 60}")
    print(f"Results (Turn Index = {args.turn_index})")
    print(f"{'=' * 60}")
    print(f"  Total trajectories processed: {len(states)}")
    print(f"  Unique problems: {len(problem_results)}")
    print(f"  ")
    print(f"  Original success rate: {original_rate:.4f} ({original_successes}/{len(states)})")
    print(f"  New success rate:      {success_rate:.4f} ({total_successful}/{len(states)})")
    print(f"  ")
    print(f"  Problems solved (original): {problems_solved_original}/{len(problem_results)}")
    print(f"  Problems solved (new):      {problems_solved_new}/{len(problem_results)}")
    print(f"  ")
    print(f"  pass@1: {pass_at_1:.4f}")
    print(f"{'=' * 60}")
    
    # =========================================================================
    # Build output rows
    # =========================================================================
    rows = []
    for state in states:
        # Build final messages with response
        final_messages = state.truncated_messages.copy()
        final_messages.append({"role": "assistant", "content": state.response or ""})
        
        rows.append({
            "problem_id": state.problem_id,
            "trajectory_id": state.trajectory_id,
            "question": state.question,
            "messages": json.dumps(final_messages),
            "truncated_at_turn": args.turn_index,
            "response": state.response or "",
            "original_reward": state.original_reward,
            "new_reward": state.new_reward,
            "is_successful": state.new_reward > 0,
            "tests": json.dumps(state.tests),
            "rendered": render_result(
                state.truncated_messages, state.response or "", state.question,
                state.original_reward, state.new_reward, args.turn_index
            ),
        })
    
    output_dataset = Dataset.from_list(rows)
    
    # =========================================================================
    # Save metadata
    # =========================================================================
    metadata = {
        "trajectory_dataset": args.trajectory_dataset,
        "split": args.split,
        "model": args.model,
        "turn_index": args.turn_index,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "total_trajectories": len(states),
        "unique_problems": len(problem_results),
        "original_success_rate": original_rate,
        "new_success_rate": success_rate,
        "problems_solved_original": problems_solved_original,
        "problems_solved_new": problems_solved_new,
        "pass_at_1": pass_at_1,
    }
    
    # =========================================================================
    # Save locally
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    output_dataset.save_to_disk(args.output_dir)
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        **metadata,
        "num_successful_trajectories": total_successful,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved dataset to: {args.output_dir}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved summary to: {summary_path}")
    
    # =========================================================================
    # Push to hub
    # =========================================================================
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        output_dataset.push_to_hub(
            args.push_to_hub,
            private=False,
            commit_message=f"Early termination at turn {args.turn_index}, model={args.model}"
        )
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return output_dataset, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched early termination inference on trajectory datasets")
    
    # Dataset args
    parser.add_argument("--trajectory-dataset", type=str, required=True,
                        help="HuggingFace dataset with trajectory data (messages, tests, etc.)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--turn-index", type=int, required=True,
                        help="Turn index to truncate at (1-indexed)")
    
    # Model args
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    
    # Output args
    parser.add_argument("--output-dir", type=str, default="artifacts/early_termination",
                        help="Directory to save output dataset")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF repo to push to (e.g. username/repo-name)")
    
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
                        help="Max sequence length for vLLM; lower than model default to reduce KV cache")
    
    # Eval options
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel evaluator workers (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 1.0)")
    
    args = parser.parse_args()
    main(args)
