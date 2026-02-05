"""Collect multi-turn trajectories using Tinker and save as HuggingFace dataset.

Usage:
    python collect_trajectories_tinker.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --num-problems 10 \
    --num-samples 4 \
    --max-turns 10 \
    \
    --fast-eval \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 1.0 \
    --push-to-hub bicycleman15/temp

Resume from checkpoint (if previous run failed):
    python collect_trajectories_tinker.py \
        --resume-from checkpoints/20260117_143052 \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        ... (same args as original run)

Checkpoints are automatically saved after each generation round to:
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint.pkl
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint_info.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from checkpoint import CheckpointManager, get_checkpoint_dir
from intellect_env import IntellectCodeEnv, step_batch
from utils.pass_at_k import compute_pass_at_k

# Tinker imports
try:
    import tinker
    from tinker import types as tinker_types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


def render_trajectory(messages: list[dict], interactions: list[dict], question: str, reward: float, num_turns: int, terminated: bool, truncated: bool) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.2f} | Turns: {num_turns} | Terminated: {terminated} | Truncated: {truncated}")
    lines.append("=" * 80)
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")
    
    if interactions:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Code Interactions ({len(interactions)}):")
        for i, inter in enumerate(interactions):
            lines.append(f"  [{i+1}] Code:\n{inter['code']}")
            lines.append(f"  Output:\n{inter['output']}")
    
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a helpful coding assistant.

You have access to a Python interpreter.
To execute code, wrap it inside <interact></interact>. I will run it and return stdout/stderr in a subsequent turn.

IMPORTANT CONTEXT:
- This is a multi-turn conversation.
- The Python interpreter is a tool for gathering evidence: testing hypotheses, validating assumptions, checking edge cases, and falsifying incorrect reasoning.
- You should interact only when doing so provides information that can change your understanding, reasoning, or final decision.

────────────────────────
HARD RULES (NON-NEGOTIABLE)
────────────────────────
- BEFORE you output ANY final solution code in a ```python``` block, you MUST have completed at least one successful <interact></interact> execution in an earlier turn.
- If you have NOT yet completed a successful <interact></interact>, you are FORBIDDEN from outputting any ```python``` code block (even partial solutions).
- In the FIRST assistant response after receiving a new coding problem, you MUST perform an <interact></interact> intended to test, validate, or falsify some part of your reasoning.
- Interactions performed solely to satisfy this requirement (without testing a hypothesis or reducing uncertainty) are INVALID.

────────────────────────
EXECUTION ENVIRONMENT (CRITICAL)
────────────────────────
- The execution environment shows ONLY what you PRINT to stdout.
- EVERY <interact></interact> MUST include explicit print(...) statements.
- Do NOT rely on REPL-style expression outputs or implicit returns.

────────────────────────
DEFINITION OF "SUCCESSFUL <interact>"
────────────────────────
An interaction is successful ONLY if ALL of the following hold:
- The code executes without exceptions, AND
- It prints at least 2 lines of task-relevant evidence, AND
- At least one printed line is a newly computed result (not already given in the prompt), AND
- The subsequent assistant message explicitly uses this evidence to confirm, revise, or reject a stated hypothesis.

────────────────────────
MANDATORY INTERACTION STRUCTURE
────────────────────────
Before each <interact></interact>, you MUST clearly state:
- The specific hypothesis, assumption, or uncertainty being tested
- Why this cannot be fully resolved by reasoning alone
- What outcome you expect if the hypothesis is correct vs incorrect

After receiving the output, you MUST clearly state:
- What the output shows (summarize or quote key lines)
- Whether the hypothesis was confirmed, weakened, or falsified
- What (if anything) changed in your approach

────────────────────────
SOLUTION STRESS TEST (CRITICAL)
────────────────────────
- For algorithmic correctness problems, you MUST run at least one interaction that attempts to falsify your proposed solution.
- This interaction MUST compare your approach against a correct reference implementation using:
  (a) brute force / exhaustive checking for small inputs (e.g., n ≤ 6–8), OR
  (b) randomized testing against a slower but correct oracle.
- This interaction MUST print either:
  • "No counterexample found in K tests" (K ≥ 100), OR
  • A concrete counterexample where your approach disagrees with the oracle.
- If a counterexample is found, you MUST revise your approach and repeat the oracle test.

Testing only the examples provided in the prompt does NOT count as validation or falsification.

────────────────────────
ANTI-THRASHING RULE
────────────────────────
- If an <interact></interact> produces no output, insufficient output, or redundant output, your NEXT interaction MUST fix this and MUST NOT repeat the same interaction pattern.

────────────────────────
ITERATIVE WORKFLOW
────────────────────────
1. State your approach and any assumptions or uncertainties.
2. Use <interact></interact> to gather evidence addressing those uncertainties.
3. Update your reasoning based on the evidence.
4. Repeat steps 2–3 if meaningful uncertainty remains.
5. ONLY when no critical uncertainty remains, produce the final solution.

────────────────────────
FINAL CODE REQUIREMENTS
────────────────────────
- The final code MUST be inside a ```python``` code block.
- The final code MUST read inputs from stdin and MUST NOT hardcode inputs.
- The final answer MUST clearly depend on interaction-generated evidence.
- Do NOT include <interact></interact> blocks after the final code.
"""


@dataclass
class RolloutState:
    """Track state of a single rollout for batched processing."""
    problem_index: int
    sample_index: int
    env: IntellectCodeEnv
    history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    obs: str = ""
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    interaction_timeout_count: int = 0  # Number of interactions that timed out
    eval_timeout_count: int = 0  # Number of test cases that timed out during final eval


def serialize_rollout_state(state: RolloutState) -> dict:
    """Serialize a RolloutState to a dictionary for checkpointing."""
    return {
        "problem_index": state.problem_index,
        "sample_index": state.sample_index,
        "history": [msg.copy() for msg in state.history],
        "messages": [msg.copy() for msg in state.messages],
        "interactions": [inter.copy() for inter in state.interactions],
        "total_reward": state.total_reward,
        "obs": state.obs,
        "done": state.done,
        "terminated": state.terminated,
        "truncated": state.truncated,
        "interaction_timeout_count": state.interaction_timeout_count,
        "eval_timeout_count": state.eval_timeout_count,
        # Env-related data needed for reconstruction
        "question": state.env.question,
        "tests": state.env.tests,
        "current_turn": state.env.current_turn,
        "env_history": state.env.history.copy() if state.env.history else [],
        "has_interacted": state.env.has_interacted,
    }


def deserialize_rollout_state(data: dict, shared_dataset, args) -> RolloutState:
    """Deserialize a dictionary back to a RolloutState."""
    env = IntellectCodeEnv(
        system_prompt="",
        dataset_name=args.dataset,
        problem_index=data["problem_index"],
        max_turns=args.max_turns,
        dataset=shared_dataset,
    )
    env.reset()
    # Restore env state
    env.current_turn = data["current_turn"]
    env.history = data["env_history"]
    env.has_interacted = data["has_interacted"]
    
    state = RolloutState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        env=env,
        history=data["history"],
        messages=data["messages"],
        interactions=data["interactions"],
        total_reward=data["total_reward"],
        obs=data["obs"],
        done=data["done"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        interaction_timeout_count=data.get("interaction_timeout_count", 0),
        eval_timeout_count=data.get("eval_timeout_count", 0),
    )
    return state


def create_sampling_client(args):
    """Create tinker sampling client."""
    if not TINKER_AVAILABLE:
        raise ImportError("tinker not installed. Please install tinker to use this script.")
    service_client = tinker.ServiceClient()
    return service_client.create_sampling_client(base_model=args.model)


def create_sampling_params(args):
    """Create sampling params for tinker."""
    return tinker_types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        stop=["</interact>"],
    )


def build_prompt(state: RolloutState, tokenizer) -> list[int]:
    """Build tokenized prompt from rollout state."""
    messages = state.history + [{"role": "user", "content": state.obs}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


def postprocess_response(response: str) -> str:
    """Postprocess LLM response."""
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    return response


def truncate_interaction_output(
    output_text: str,
    tokenizer,
    max_tokens: int,
) -> str:
    """If output_text has more than max_tokens tokens, keep only the last max_tokens and prepend a notice."""
    if not output_text or max_tokens <= 0:
        return output_text
    ids = tokenizer.encode(output_text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return output_text
    keep = ids[-max_tokens:]
    tail_text = tokenizer.decode(keep, skip_special_tokens=False)
    return f"Output too long, showing only last {max_tokens} tokens.\n{tail_text}"


def maybe_truncate_obs(obs: str, tokenizer, max_interaction_output_tokens: Optional[int]) -> str:
    """If obs is <output>...</output> and body exceeds max_interaction_output_tokens, truncate to tail and prepend notice."""
    if obs is None or not obs or max_interaction_output_tokens is None or max_interaction_output_tokens <= 0:
        return obs
    match = re.match(r"^<output>\n(.*)</output>\s*$", obs, re.DOTALL)
    if not match:
        return obs
    body = match.group(1)
    truncated = truncate_interaction_output(body, tokenizer, max_interaction_output_tokens)
    if truncated is body:
        return obs
    return f"<output>\n{truncated}</output>"


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params) -> list[str]:
    """Batch sample using tinker (via async gather)."""
    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        return result.sequences[0].tokens
    
    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


def run_batched_rollouts(
    args,
    client,
    tokenizer,
    sampling_params,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> list[list[dict[str, Any]]]:
    """Run batched rollouts across all problems and samples."""
    # Load dataset ONCE before creating environments
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
    elif args.dataset.__contains__('lcb'):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="test")
    else:
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, "code", split="train")
    
    # Select slice based on start_problem and num_problems
    end_problem = min(args.start_problem + args.num_problems, len(full_dataset))
    shared_dataset = full_dataset.select(range(args.start_problem, end_problem))
    actual_num_problems = len(shared_dataset)
    print(f"Dataset loaded with {len(full_dataset)} total problems.")
    print(f"Selected slice: problems {args.start_problem} to {end_problem - 1} ({actual_num_problems} problems)")
    
    # Update args.num_problems to reflect actual slice size
    if actual_num_problems < args.num_problems:
        print(f"Warning: Requested {args.num_problems} problems but only {actual_num_problems} available in slice.")
        args.num_problems = actual_num_problems
    
    # Track generation round for checkpointing
    generation_round = 0
    active_states: list[RolloutState] = []
    completed_states: list[RolloutState] = []
    skip_generation = False  # Flag to skip generation on first iteration after resume
    
    # Resume from checkpoint if available
    if checkpoint_manager and checkpoint_manager.has_checkpoint():
        print(f"\nResuming from checkpoint: {checkpoint_manager.checkpoint_dir}")
        checkpoint_data = checkpoint_manager.load()
        
        # Verify args match
        for warning in checkpoint_manager.verify_args({
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "dataset": args.dataset,
            "model": args.model,
        }):
            print(warning)
        
        # Restore states
        generation_round = checkpoint_data.current_round
        print(f"  Resuming from generation round {generation_round}")
        print(f"  Restoring {len(checkpoint_data.active_states_data)} active states...")
        print(f"  Restoring {len(checkpoint_data.completed_states_data)} completed states...")
        
        for state_data in checkpoint_data.active_states_data:
            state = deserialize_rollout_state(state_data, shared_dataset, args)
            active_states.append(state)
        
        for state_data in checkpoint_data.completed_states_data:
            state = deserialize_rollout_state(state_data, shared_dataset, args)
            completed_states.append(state)
        
        print(f"  Successfully restored {len(active_states)} active and {len(completed_states)} completed states.")
        
        # Skip generation on first iteration - checkpoint was saved after generation but before step_batch
        skip_generation = True
    else:
        # Initialize all rollout states from scratch
        for problem_idx in range(args.num_problems):
            for sample_idx in range(args.num_samples):
                env = IntellectCodeEnv(
                    system_prompt="",
                    dataset_name=args.dataset,
                    problem_index=problem_idx,
                    max_turns=args.max_turns,
                    dataset=shared_dataset,  # Pass pre-loaded dataset
                )
                obs, info = env.reset()
                state = RolloutState(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    env=env,
                    history=[{"role": "system", "content": SYSTEM_PROMPT}],
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}],
                    obs=obs,
                )
                active_states.append(state)
    
    problems_completed = set()
    # Count already completed problems from checkpoint
    for state in completed_states:
        problem_samples_done = sum(1 for s in completed_states if s.problem_index == state.problem_index)
        if problem_samples_done == args.num_samples:
            problems_completed.add(state.problem_index)
    
    pbar = tqdm(total=args.num_problems, desc="Problems completed", initial=len(problems_completed))
    
    while active_states:
        # Skip generation if resuming (checkpoint was saved after generation, before step_batch)
        if skip_generation:
            print(f"\n[Resuming round {generation_round}] Skipping generation, going directly to code execution...")
            # Extract the last assistant response from each state's history
            processed_responses = []
            for state in active_states:
                # The last message in history should be the assistant's response
                last_assistant_msg = None
                for msg in reversed(state.history):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg["content"]
                        break
                if last_assistant_msg is None:
                    raise ValueError(f"No assistant response found in history for state {state.problem_index}:{state.sample_index}")
                processed_responses.append(last_assistant_msg)
            skip_generation = False  # Only skip once
        else:
            generation_round += 1
            print(f"\n[Generation round {generation_round}] Processing {len(active_states)} active states...")
            
            # Build prompts for all active states
            prompts = [build_prompt(s, tokenizer) for s in active_states]
            
            # Batch sample using tinker
            token_results = asyncio.run(sample_batch_tinker(client, prompts, sampling_params))
            responses = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in token_results]
            
            # Process responses and step environments
            processed_responses = []
            for state, response in zip(active_states, responses):
                response = postprocess_response(response)
                
                # Update history and messages
                state.history.append({"role": "user", "content": state.obs})
                state.history.append({"role": "assistant", "content": response})
                state.messages.append({"role": "assistant", "content": response})
                processed_responses.append(response)

            # Save checkpoint BEFORE code execution (so we can retry if execution fails)
            if checkpoint_manager:
                checkpoint_manager.save(
                    active_states_data=[serialize_rollout_state(s) for s in active_states],
                    completed_states_data=[serialize_rollout_state(s) for s in completed_states],
                    current_round=generation_round,
                    total_rounds=args.max_turns * args.num_problems * args.num_samples,  # Rough estimate
                )

        still_active = []

        if args.fast_eval:
            step_results = step_batch(
                [s.env for s in active_states],
                processed_responses,
                eval_workers=args.eval_workers,
                eval_batch_size=args.eval_batch_size,
                eval_timeout_s=args.eval_timeout_s,
                show_progress=True,
            )
        else:
            step_results = [s.env.step(r) for s, r in zip(active_states, processed_responses)]

        max_out_tokens = getattr(args, "max_interaction_output_tokens", None)
        for state, _response, (obs, reward, terminated, truncated, info) in zip(
            active_states, processed_responses, step_results
        ):
            
            # Step environment
            state.total_reward += reward
            state.terminated = terminated
            state.truncated = truncated
            
            # Track timeout counts from info
            if info.get("interaction_timed_out", False):
                state.interaction_timeout_count += 1
            if info.get("eval_timeout_count", 0) > 0:
                state.eval_timeout_count = info["eval_timeout_count"]
            
            # Truncate interaction output if over limit (guardrail against huge terminal output)
            if obs and max_out_tokens is not None and max_out_tokens > 0:
                obs = maybe_truncate_obs(obs, tokenizer, max_out_tokens)
            
            # Update interactions (truncation already applied to obs that goes into history)
            if state.env.history and len(state.env.history) > len(state.interactions):
                state.interactions = state.env.history.copy()
            
            if obs:
                state.messages.append({"role": "user", "content": obs})
            
            if terminated or truncated:
                state.done = True
                completed_states.append(state)
                
                # Check if all samples for this problem are done
                problem_samples_done = sum(
                    1 for s in completed_states if s.problem_index == state.problem_index
                )
                if problem_samples_done == args.num_samples and state.problem_index not in problems_completed:
                    problems_completed.add(state.problem_index)
                    pbar.update(1)
            else:
                state.obs = obs
                still_active.append(state)
        
        active_states = still_active
    
    pbar.close()
    
    # Organize results by problem
    all_trajectories: list[list[dict]] = [[] for _ in range(args.num_problems)]
    for state in completed_states:
        traj = {
            "question": state.env.question,
            "messages": state.messages,
            "num_turns": state.env.current_turn,
            "final_reward": state.total_reward,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "interactions": state.interactions,
            "tests": state.env.tests,
            "interaction_timeout_count": state.interaction_timeout_count,
            "eval_timeout_count": state.eval_timeout_count,
        }
        all_trajectories[state.problem_index].append(traj)
    
    return all_trajectories

def main(args):
    print(f"=" * 60)
    print(f"Collecting trajectories (Tinker backend)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Problem range: {args.start_problem} to {args.start_problem + args.num_problems - 1} ({args.num_problems} problems)")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Output: {args.output_dir}")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print(f"=" * 60)
    
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
            "num_samples": args.num_samples,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    )
    
    # Initialize tinker client
    print(f"\nInitializing tinker client...")
    sampling_client = create_sampling_client(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = create_sampling_params(args)
    
    print(f"\nCollecting trajectories for {args.num_problems} problems (batched)...")
    
    # Run batched rollouts
    all_trajectories = run_batched_rollouts(
        args=args,
        client=sampling_client,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        checkpoint_manager=checkpoint_manager,
    )
    
    # Flatten into rows (one per trajectory)
    rows = []
    all_results = []
    
    for problem_idx, problem_trajectories in enumerate(all_trajectories):
        problem_results = []
        
        for traj_idx, traj in enumerate(problem_trajectories):
            is_successful = traj["final_reward"] > 0
            problem_results.append(is_successful)
            
            rows.append({
                "problem_id": problem_idx,
                "trajectory_id": traj_idx,
                "question": traj["question"],
                "messages": json.dumps(traj["messages"]),
                "num_turns": traj["num_turns"],
                "final_reward": traj["final_reward"],
                "terminated": traj["terminated"],
                "truncated": traj["truncated"],
                "interactions": json.dumps(traj["interactions"]),
                "tests": json.dumps(traj["tests"]),
                "is_successful": is_successful,
                "interaction_timeout_count": traj["interaction_timeout_count"],
                "eval_timeout_count": traj["eval_timeout_count"],
                "rendered": render_trajectory(
                    traj["messages"], traj["interactions"], traj["question"],
                    traj["final_reward"], traj["num_turns"], traj["terminated"], traj["truncated"]
                ),
            })
        
        all_results.append(problem_results)
    
    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_2 = compute_pass_at_k(all_results, k=2)
    pass_at_4 = compute_pass_at_k(all_results, k=4)
    pass_at_8 = compute_pass_at_k(all_results, k=min(8, args.num_samples))
    
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {len(rows)}")
    print(f"  pass@1: {pass_at_1:.4f}")
    print(f"  pass@2: {pass_at_2:.4f}")
    print(f"  pass@4: {pass_at_4:.4f}")
    print(f"  pass@8: {pass_at_8:.4f}")
    print(f"{'=' * 60}")
    
    dataset = Dataset.from_list(rows)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "start_problem": args.start_problem,
        "num_problems": args.num_problems,
        "num_samples": args.num_samples,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "pass_at_1": pass_at_1,
        "pass_at_2": pass_at_2,
        "pass_at_4": pass_at_4,
        "pass_at_8": pass_at_8,
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved dataset to: {args.output_dir}")
    print(f"Saved metadata to: {metadata_path}")
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        **metadata,
        "num_successful_trajectories": sum(1 for r in rows if r["is_successful"]),
        "problems_solved": sum(1 for pr in all_results if any(pr)),
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
    parser = argparse.ArgumentParser(description="Collect multi-turn trajectories for code problems using Tinker")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL", "anirudhb11/lcb_v6_feb_may_2025_formatted"])
    parser.add_argument("--start-problem", type=int, default=0,
                        help="Starting problem index for dataset slicing (default: 0)")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    # Checkpointing options
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. checkpoints/20260117_143052)")
    
    # Evaluation options
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 5.0)")
    parser.add_argument("--max-interaction-output-tokens", type=int, default=4000,
                        help="Truncate terminal output in <output> to this many tokens (keep tail); prepend 'Output too long, showing only last X tokens' to avoid context overflow (default: 4000).")
    
    args = parser.parse_args()
    main(args)
