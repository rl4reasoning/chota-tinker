"""
Demo script for early termination inference on trajectory datasets.

Takes a collected trajectory dataset, truncates at a specific turn,
adds a prompt asking the model to write final code, and evaluates the result.

Usage:
    python gem_math_demo_early_termination.py \
        --dataset anirudhb11/qwen3_4b_instruct_start_425_end_450_interations_10_turns \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --problem_id 2 \
        --trajectory_id 32 \
        --turn_index 2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from typing import Any, List, Dict

import tinker
from tinker import types
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.fast_eval import _evaluate_code
from code_env.code_env.utils.deepcoder_utils import extract_code_from_model


FINAL_PROMPT = """STOP. Do NOT use <interact> anymore. Your interaction budget is exhausted.

You MUST now output your final solution code wrapped in ```python``` code blocks.

Based on all the information and debugging you have done so far, write your best solution now. The code must:
- Read inputs from stdin
- NOT hardcode any inputs
- Be wrapped in ```python``` delimiters

Output ONLY the final ```python``` code block. No more <interact> blocks allowed."""


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
    
    This way the model has seen the feedback from turn N before being asked to write final code.
    """
    if turn_index < 1:
        raise ValueError("turn_index must be >= 1")
    
    # For turn_index N: keep 2*N + 2 messages
    # turn_index=1: indices 0-3 = 4 messages (system, user, assistant1, obs1)
    # turn_index=2: indices 0-5 = 6 messages (system, user, assistant1, obs1, assistant2, obs2)
    num_messages_to_keep = 2 * turn_index + 2
    
    if len(messages) <= num_messages_to_keep:
        return messages
    
    return messages[:num_messages_to_keep]


async def get_llm_action(messages: List[Dict], tokenizer, client, sampling_params) -> str:
    """Get LLM response for the given messages."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    input_ids = tokenizer(prompt_text)["input_ids"]
    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(input_ids),
        sampling_params=sampling_params,
        num_samples=1,
    )
    response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    return response


async def run_demo(args):
    # Load trajectory dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"  Total examples: {len(dataset)}")
    
    # Get specific example by problem_id and trajectory_id
    matching_indices = [
        i for i, row in enumerate(dataset)
        if row["problem_id"] == args.problem_id and row["trajectory_id"] == args.trajectory_id
    ]
    
    if not matching_indices:
        raise ValueError(f"No row found with problem_id={args.problem_id} and trajectory_id={args.trajectory_id}")
    
    assert len(matching_indices) == 1, f"Expected 1 matching row, found {len(matching_indices)}"
    
    example = dataset[matching_indices[0]]
    
    # Parse fields
    messages = _parse_json_field(example.get("messages", []))
    tests = _parse_json_field(example.get("tests", {}))
    question = example.get("question", "")
    original_reward = example.get("final_reward", 0.0)
    
    print(f"Problem ID: {args.problem_id}, Trajectory ID: {args.trajectory_id}")
    print(f"Original Reward: {original_reward}")
    print(f"Total Messages: {len(messages)}")
    print()
    
    # Calculate max possible turn index
    max_turn = (len(messages) - 2) // 2
    print(f"Max turn index available: {max_turn}")
    
    if args.turn_index > max_turn:
        print(f"Warning: turn_index {args.turn_index} > max available {max_turn}, using {max_turn}")
        args.turn_index = max_turn
    
    # Truncate messages
    truncated_messages = truncate_messages_at_turn(messages, args.turn_index)
    print(f"Truncated to {len(truncated_messages)} messages (turn {args.turn_index})")
    print()
    
    # Print truncated conversation
    for i, msg in enumerate(truncated_messages):
        role = msg["role"]
        content = msg["content"]
        print(f"[{role}]\n{content}\n")
    
    # Add final prompt
    truncated_messages.append({"role": "user", "content": FINAL_PROMPT})
    print(f"[user]\n{FINAL_PROMPT}\n")
    
    # Initialize model
    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        stop=[],  # No stop - we want full response
    )
    
    # Print full prompt sent to model
    full_prompt = tokenizer.apply_chat_template(
        truncated_messages, tokenize=False, add_generation_prompt=True
    )
    print(f"[full_prompt]\n{full_prompt}\n")
    
    # Run inference
    response = await get_llm_action(truncated_messages, tokenizer, client, sampling_params)
    print(f"[assistant]\n{response}\n")
    
    # Extract code from response
    code = extract_code_from_model(response)
    if not code:
        print(f"[warning] No code found in response\n")
        reward = 0.0
    else:
        # Evaluate using _evaluate_code (same as IntellectCodeEnv uses internally)
        reward, _, _ = _evaluate_code(
            code=code,
            tests=tests,
            max_tests=15,
            timeout_s=1.0,
            timeout_record_limit=0,
            require_solution_class=True,
        )
    
    print(f"[reward] {reward:.3f}\n")
    
    print(f"=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"  Original reward (full trajectory): {original_reward}")
    print(f"  New reward (truncated at turn {args.turn_index}): {reward}")
    print(f"  Success: {'YES' if reward > 0 else 'NO'}")
    print(f"=" * 60)
    
    return reward


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name with trajectory data")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--problem_id", type=int, default=0,
                        help="Problem ID to use")
    parser.add_argument("--trajectory_id", type=int, default=0,
                        help="Trajectory ID to use (for the given problem)")
    parser.add_argument("--turn_index", type=int, default=1,
                        help="Turn index to truncate at (1-indexed)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    args = parser.parse_args()
    
    print(f"Using dataset: {args.dataset}")
    print(f"Using problem_id: {args.problem_id}, trajectory_id: {args.trajectory_id}")
    print(f"Using turn index: {args.turn_index}")
    print(f"Using model: {args.model}")
    print()
    
    await run_demo(args)


if __name__ == "__main__":
    asyncio.run(main())
