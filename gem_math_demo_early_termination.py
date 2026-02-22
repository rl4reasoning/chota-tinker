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

    # For GPT-OSS models (uses Harmony format):
    python gem_math_demo_early_termination.py \
        --dataset anirudhb11/qwen3_4b_instruct_start_425_end_450_interations_10_turns \
        --model openai/gpt-oss-120b \
        --problem_id 2 \
        --trajectory_id 32 \
        --turn_index 2 \
        --reasoning-effort medium
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from datetime import date
from typing import Any, List, Dict, Optional

import tinker
from tinker import types
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.fast_eval import _evaluate_code
from code_env.code_env.utils.deepcoder_utils import extract_code_from_model

# Harmony utilities for GPT-OSS models
from utils.harmony_utils import (
    is_gpt_oss_model,
    parse_harmony_response,
    load_harmony_encoding,
    HarmonyEncodingName,
    HarmonyRole,
    HarmonyMessage,
    Conversation,
    DeveloperContent,
    SystemContent,
    ReasoningEffort,
)


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


def build_harmony_conversation_from_messages(
    messages: List[Dict], 
    encoding, 
    reasoning_effort: str = "medium"
) -> Conversation:
    """Build a Harmony Conversation from a list of message dictionaries.
    
    For Harmony format, we need to:
    1. Add a SystemContent with reasoning effort and date
    2. Convert system messages to developer messages with DeveloperContent
    3. Handle user/assistant messages normally
    """
    harmony_messages = []
    
    # First, add a system message with reasoning effort
    system_content = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort[reasoning_effort.upper()])
        .with_conversation_start_date(date.today().isoformat())
    )
    harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, system_content))
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            # Convert system to developer for Harmony
            developer_content = DeveloperContent.new().with_instructions(content)
            harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, developer_content))
        elif role == "developer":
            # Already a developer message
            if isinstance(content, str):
                developer_content = DeveloperContent.new().with_instructions(content)
                harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, developer_content))
            else:
                harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
        elif role == "user":
            harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))
        elif role == "assistant":
            channel = msg.get("channel", "final")
            h_msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content)
            h_msg = h_msg.with_channel(channel)
            harmony_messages.append(h_msg)
    
    return Conversation.from_messages(harmony_messages)


async def get_llm_action_harmony(
    messages: List[Dict], 
    encoding, 
    client, 
    sampling_params,
    reasoning_effort: str = "medium"
) -> tuple[str, str, Optional[str]]:
    """Get LLM response using Harmony format for GPT-OSS models."""
    conversation = build_harmony_conversation_from_messages(messages, encoding, reasoning_effort)
    input_ids = encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)
    
    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(input_ids),
        sampling_params=sampling_params,
        num_samples=1,
    )
    
    response_tokens = result.sequences[0].tokens
    response_content, channel, analysis_content = parse_harmony_response(response_tokens, encoding)
    
    return response_content, channel, analysis_content


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
    num_turns = example.get("num_turns", float('inf'))
    terminated = example.get("terminated", False)
    
    print(f"Problem ID: {args.problem_id}, Trajectory ID: {args.trajectory_id}")
    print(f"Original Reward: {original_reward}")
    print(f"Total Messages: {len(messages)}")
    print(f"Original num_turns: {num_turns}, Terminated: {terminated}")
    print()
    
    # Detect if model is a GPT-OSS model (requires Harmony format)
    use_harmony = is_gpt_oss_model(args.model)
    if use_harmony:
        print(f"[INFO] Detected GPT-OSS model: {args.model}")
        print(f"[INFO] Using Harmony format with reasoning_effort={args.reasoning_effort}")
        print()
    
    # Check if trajectory already ended within turn budget (BEFORE capping turn_index)
    # (either terminated with answer OR truncated at max turns)
    already_ended = num_turns <= args.turn_index
    
    if already_ended:
        # Trajectory already ended - use original result
        print(f"\nTrajectory already ended at turn {num_turns} (within budget of {args.turn_index})")
        print(f"Terminated: {terminated}, using original result without inference.\n")
        reward = original_reward
    else:
        # Need to truncate and run inference
        print(f"\nTrajectory did not complete by turn {args.turn_index}, running inference...\n")
        
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
        
        if use_harmony:
            # GPT-OSS model: use Harmony encoding
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            
            # Get stop token IDs from encoding
            # Use stop_tokens_for_assistant_actions() which returns only <|return|> and <|call|>
            # Do NOT use stop_tokens() which includes <|end|> - that marks the end of ONE message,
            # but the model outputs multiple messages (analysis channel -> final channel)
            stop_token_ids = encoding.stop_tokens_for_assistant_actions()
            
            sampling_params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_token_ids=stop_token_ids,
            )
            
            print(f"[system] (Harmony format)")
            print(f"  Reasoning effort: {args.reasoning_effort}")
            print(f"  Date: {date.today().isoformat()}\n")
            
            # Run inference with Harmony format
            response, channel, analysis_content = await get_llm_action_harmony(
                truncated_messages, encoding, client, sampling_params, args.reasoning_effort
            )
            if analysis_content:
                print(f"[analysis]\n{analysis_content}\n")
            print(f"[assistant (channel: {channel})]\n{response}\n")
        else:
            # Standard model: use tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            
            sampling_params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
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
    print(f"  New reward (at turn {args.turn_index}): {reward}")
    print(f"  Already ended: {already_ended}")
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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p",
                        help="Top-p sampling")
    
    # Harmony-specific arguments (for GPT-OSS models)
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort level for Harmony models (default: medium)")
    
    args = parser.parse_args()
    
    print(f"Using dataset: {args.dataset}")
    print(f"Using problem_id: {args.problem_id}, trajectory_id: {args.trajectory_id}")
    print(f"Using turn index: {args.turn_index}")
    print(f"Using model: {args.model}")
    print()
    
    await run_demo(args)


if __name__ == "__main__":
    asyncio.run(main())
