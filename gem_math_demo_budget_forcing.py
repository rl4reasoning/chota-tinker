"""
Budget Forcing demo implementing the s1 test-time scaling method.

This implements the "budget forcing" technique from:
    s1: Simple test-time scaling (https://arxiv.org/abs/2501.19393)

The key idea: when the model tries to end generation (hits EOS/stop token),
forcefully append "Wait" to make it continue reasoning. This often leads
the model to double-check and fix incorrect reasoning steps.

Usage:
    python gem_math_demo_budget_forcing.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --difficulty easy_medium \
        --problem_index 0 \
        --num_attempts 3 \
        --max_tokens 4096

    # For GPT-OSS models (uses Harmony format):
    python gem_math_demo_budget_forcing.py \
        --model openai/gpt-oss-120b \
        --difficulty easy_medium \
        --problem_index 0 \
        --num_attempts 3 \
        --reasoning-effort medium
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
from datetime import date
from typing import Optional

import tinker
from tinker import types
from transformers import AutoTokenizer
from intellect_env import IntellectCodeEnv

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

# System prompt - no special thinking tags needed
SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""

# For Harmony format, use the same instructions
DEVELOPER_INSTRUCTIONS = SYSTEM_PROMPT

# The magic word that extends thinking
WAIT_TOKEN = "Wait"


# =============================================================================
# HARMONY FORMAT HELPERS (for GPT-OSS models)
# =============================================================================

def build_harmony_conversation(history: list, obs: str, encoding) -> Conversation:
    """Build a Harmony Conversation from the history list and current observation."""
    messages = []
    
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        
        if role == "system":
            # SystemContent objects are passed directly
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, content))
        elif role == "developer":
            # DeveloperContent objects are passed directly
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
        elif role == "user":
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))
        elif role == "assistant":
            channel = entry.get("channel", "final")
            msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content)
            msg = msg.with_channel(channel)
            messages.append(msg)
    
    # Add the current observation as a user message
    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, obs))
    
    return Conversation.from_messages(messages)


async def get_llm_action_with_budget_forcing_harmony(
    obs: str,
    history: list,
    encoding,
    client,
    sampling_params: types.SamplingParams,
    num_attempts: int = 2,
    verbose: bool = True,
) -> tuple[str, str, Optional[str]]:
    """
    Generate a response with budget forcing using Harmony format for GPT-OSS models.
    
    Returns:
        tuple of (full_response, channel, analysis_content)
    """
    # Build initial conversation
    conversation = build_harmony_conversation(history, obs, encoding)
    input_ids = encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)
    
    num_ignores = num_attempts - 1
    
    if verbose:
        print(f"[budget forcing] {num_attempts} generation round(s), {num_ignores} forced extension(s) with '{WAIT_TOKEN}'")
    
    total_tokens = 0
    max_total_tokens = sampling_params.max_tokens
    full_response = ""
    last_channel = "final"
    last_analysis = None
    
    for attempt_idx in range(num_attempts):
        remaining_tokens = max_total_tokens - total_tokens
        
        if remaining_tokens <= 0:
            if verbose:
                print(f"[budget forcing] Reached max tokens, stopping")
            break
        
        current_params = types.SamplingParams(
            max_tokens=remaining_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            stop_token_ids=sampling_params.stop_token_ids,
        )
        
        result = await client.sample_async(
            prompt=types.ModelInput.from_ints(input_ids),
            sampling_params=current_params,
            num_samples=1,
        )
        
        response_tokens = result.sequences[0].tokens
        generated_tokens = len(response_tokens)
        total_tokens += generated_tokens
        
        # Parse the response using Harmony
        response_content, channel, analysis_content = parse_harmony_response(response_tokens, encoding)
        last_channel = channel
        if analysis_content:
            last_analysis = analysis_content
        
        if verbose:
            print(f"[budget forcing] Generation {attempt_idx + 1}/{num_attempts}: {generated_tokens} tokens (channel: {channel})")
        
        is_last_iteration = (attempt_idx == num_attempts - 1)
        
        if is_last_iteration:
            full_response += response_content
            if verbose:
                print(f"[budget forcing] Final generation complete")
            break
        else:
            full_response += response_content + " " + WAIT_TOKEN
            # Re-render conversation with the extended response
            # Add the response to history and rebuild
            extended_history = history + [
                {"role": "user", "content": obs},
                {"role": "assistant", "content": full_response, "channel": channel}
            ]
            # Build new prompt continuing from the extended response
            new_conv = build_harmony_conversation(extended_history[:-1], extended_history[-2]["content"], encoding)
            # Actually we need to continue from the partial response, so just extend input_ids
            # Simpler approach: re-encode the full prompt including the partial response
            temp_messages = []
            for entry in history:
                role = entry["role"]
                content = entry["content"]
                if role == "system":
                    temp_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, content))
                elif role == "developer":
                    temp_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
            temp_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, obs))
            # Add partial assistant response
            partial_msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, full_response)
            partial_msg = partial_msg.with_channel(channel)
            temp_messages.append(partial_msg)
            temp_conv = Conversation.from_messages(temp_messages)
            input_ids = encoding.render_conversation_for_completion(temp_conv, HarmonyRole.ASSISTANT)
            
            if verbose:
                print(f"[budget forcing] Appending '{WAIT_TOKEN}' to extend generation...")
    
    if verbose:
        print(f"[budget forcing] Total tokens generated: {total_tokens}")
    
    return full_response, last_channel, last_analysis


async def get_llm_action_with_budget_forcing(
    obs: str,
    history: list,
    tokenizer,
    client,
    sampling_params: types.SamplingParams,
    num_attempts: int = 2,
    verbose: bool = True,
) -> str:
    """
    Generate a response with budget forcing (EOS-based).
    
    Budget forcing works by:
    1. Generate until the model hits EOS/stop token
    2. Append "Wait" to force the model to continue
    3. Repeat for num_attempts total rounds
    4. On the final generation, let the model finish naturally
    
    Args:
        obs: The observation/prompt from the environment
        history: Chat history
        tokenizer: The tokenizer
        client: The sampling client
        sampling_params: Base sampling parameters
        num_attempts: Total number of generation rounds (comparable to max_turns)
        verbose: Whether to print progress
        
    Returns:
        The full response including all extended generations
    """
    messages = history + [{"role": "user", "content": obs}]
    
    # Build the initial prompt
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    num_ignores = num_attempts - 1  # Number of "Wait" injections
    
    if verbose:
        print(f"[budget forcing] {num_attempts} generation round(s), {num_ignores} forced extension(s) with '{WAIT_TOKEN}'")
    
    # Track total tokens and full response
    total_tokens = 0
    max_total_tokens = sampling_params.max_tokens
    full_response = ""
    
    # Generate with budget forcing - append "Wait" when model tries to stop
    for attempt_idx in range(num_attempts):
        remaining_tokens = max_total_tokens - total_tokens
        
        if remaining_tokens <= 0:
            if verbose:
                print(f"[budget forcing] Reached max tokens, stopping")
            break
        
        # Create params for this generation
        current_params = types.SamplingParams(
            max_tokens=remaining_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
        )
        
        # Tokenize current prompt
        input_ids = tokenizer(prompt_text)["input_ids"]
        
        # Generate
        result = await client.sample_async(
            prompt=types.ModelInput.from_ints(input_ids),
            sampling_params=current_params,
            num_samples=1,
        )
        
        generated_text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        generated_tokens = len(result.sequences[0].tokens)
        total_tokens += generated_tokens
        
        if verbose:
            print(f"[budget forcing] Generation {attempt_idx + 1}/{num_attempts}: {generated_tokens} tokens")
        
        # Check if we should force continuation
        is_last_iteration = (attempt_idx == num_attempts - 1)
        
        if is_last_iteration:
            # Final generation - just append and finish
            full_response += generated_text
            if verbose:
                print(f"[budget forcing] Final generation complete")
            break
        else:
            # Force continuation by appending "Wait"
            full_response += generated_text + " " + WAIT_TOKEN
            prompt_text += generated_text + " " + WAIT_TOKEN
            
            if verbose:
                print(f"[budget forcing] Appending '{WAIT_TOKEN}' to extend generation...")
    
    if verbose:
        print(f"[budget forcing] Total tokens generated: {total_tokens}")
    
    return full_response


async def run_single_turn_with_budget_forcing(
    env,
    tokenizer_or_encoding,
    client,
    sampling_params,
    num_attempts: int = 2,
    use_harmony: bool = False,
    reasoning_effort: str = "medium",
):
    """Run exactly one assistant response with budget forcing and evaluate it."""
    obs, info = env.reset()
    # Single-turn mode: allow immediate final answer without prior <interact>
    env.has_interacted = True
    
    if use_harmony:
        # Build Harmony-style history with system and developer messages
        system_content = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort[reasoning_effort.upper()])
            .with_conversation_start_date(date.today().isoformat())
        )
        developer_content = (
            DeveloperContent.new()
            .with_instructions(DEVELOPER_INSTRUCTIONS)
        )
        
        history = [
            {"role": "system", "content": system_content},
            {"role": "developer", "content": developer_content},
        ]
        
        print(f"[system] (Harmony format)")
        print(f"  Reasoning effort: {reasoning_effort}")
        print(f"  Date: {date.today().isoformat()}\n")
        print(f"[developer]\n{DEVELOPER_INSTRUCTIONS}\n")
    else:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"[system]\n{SYSTEM_PROMPT}\n")

    print(f"[user]\n{obs}\n")

    if use_harmony:
        action, channel, analysis_content = await get_llm_action_with_budget_forcing_harmony(
            obs=obs,
            history=history,
            encoding=tokenizer_or_encoding,
            client=client,
            sampling_params=sampling_params,
            num_attempts=num_attempts,
        )
        if analysis_content:
            print(f"[analysis]\n{analysis_content}\n")
        print(f"[assistant (channel: {channel})]\n{action}\n")
    else:
        action = await get_llm_action_with_budget_forcing(
            obs=obs,
            history=history,
            tokenizer=tokenizer_or_encoding,
            client=client,
            sampling_params=sampling_params,
            num_attempts=num_attempts,
        )
        print(f"[assistant]\n{action}\n")

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"[reward] {reward:.3f} | terminated={terminated} | truncated={truncated}\n")

    return reward


# Dataset mapping for difficulty levels
DATASET_MAP = {
    "original": "PrimeIntellect/INTELLECT-3-RL",
    "easy_medium": "bicycleman15/intellect_3_code_easy_medium",
    "hard": "bicycleman15/intellect_3_code_hard",
    "very_hard": "bicycleman15/intellect_3_code_very_hard",
}


async def main():
    parser = argparse.ArgumentParser(
        description="Run coding problems with s1 budget forcing for extended thinking"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for generation (shared across all extensions)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--difficulty", type=str, default="original",
                        choices=["original", "easy_medium", "hard", "very_hard"],
                        help="Problem difficulty level")
    parser.add_argument("--problem_index", type=int, default=None,
                        help="Specific problem index to use")
    
    # Budget forcing specific arguments
    parser.add_argument("--num_attempts", type=int, default=2,
                        help="Total number of generation rounds (comparable to max_turns)")
    
    # Harmony-specific arguments (for GPT-OSS models)
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort level for Harmony models (default: medium)")
    
    args = parser.parse_args()

    # Detect if model is a GPT-OSS model (requires Harmony format)
    use_harmony = is_gpt_oss_model(args.model)
    
    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)
    
    if use_harmony:
        # GPT-OSS model: use Harmony encoding instead of tokenizer
        print(f"[INFO] Detected GPT-OSS model: {args.model}")
        print(f"[INFO] Using Harmony format with reasoning_effort={args.reasoning_effort}")
        
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokenizer_or_encoding = encoding
        
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
    else:
        # Standard model: use tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer_or_encoding = tokenizer
        
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    dataset_name = DATASET_MAP[args.difficulty]
    print(f"Using dataset: {dataset_name} (difficulty: {args.difficulty})")
    print(f"Budget forcing: {args.num_attempts} generation round(s)")
    if args.problem_index is not None:
        print(f"Using problem index: {args.problem_index}")
    print()

    env = IntellectCodeEnv(
        system_prompt="",
        max_turns=1,
        dataset_name=dataset_name,
        problem_index=args.problem_index,
    )

    reward = await run_single_turn_with_budget_forcing(
        env=env,
        tokenizer_or_encoding=tokenizer_or_encoding,
        client=client,
        sampling_params=sampling_params,
        num_attempts=args.num_attempts,
        use_harmony=use_harmony,
        reasoning_effort=args.reasoning_effort,
    )
    print(f"Final reward: {reward:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
