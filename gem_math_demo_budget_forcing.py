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
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import tinker
from tinker import types
from transformers import AutoTokenizer
from intellect_env import IntellectCodeEnv

# System prompt - no special thinking tags needed
SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""

# The magic word that extends thinking
WAIT_TOKEN = "Wait"


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
    tokenizer,
    client,
    sampling_params,
    num_attempts: int = 2,
):
    """Run exactly one assistant response with budget forcing and evaluate it."""
    obs, info = env.reset()
    # Single-turn mode: allow immediate final answer without prior <interact>
    env.has_interacted = True
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"[system]\n{SYSTEM_PROMPT}\n")
    print(f"[user]\n{obs}\n")

    action = await get_llm_action_with_budget_forcing(
        obs=obs,
        history=history,
        tokenizer=tokenizer,
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
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--difficulty", type=str, default="original",
                        choices=["original", "easy_medium", "hard", "very_hard"],
                        help="Problem difficulty level")
    parser.add_argument("--problem_index", type=int, default=None,
                        help="Specific problem index to use")
    
    # Budget forcing specific arguments
    parser.add_argument("--num_attempts", type=int, default=2,
                        help="Total number of generation rounds (comparable to max_turns)")
    
    args = parser.parse_args()

    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
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
        tokenizer=tokenizer,
        client=client,
        sampling_params=sampling_params,
        num_attempts=args.num_attempts,
    )
    print(f"Final reward: {reward:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
