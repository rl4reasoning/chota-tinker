"""
Single-turn code environment demo using GEM with an LLM agent.

Runs a single problem in one shot: the assistant responds once, and the
environment evaluates that response.

Usage:
    python gem_math_demo_single_turn.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --difficulty easy_medium \
        --problem_index 0
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import tinker
from tinker import types
from transformers import AutoTokenizer
from intellect_env import IntellectCodeEnv

# Single-turn system prompt (no interactive refinement)
SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""


async def get_llm_action(obs: str, history: list, tokenizer, client, sampling_params) -> str:
    """Single-shot generation without interactive tooling."""
    messages = history + [{"role": "user", "content": obs}]
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


async def run_single_turn(env, tokenizer, client, sampling_params):
    """Run exactly one assistant response and evaluate it."""
    obs, info = env.reset()
    # Single-turn mode: allow immediate final answer without prior <interact>
    env.has_interacted = True
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"[system]\n{SYSTEM_PROMPT}\n")
    print(f"[user]\n{obs}\n")

    action = await get_llm_action(obs, history, tokenizer, client, sampling_params)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--difficulty", type=str, default="original",
                        choices=["original", "easy_medium", "hard", "very_hard"],
                        help="Problem difficulty: easy_medium (0.3-1.0), hard (0.1-0.3), very_hard (0.0-0.1), or original (all)")
    parser.add_argument("--problem_index", type=int, default=None,
                        help="Specific problem index to use (if not set, iterates through dataset)")
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
    if args.problem_index is not None:
        print(f"Using problem index: {args.problem_index}")
    print()

    env = IntellectCodeEnv(
        system_prompt="",
        max_turns=1,
        dataset_name=dataset_name,
        problem_index=args.problem_index,
    )

    reward = await run_single_turn(env, tokenizer, client, sampling_params)
    print(f"final reward: {reward:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
