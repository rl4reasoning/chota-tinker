"""
Single-turn code environment demo using GEM with an LLM agent.

Runs a single problem in one shot: the assistant responds once, and the
environment evaluates that response.

Usage:
    python gem_math_demo_single_turn.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --difficulty very_hard \
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

# SYSTEM_PROMPT = """You are a helpful coding assistant.
# Solve the given programming problem and provide your solution.

# First, think about the problem step by step.
# Then, provide your final solution wrapped in ```python``` code blocks.
# """

SYSTEM_PROMPT = """You are a helpful coding assistant.

IMPORTANT CONTEXT:
- This is a single-turn conversation.

────────────────────────
HARD RULES (NON-NEGOTIABLE)
────────────────────────

- You must first reason about the problem step-by-step, and only then output the final answer. You are FORBIDDEN from outputting any ```python``` code block (even partial solutions) without any reasoning.
- The final code should execute without any exceptions.
- Use your reasoning to confirm, revise, or reject a stated hypothesis.

────────────────────────
MANDATORY GUIDELINES
────────────────────────

While formulating hypothesis, you MUST clearly state:
- The specific assumption, or uncertainty being tested
- What do you expect if the hypothesis is correct vs incorrect

After testing the hypothesis using reasoning, you MUST then clearly state:
- What the reasoning resulted in (summarize or quote key lines)
- Whether the hypothesis was confirmed, weakened, or falsified
- What (if anything) changed in your approach

────────────────────────
SOLUTION STRESS TEST (CRITICAL)
────────────────────────
- For algorithmic correctness problems, you could compare whether your implementation gives the same output compared to a bruteforce correct reference implementation
- You can build brute force / exhaustive checking for small inputs (e.g., n ≤ 6–8) and check against those
- If a counterexample is found, you MUST revise your approach and repeat the above tests.

Testing only the examples provided in the prompt does NOT count as validation or falsification.

────────────────────────
ITERATIVE WORKFLOW
────────────────────────
1. State your approach and any assumptions or uncertainties.
2. Use reasoning to address those uncertainties.
4. Repeat steps 1–2 if meaningful uncertainty remains.
5. ONLY when no critical uncertainty remains, produce the final solution.

────────────────────────
FINAL CODE REQUIREMENTS
────────────────────────
- The final code MUST be inside a ```python``` code block.
- The final code MUST read inputs from stdin and MUST NOT hardcode inputs.
- The final answer MUST clearly be supported by your reasoning evidence.
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
        interaction_mode=False,
    )

    reward = await run_single_turn(env, tokenizer, client, sampling_params)
    print(f"final reward: {reward:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
