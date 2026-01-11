"""
Code environment demo using GEM with LLM agent.

Uses tinker API for sampling.

Usage:
    python gem_math_demo.py --model Qwen/Qwen3-4B-Instruct-2507 --difficulty easy_medium --problem_index 0

possible models:
deepseek-ai/DeepSeek-V3.1
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import tinker
from tinker import types
from transformers import AutoTokenizer
from intellect_env import IntellectCodeEnv

# Edit this to customize the system prompt
SYSTEM_PROMPT = """You are a helpful coding assistant.
You are allowed to interact with the Python interpreter.
You can wrap your code in <interact></interact>, and I will run it for you and give you the output.
Make sure that you define the inputs (or hardcode inputs) yourself when you give me <interact></interact> block.
You can use the output to refine your code.

Once you are done, wrap the final code in ```python``` code blocks. 
When returning the final code, there is no need to hardcode inputs, you will take inputs from stdin.

Please first think about the problem before you output <interact></interact> or ```python``` code blocks.

NOTE: You must interact atleast once successfully before you submit the final code!
"""


async def get_llm_action(obs: str, history: list, tokenizer, client, sampling_params) -> str:
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
    
    # if stopped by </interact>, append it back so regex can match
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    
    return response


async def run_episode(env, tokenizer, client, sampling_params, max_steps: int = 5):
    obs, info = env.reset()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0
    
    print(f"[system]\n{SYSTEM_PROMPT}\n")
    print(f"[user]\n{obs}\n")
    
    for step in range(max_steps):
        action = await get_llm_action(obs, history, tokenizer, client, sampling_params)
        print(f"[assistant]\n{action}\n")
        
        history.append({"role": "user", "content": obs})
        history.append({"role": "assistant", "content": action})
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if obs:
            print(f"[user]\n{obs}\n")
        print(f"[reward] {reward:.3f}\n")
        
        if terminated or truncated:
            break
    
    return total_reward


# Dataset mapping for difficulty levels
DATASET_MAP = {
    "original": "PrimeIntellect/INTELLECT-3-RL",
    "easy_medium": "bicycleman15/intellect_3_code_easy_medium",
    "hard": "bicycleman15/intellect_3_code_hard",
    "very_hard": "bicycleman15/intellect_3_code_very_hard",
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5)
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
        stop=["</interact>"],
    )
    
    dataset_name = DATASET_MAP[args.difficulty]
    print(f"Using dataset: {dataset_name} (difficulty: {args.difficulty})")
    if args.problem_index is not None:
        print(f"Using problem index: {args.problem_index}")
    print()
    env = IntellectCodeEnv(system_prompt="", max_turns=args.max_steps, dataset_name=dataset_name, problem_index=args.problem_index)
    
    rewards = []
    for ep in range(args.num_episodes):
        print(f"episode {ep + 1}\n")
        r = await run_episode(env, tokenizer, client, sampling_params, args.max_steps)
        rewards.append(r)
    
    print(f"avg reward: {sum(rewards)/len(rewards):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
