"""
Code environment demo using GEM with LLM agent.

Uses tinker API for sampling.

Usage:
    python gem_math_demo.py --model openai/gpt-oss-120b --difficulty very_hard --problem_index 8 --fast-eval --eval-timeout-s 5.0 --max_tokens 4096 --max_steps 15

possible models:
deepseek-ai/DeepSeek-V3.1
Qwen/Qwen3-235B-A22B-Instruct-2507
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-4B-Instruct-2507
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import tinker
from tinker import types
from transformers import AutoTokenizer
from intellect_env import IntellectCodeEnv, step_batch

# Edit this to customize the system prompt
# SYSTEM_PROMPT = """You are a helpful coding assistant.
# You are allowed to interact with the Python interpreter.
# You can wrap your code in <interact></interact>, and I will run it for you and give you the output.
# Make sure that you define the inputs (or hardcode inputs) yourself when you give me <interact></interact> block.
# You can use the output to refine your code.

# Once you are done, wrap the final code in ```python``` code blocks. 
# When returning the final code, there is no need to hardcode inputs, you will take inputs from stdin.

# Please first think about the problem before you output <interact></interact> or ```python``` code blocks.

# NOTE: You must interact atleast once successfully before you submit the final code!
# """

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
- The execution environment does NOT take input from stdin. You MUST hardcode inputs in your code.
- The execution environment shows ONLY what you PRINT to stdout.
- EVERY <interact></interact> MUST include explicit print(...) statements.
- Do NOT rely on REPL-style expression outputs or implicit returns.

────────────────────────
DEFINITION OF “SUCCESSFUL <interact>”
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
  • “No counterexample found in K tests” (K ≥ 100), OR
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


async def run_episode(env, tokenizer, client, sampling_params, max_steps: int = 5, 
                      fast_eval: bool = False, eval_workers: int = 8, 
                      eval_batch_size: int = 8, eval_timeout_s: float = 5.0):
    obs, info = env.reset()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0
    
    # Track timeouts
    interaction_timeout_count = 0
    eval_timeout_count = 0
    
    print(f"[system]\n{SYSTEM_PROMPT}\n")
    print(f"[user]\n{obs}\n")
    
    for step in range(max_steps):
        action = await get_llm_action(obs, history, tokenizer, client, sampling_params)
        print(f"[assistant]\n{action}\n")
        
        history.append({"role": "user", "content": obs})
        history.append({"role": "assistant", "content": action})
        
        if fast_eval:
            results = step_batch(
                [env], [action],
                eval_workers=eval_workers,
                eval_batch_size=eval_batch_size,
                eval_timeout_s=eval_timeout_s,
            )
            obs, reward, terminated, truncated, info = results[0]
            
            # Track timeouts from info
            if info.get("interaction_timed_out", False):
                interaction_timeout_count += 1
                print(f"[timeout] Interaction timed out!")
            if info.get("eval_timeout_count", 0) > 0:
                eval_timeout_count = info["eval_timeout_count"]
                print(f"[timeout] {eval_timeout_count} test case(s) timed out during evaluation")
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if obs:
            print(f"[user]\n{obs}\n")
        print(f"[reward] {reward:.3f}\n")

        print(f"[turn {step} ends] terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            # Print episode end summary
            print(f"\n[episode end] terminated={terminated}, truncated={truncated}, info={info}")
            break
    
    # Print timeout summary
    if interaction_timeout_count > 0 or eval_timeout_count > 0:
        print(f"\n[timeout summary] Interactions: {interaction_timeout_count}, Eval test cases: {eval_timeout_count}")
    
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
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--difficulty", type=str, default="original",
                        choices=["original", "easy_medium", "hard", "very_hard"],
                        help="Problem difficulty: easy_medium (0.3-1.0), hard (0.1-0.3), very_hard (0.0-0.1), or original (all)")
    parser.add_argument("--problem_index", type=int, default=None,
                        help="Specific problem index to use (if not set, iterates through dataset)")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for code execution")
    parser.add_argument("--eval-workers", type=int, default=8,
                        help="Number of parallel evaluator workers (default: 8)")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=5.0,
                        help="Timeout in seconds for interactions and evaluation (default: 5.0)")
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
    env = IntellectCodeEnv(system_prompt="", max_turns=args.max_steps, dataset_name=dataset_name, problem_index=args.problem_index, interaction_mode=True)
    
    rewards = []
    for ep in range(args.num_episodes):
        print(f"episode {ep + 1}\n")
        r = await run_episode(
            env, tokenizer, client, sampling_params, args.max_steps,
            fast_eval=args.fast_eval,
            eval_workers=args.eval_workers,
            eval_batch_size=args.eval_batch_size,
            eval_timeout_s=args.eval_timeout_s,
        )
        rewards.append(r)
    
    print(f"avg reward: {sum(rewards)/len(rewards):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
