"""
Single-turn code environment demo using GEM with an LLM agent.

Runs a single problem in one shot: the assistant responds once, and the
environment evaluates that response.

Usage:
    python gem_math_demo_single_turn.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --difficulty very_hard \
        --problem_index 0

    # For GPT-OSS models (uses Harmony format):
    python gem_math_demo_single_turn.py \
        --model openai/gpt-oss-120b \
        --difficulty very_hard \
        --problem_index 0 \
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

# =============================================================================
# PROMPTS
# =============================================================================
# For standard models, SYSTEM_PROMPT goes in the "system" role.
# For GPT-OSS models using Harmony, these instructions go in the "developer" role.

# SYSTEM_PROMPT = """You are a helpful coding assistant.
# Solve the given programming problem and provide your solution.

# First, think about the problem step by step.
# Then, provide your final solution wrapped in ```python``` code blocks.
# """


# prompt_v2
# SYSTEM_PROMPT = """You are a helpful coding assistant.

# IMPORTANT CONTEXT:
# - This is a single-turn conversation.

# ────────────────────────
# HARD RULES (NON-NEGOTIABLE)
# ────────────────────────

# - You must first reason about the problem step-by-step, and only then output the final answer. You are FORBIDDEN from outputting any ```python``` code block (even partial solutions) without any reasoning.
# - The final code should execute without any exceptions.
# - Use your reasoning to confirm, revise, or reject a stated hypothesis.

# ────────────────────────
# MANDATORY GUIDELINES
# ────────────────────────

# While formulating hypothesis, you MUST clearly state:
# - The specific assumption, or uncertainty being tested
# - What do you expect if the hypothesis is correct vs incorrect

# After testing the hypothesis using reasoning, you MUST then clearly state:
# - What the reasoning resulted in (summarize or quote key lines)
# - Whether the hypothesis was confirmed, weakened, or falsified
# - What (if anything) changed in your approach

# ────────────────────────
# SOLUTION STRESS TEST (CRITICAL)
# ────────────────────────
# - For algorithmic correctness problems, you could compare whether your implementation gives the same output compared to a bruteforce correct reference implementation
# - You can build brute force / exhaustive checking for small inputs (e.g., n ≤ 6–8) and check against those
# - If a counterexample is found, you MUST revise your approach and repeat the above tests.

# Testing only the examples provided in the prompt does NOT count as validation or falsification.

# ────────────────────────
# ITERATIVE WORKFLOW
# ────────────────────────
# 1. State your approach and any assumptions or uncertainties.
# 2. Use reasoning to address those uncertainties.
# 4. Repeat steps 1–2 if meaningful uncertainty remains.
# 5. ONLY when no critical uncertainty remains, produce the final solution.

# ────────────────────────
# FINAL CODE REQUIREMENTS
# ────────────────────────
# - The final code MUST be inside a ```python``` code block.
# - The final code MUST read inputs from stdin and MUST NOT hardcode inputs.
# - The final answer MUST clearly be supported by your reasoning evidence.
# """

# prompt_v3
# SYSTEM_PROMPT = """You are an expert competitive programming assistant.

# ----------------------------
# PROBLEM-SOLVING APPROACH
# ----------------------------
# 1. UNDERSTAND: Carefully read and restate the problem in your own words.
# 2. ANALYZE: Identify key constraints, edge cases, and the core algorithmic challenge.
# 3. VERIFY: Mentally trace through the provided examples step-by-step.
# 4. IMPLEMENT: Write clean, correct, and efficient code.

# ----------------------------
# REASONING REQUIREMENTS
# ----------------------------
# Before writing any code, you MUST:
# - Identify the input/output format precisely
# - State the time and space complexity constraints
# - Walk through at least one example by hand to verify your understanding

# ----------------------------
# CODE REQUIREMENTS
# ----------------------------
# - The solution MUST be inside a ```python``` code block
# """

# For Harmony format, use the same instructions

# prompt_v4
SYSTEM_PROMPT = """You are an expert competitive programming assistant.

----------------------------
PROBLEM-SOLVING APPROACH
----------------------------
1. UNDERSTAND: Carefully read and restate the problem in your own words.
2. ANALYZE: Identify key constraints, edge cases, and the core algorithmic challenge.
3. DESIGN: Choose an appropriate algorithm/data structure and justify your choice.
4. VERIFY: Mentally trace through the provided examples step-by-step.
5. IMPLEMENT: Write clean, correct, and efficient code.

----------------------------
REASONING REQUIREMENTS
----------------------------
Before writing any code, you MUST:
- Identify the input/output format precisely
- State the time and space complexity constraints
- Consider edge cases (empty input, single element, maximum values, etc.)
- Walk through at least one example by hand to verify your understanding

----------------------------
CODE REQUIREMENTS
----------------------------
- The solution MUST be inside a ```python``` code block
- The code MUST handle all edge cases mentioned in the problem
- Use appropriate data structures for the problem's constraints

----------------------------
COMMON PITFALLS TO AVOID
----------------------------
- Off-by-one errors in loops and array indexing
- Integer overflow (use appropriate types if needed)
- Not handling edge cases (n=0, n=1, empty strings, etc.)
- Inefficient algorithms that exceed time limits
- Incorrect input parsing (watch for multiple test cases, line formats)
- Forgetting to flush output when required
"""

DEVELOPER_INSTRUCTIONS = SYSTEM_PROMPT


# =============================================================================
# HARMONY FORMAT HELPERS (for GPT-OSS models)
# =============================================================================

def build_harmony_conversation(history: list, obs: str, encoding) -> Conversation:
    """
    Build a Harmony Conversation from the history list and current observation.
    
    For single-turn, the history contains:
    - history[0]: SystemContent message
    - history[1]: DeveloperContent message
    """
    messages = []
    
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        
        if role == "system":
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, content))
        elif role == "developer":
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


async def get_llm_action_harmony(obs: str, history: list, encoding, client, sampling_params) -> tuple[str, str, Optional[str]]:
    """
    Get LLM action using Harmony format for GPT-OSS models.
    
    Returns:
        tuple of (response_content, channel, analysis_content)
        - response_content: The user-facing content
        - channel: The channel of the response ('final', 'analysis', 'commentary')
        - analysis_content: Chain-of-thought content (for logging, not shown to user)
    """
    conversation = build_harmony_conversation(history, obs, encoding)
    
    # Render the conversation for completion
    input_ids = encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)
    
    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(input_ids),
        sampling_params=sampling_params,
        num_samples=1,
    )
    
    response_tokens = result.sequences[0].tokens
    
    # Parse the response using Harmony
    response_content, channel, analysis_content = parse_harmony_response(response_tokens, encoding)
    
    return response_content, channel, analysis_content


async def get_llm_action(obs: str, history: list, tokenizer, client, sampling_params) -> str:
    """Single-shot generation without interactive tooling (standard models)."""
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


async def run_single_turn(env, tokenizer_or_encoding, client, sampling_params,
                          use_harmony: bool = False, reasoning_effort: str = "high"):
    """
    Run exactly one assistant response and evaluate it.
    
    Args:
        tokenizer_or_encoding: Either a HuggingFace tokenizer (standard) or HarmonyEncoding (GPT-OSS)
        use_harmony: If True, use Harmony format for GPT-OSS models
        reasoning_effort: For Harmony models, the reasoning effort level (low/medium/high)
    """
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
        action, channel, analysis = await get_llm_action_harmony(
            obs, history, tokenizer_or_encoding, client, sampling_params
        )
        
        # Log the chain-of-thought if present
        if analysis:
            print(f"[assistant] (channel: analysis) [INTERNAL COT - not shown to user]")
            print(f"{analysis[:500]}{'...' if len(analysis) > 500 else ''}\n")
        
        print(f"[assistant] (channel: {channel})\n{action}\n")
    else:
        action = await get_llm_action(obs, history, tokenizer_or_encoding, client, sampling_params)
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
    parser.add_argument("--reasoning-effort", type=str, default="high",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-OSS models (default: high)")
    args = parser.parse_args()

    # Detect if we're using a GPT-OSS model that requires Harmony format
    use_harmony = is_gpt_oss_model(args.model)
    
    if use_harmony:
        print(f"Detected GPT-OSS model: {args.model}")
        print(f"Using Harmony format with reasoning effort: {args.reasoning_effort}")
        
        # Load Harmony encoding instead of HuggingFace tokenizer
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokenizer_or_encoding = encoding
        
        # Get stop tokens from Harmony encoding
        harmony_stop_tokens = encoding.stop_tokens()
        
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop_token_ids=harmony_stop_tokens,
        )
    else:
        print(f"Using standard model: {args.model}")
        tokenizer_or_encoding = AutoTokenizer.from_pretrained(args.model)
        
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )

    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)

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

    reward = await run_single_turn(
        env, tokenizer_or_encoding, client, sampling_params,
        use_harmony=use_harmony,
        reasoning_effort=args.reasoning_effort,
    )
    print(f"final reward: {reward:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
