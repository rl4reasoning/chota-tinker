"""
RLEF (Reinforcement Learning with Execution Feedback) demo.

Multi-turn inference where the model generates code, receives structured
feedback from public test cases, and refines its solution.
Implements the inference loop from arXiv:2410.02089.

Uses tinker API for sampling.

Usage:
    # Batch mode (default)
    python gem_math_demo_rlef.py --model Qwen/Qwen3-4B-Instruct-2507 --difficulty very_hard --problem_index 0 --eval-timeout-s 10.0 --max_tokens 4096 --max_steps 3

    # Streaming mode (requires checkpoint)
    python gem_math_demo_rlef.py --model Qwen/Qwen3-4B-Instruct-2507 --difficulty very_hard --problem_index 11 --max_tokens 4096 --max_steps 3 --stream

    # For GPT-OSS models (uses Harmony format, streaming not supported):
    python gem_math_demo_rlef.py --model openai/gpt-oss-120b --difficulty very_hard --problem_index 10 --eval-timeout-s 10.0 --max_tokens 8192 --max_steps 5 --reasoning-effort medium

For streaming, first create a checkpoint (one-time per model):
    python create_checkpoint.py --model Qwen/Qwen3-4B-Instruct-2507

possible models:
deepseek-ai/DeepSeek-V3.1
Qwen/Qwen3-235B-A22B-Instruct-2507
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-4B-Instruct-2507
openai/gpt-oss-120b (uses Harmony format)
openai/gpt-oss-20b (uses Harmony format)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import sys
from datetime import date
from typing import Optional

import tinker
from tinker import types
from openai import OpenAI
from transformers import AutoTokenizer
from rlef_env import RLEFCodeEnv
from create_checkpoint import get_checkpoint

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role as HarmonyRole,
    Message as HarmonyMessage,
    Conversation,
    DeveloperContent,
    SystemContent,
    ReasoningEffort,
)


def is_gpt_oss_model(model_name: str) -> bool:
    """Check if the model is a GPT-OSS model that requires Harmony format."""
    return "gpt-oss" in model_name.lower()


# =============================================================================
# PROMPTS
# =============================================================================

DEVELOPER_INSTRUCTIONS = """You are an expert competitive programmer.

You will be given a programming problem along with public test cases.
Solve it in Python by submitting your solution inside a ```python``` code block.

Your code will be automatically tested against the public test cases. You will
receive structured feedback showing which tests passed or failed, including
the expected vs actual output for failures.

If any tests fail, carefully analyze the feedback, identify the bug or
misunderstanding, and submit an improved solution. You have multiple attempts.

RULES:
- Your code MUST read input from stdin and write output to stdout.
- Do NOT hardcode test inputs.
- Each response should contain exactly ONE ```python``` code block with your
  complete solution.
- Before the code block you may include a brief analysis of the problem or
  the feedback from the previous attempt.
"""

SYSTEM_PROMPT = DEVELOPER_INSTRUCTIONS

FINAL_PROMPT = """This is your LAST attempt. Submit your best solution now.

You MUST output your final solution code wrapped in a ```python``` code block."""


# =============================================================================
# HARMONY FORMAT HELPERS (for GPT-OSS models)
# =============================================================================

def build_harmony_conversation(history: list, obs: str, encoding) -> Conversation:
    """Build a Harmony Conversation from the history list and current observation."""
    messages = []

    for i, entry in enumerate(history):
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

            should_drop_analysis = False
            if channel == "analysis":
                for j in range(i + 1, len(history)):
                    if history[j]["role"] == "assistant":
                        if history[j].get("channel") == "final":
                            should_drop_analysis = True
                        break

            if not should_drop_analysis:
                msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content)
                msg = msg.with_channel(channel)
                messages.append(msg)

    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, obs))
    return Conversation.from_messages(messages)


def parse_harmony_response(tokens: list, encoding) -> tuple[str, str, Optional[str]]:
    """Parse Harmony response tokens and extract content by channel."""
    try:
        parsed_messages = encoding.parse_messages_from_completion_tokens(
            tokens, role=HarmonyRole.ASSISTANT, strict=False
        )
    except Exception as e:
        print(f"Error parsing Harmony response: {e}")
        try:
            raw_text = encoding.decode(tokens)
            return raw_text, "parse_error", None
        except Exception:
            return "", "parse_error", None

    analysis_content = None
    final_content = None
    commentary_content = None

    for msg in parsed_messages:
        channel = msg.channel or "final"
        text_parts = []
        for content_item in msg.content:
            if hasattr(content_item, "text"):
                text_parts.append(content_item.text)
        combined_text = "\n".join(text_parts) if text_parts else ""
        if channel == "final":
            final_content = combined_text
        elif channel == "analysis":
            analysis_content = combined_text
        elif channel == "commentary":
            commentary_content = combined_text

    if final_content:
        return final_content, "final", analysis_content
    elif commentary_content:
        return commentary_content, "commentary", analysis_content
    elif analysis_content:
        return analysis_content, "analysis", None
    else:
        try:
            raw_text = encoding.decode(tokens)
            return raw_text, "unknown", None
        except Exception:
            return "", "unknown", None


# =============================================================================
# LLM SAMPLING
# =============================================================================

async def get_llm_action_harmony(
    obs: str, history: list, encoding, client, sampling_params
) -> tuple[str, str, Optional[str]]:
    """Get LLM action using Harmony format for GPT-OSS models."""
    conversation = build_harmony_conversation(history, obs, encoding)
    input_ids = encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)

    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(input_ids),
        sampling_params=sampling_params,
        num_samples=1,
    )
    response_tokens = result.sequences[0].tokens
    return parse_harmony_response(response_tokens, encoding)


async def get_llm_action(
    obs: str,
    history: list,
    tokenizer,
    client,
    sampling_params,
    stream: bool = False,
    model: str = None,
    oai_client: OpenAI = None,
) -> str:
    """Get LLM action using standard chat template."""
    messages = history + [{"role": "user", "content": obs}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text)["input_ids"]

    if stream and oai_client is not None and model is not None:
        response = ""
        for chunk in oai_client.completions.create(
            model=model,
            prompt=prompt_text,
            max_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            stream=True,
        ):
            text = chunk.choices[0].text
            print(text, end="", flush=True)
            response += text
        print()
    else:
        result = await client.sample_async(
            prompt=types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

    return response


# =============================================================================
# EPISODE LOOP
# =============================================================================

async def run_episode(
    env: RLEFCodeEnv,
    tokenizer_or_encoding,
    client,
    sampling_params,
    max_steps: int = 3,
    use_harmony: bool = False,
    reasoning_effort: str = "high",
    stream: bool = False,
    model: str = None,
    oai_client: OpenAI = None,
):
    """Run a single RLEF episode.

    The model generates code, the environment evaluates it on public tests
    and returns structured feedback. When all public tests pass or the turn
    limit is reached, the solution is scored on all tests (public + private).
    """
    obs, info = env.reset()
    total_reward = 0

    if use_harmony:
        system_content = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort[reasoning_effort.upper()])
            .with_conversation_start_date(date.today().isoformat())
        )
        developer_content = DeveloperContent.new().with_instructions(DEVELOPER_INSTRUCTIONS)
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

    for step in range(max_steps):
        is_last_turn = step == max_steps - 1
        if is_last_turn:
            obs_for_llm = f"{obs}\n\n{FINAL_PROMPT}" if obs else FINAL_PROMPT
            print(f"[user]\n{obs_for_llm}\n")
        else:
            obs_for_llm = obs

        # --- Sample from LLM ---
        if use_harmony:
            action, channel, analysis = await get_llm_action_harmony(
                obs_for_llm, history, tokenizer_or_encoding, client, sampling_params
            )
            if analysis:
                print(f"[assistant] (channel: analysis) [Internal CoT]")
                print(f"{analysis}\n")
            print(f"[assistant] (channel: {channel})\n{action}\n")

            history.append({"role": "user", "content": obs_for_llm})
            if analysis and channel == "final":
                history.append({"role": "assistant", "content": analysis, "channel": "analysis"})
            history.append({"role": "assistant", "content": action, "channel": channel})
        else:
            if stream:
                print(f"[assistant]")
            action = await get_llm_action(
                obs_for_llm, history, tokenizer_or_encoding, client, sampling_params,
                stream=stream, model=model, oai_client=oai_client,
            )
            if not stream:
                print(f"[assistant]\n{action}\n")
            else:
                print()

            history.append({"role": "user", "content": obs_for_llm})
            history.append({"role": "assistant", "content": action})

        # --- Step the RLEF environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        # If episode ended, run final evaluation on private tests
        if info.get("needs_eval") and info.get("code"):
            reward = env.evaluate_final(info["code"])

        total_reward += reward

        if obs:
            print(f"[user]\n{obs}\n")
        print(f"[reward] {reward:.3f}")
        print(f"[turn {step} ends] terminated={terminated}, truncated={truncated}, info={info}\n")

        if terminated or truncated:
            print(f"[episode end] terminated={terminated}, truncated={truncated}, info={info}")
            break

    return total_reward


# =============================================================================
# DATASET MAP & MAIN
# =============================================================================

DATASET_MAP = {
    "original": "PrimeIntellect/INTELLECT-3-RL",
    "easy_medium": "bicycleman15/intellect_3_code_easy_medium",
    "hard": "bicycleman15/intellect_3_code_hard",
    "very_hard": "bicycleman15/intellect_3_code_very_hard",
}


async def main():
    parser = argparse.ArgumentParser(description="RLEF multi-turn inference demo")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3,
                        help="Maximum number of LLM attempts per problem (default: 3)")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--difficulty", type=str, default="original",
                        choices=["original", "easy_medium", "hard", "very_hard"])
    parser.add_argument("--problem_index", type=int, default=None)
    parser.add_argument("--num-public-tests", type=int, default=3,
                        help="Number of public test cases for feedback (default: 3)")
    parser.add_argument("--eval-timeout-s", type=float, default=10.0,
                        help="Per-test timeout in seconds (default: 10.0)")
    parser.add_argument("--reasoning-effort", type=str, default="high",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-OSS models (default: high)")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming output (requires checkpoint)")
    args = parser.parse_args()

    use_harmony = is_gpt_oss_model(args.model)

    # Streaming setup
    oai_client = None
    model_path = None
    stream = args.stream

    if stream:
        if use_harmony:
            print("Warning: Streaming not supported for Harmony models. Falling back to batch.", file=sys.stderr)
            stream = False
        else:
            model_path = get_checkpoint(args.model)
            if not model_path:
                print(f"No cached checkpoint found for {args.model}.", file=sys.stderr)
                print(f"Run: python create_checkpoint.py --model {args.model}", file=sys.stderr)
                sys.exit(1)
            print(f"[Streaming enabled, using checkpoint: {model_path}]")
            oai_client = OpenAI(
                base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
                api_key=os.environ.get("TINKER_API_KEY", ""),
            )

    if use_harmony:
        print(f"Detected GPT-OSS model: {args.model}")
        print(f"Using Harmony format with reasoning effort: {args.reasoning_effort}")
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokenizer_or_encoding = encoding

        harmony_stop_tokens = encoding.stop_tokens_for_assistant_actions()
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_token_ids=harmony_stop_tokens,
        )
    else:
        print(f"Using standard model: {args.model}")
        tokenizer_or_encoding = AutoTokenizer.from_pretrained(args.model)
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)

    dataset_name = DATASET_MAP[args.difficulty]
    print(f"Using dataset: {dataset_name} (difficulty: {args.difficulty})")
    if args.problem_index is not None:
        print(f"Using problem index: {args.problem_index}")
    print(f"RLEF mode: {args.num_public_tests} public tests, up to {args.max_steps} attempts")
    print()

    env = RLEFCodeEnv(
        system_prompt="",
        max_turns=args.max_steps,
        num_public_tests=args.num_public_tests,
        eval_timeout_s=args.eval_timeout_s,
        dataset_name=dataset_name,
        problem_index=args.problem_index,
    )

    rewards = []
    for ep in range(args.num_episodes):
        print(f"{'=' * 60}")
        print(f"Episode {ep + 1}")
        print(f"{'=' * 60}\n")
        r = await run_episode(
            env, tokenizer_or_encoding, client, sampling_params, args.max_steps,
            use_harmony=use_harmony,
            reasoning_effort=args.reasoning_effort,
            stream=stream,
            model=model_path,
            oai_client=oai_client,
        )
        rewards.append(r)
        print(f"\n[episode {ep + 1} reward] {r:.3f}\n")

    print(f"{'=' * 60}")
    print(f"avg reward: {sum(rewards) / len(rewards):.3f}")
    print(f"solve rate: {sum(1 for r in rewards if r > 0) / len(rewards):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
