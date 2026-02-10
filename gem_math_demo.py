"""
Code environment demo using GEM with LLM agent.

Uses tinker API for sampling.

Usage:
    python gem_math_demo.py --model Qwen/Qwen3-4B-Instruct-2507 --difficulty very_hard --problem_index 32 --fast-eval --eval-timeout-s 5.0 --max_tokens 4096 --max_steps 3

    # For GPT-OSS models (uses Harmony format):
    python gem_math_demo.py --model openai/gpt-oss-120b --difficulty very_hard --problem_index 32 --fast-eval --eval-timeout-s 5.0 --max_tokens 4096 --max_steps 3 --reasoning-effort medium

possible models:
deepseek-ai/DeepSeek-V3.1
Qwen/Qwen3-235B-A22B-Instruct-2507
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-4B-Instruct-2507
openai/gpt-oss-120b (uses Harmony format)
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
from intellect_env import IntellectCodeEnv, step_batch

# Harmony library for GPT-OSS models
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
# For standard models, SYSTEM_PROMPT goes in the "system" role.
# For GPT-OSS models using Harmony, these instructions go in the "developer" role,
# while the "system" role contains model identity, reasoning effort, etc.

DEVELOPER_INSTRUCTIONS = """You are a helpful coding assistant.

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

# For backward compatibility with non-Harmony models
SYSTEM_PROMPT = DEVELOPER_INSTRUCTIONS

FINAL_PROMPT = """STOP. Do NOT use <interact> anymore. Your interaction budget is exhausted.

You MUST now output your final solution code wrapped in ```python``` code blocks.

Based on all the information and debugging you have done so far, write your best solution now.

No more <interact> blocks allowed."""
# Output ONLY the final ```python``` code block. No more <interact> blocks allowed.


# =============================================================================
# HARMONY FORMAT HELPERS (for GPT-OSS models)
# =============================================================================

def build_harmony_conversation(history: list, obs: str, encoding) -> Conversation:
    """
    Build a Harmony Conversation from the history list and current observation.
    
    The history format for Harmony is different:
    - history[0] is the SystemContent message
    - history[1] is the DeveloperContent message  
    - Subsequent entries are user/assistant messages
    
    For assistant messages, we track channel info to properly drop analysis on subsequent turns.
    According to Harmony docs: "you should drop any previous CoT content on subsequent sampling 
    if the responses by the assistant ended in a message to the final channel."
    """
    messages = []
    
    for i, entry in enumerate(history):
        role = entry["role"]
        content = entry["content"]
        
        if role == "system":
            # This is the Harmony SystemContent
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, content))
        elif role == "developer":
            # This is the Harmony DeveloperContent with our instructions
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
        elif role == "user":
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))
        elif role == "assistant":
            channel = entry.get("channel", "final")
            
            # Check if this is an analysis message that should be dropped
            # We drop analysis if:
            # 1. This message is in the analysis channel, AND
            # 2. The next assistant message (if any) ended in final channel
            # For simplicity, we only keep analysis messages if they're the last assistant message
            # and didn't end with a final response
            should_drop_analysis = False
            if channel == "analysis":
                # Look ahead to see if there's a subsequent final message
                for j in range(i + 1, len(history)):
                    if history[j]["role"] == "assistant":
                        if history[j].get("channel") == "final":
                            should_drop_analysis = True
                        break
            
            if not should_drop_analysis:
                msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content)
                msg = msg.with_channel(channel)
                messages.append(msg)
    
    # Add the current observation as a user message
    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, obs))
    
    return Conversation.from_messages(messages)


def parse_harmony_response(tokens: list, encoding) -> tuple[str, str, Optional[str]]:
    """
    Parse the Harmony response tokens and extract content by channel.
    
    The model may output:
    - analysis (chain of thought) followed by final (user-facing response)
    - Just analysis (if it needs to do a tool call or is still reasoning)
    - Just final
    
    Returns:
        tuple of (user_facing_content, channel, analysis_content)
        - user_facing_content: The content to show/use (from 'final' if available, else from other channels)
        - channel: The channel of the user_facing_content ('final', 'analysis', or 'commentary')  
        - analysis_content: The chain-of-thought content (if any), for logging purposes only
    """
    parsed_messages = encoding.parse_messages_from_completion_tokens(
        tokens, 
        role=HarmonyRole.ASSISTANT,
        strict=False  # Be tolerant of malformed headers
    )
    
    # Collect content by channel
    analysis_content = None
    final_content = None
    commentary_content = None
    
    for msg in parsed_messages:
        channel = msg.channel or "final"
        # Extract text content from the message
        text_parts = []
        for content_item in msg.content:
            if hasattr(content_item, 'text'):
                text_parts.append(content_item.text)
        
        combined_text = "\n".join(text_parts) if text_parts else ""
        
        if channel == "final":
            final_content = combined_text
        elif channel == "analysis":
            analysis_content = combined_text
        elif channel == "commentary":
            commentary_content = combined_text
    
    # Determine user-facing content (prefer final > commentary > analysis)
    if final_content:
        return final_content, "final", analysis_content
    elif commentary_content:
        return commentary_content, "commentary", analysis_content
    elif analysis_content:
        return analysis_content, "analysis", None
    else:
        # Fallback: try to decode raw tokens
        try:
            raw_text = encoding.decode(tokens)
            return raw_text, "unknown", None
        except Exception:
            return "", "unknown", None


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
    
    # if stopped by </interact>, append it back so regex can match
    if "<interact>" in response_content and "</interact>" not in response_content:
        response_content += "</interact>"
    
    return response_content, channel, analysis_content


async def get_llm_action(obs: str, history: list, tokenizer, client, sampling_params) -> str:
    """Get LLM action using standard chat template (non-Harmony models)."""
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


async def run_episode(env, tokenizer_or_encoding, client, sampling_params, max_steps: int = 5, 
                      fast_eval: bool = False, eval_workers: int = 8, 
                      eval_batch_size: int = 8, eval_timeout_s: float = 5.0,
                      use_harmony: bool = False, reasoning_effort: str = "high"):
    """
    Run a single episode.
    
    Args:
        tokenizer_or_encoding: Either a HuggingFace tokenizer (standard) or HarmonyEncoding (GPT-OSS)
        use_harmony: If True, use Harmony format for GPT-OSS models
        reasoning_effort: For Harmony models, the reasoning effort level (low/medium/high)
    """
    obs, info = env.reset()
    total_reward = 0
    
    # Track timeouts
    interaction_timeout_count = 0
    eval_timeout_count = 0
    
    if use_harmony:
        # Build Harmony-style history with system and developer messages
        # SystemContent: model identity, reasoning, dates, channels
        system_content = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort[reasoning_effort.upper()])
            .with_conversation_start_date(date.today().isoformat())
        )
        # DeveloperContent: our instructions (what was previously the "system prompt")
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
        # Standard format: single system message with all instructions
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"[system]\n{SYSTEM_PROMPT}\n")
    
    print(f"[user]\n{obs}\n")
    
    for step in range(max_steps):
        # On the last turn, add the final prompt to force the model to output final code
        is_last_turn = (step == max_steps - 1)
        if is_last_turn:
            obs_for_llm = f"{obs}\n\n{FINAL_PROMPT}" if obs else FINAL_PROMPT
            print(f"[user] (final turn - termination prompt appended)\n{obs_for_llm}\n")
        else:
            obs_for_llm = obs
        
        if use_harmony:
            action, channel, analysis = await get_llm_action_harmony(
                obs_for_llm, history, tokenizer_or_encoding, client, sampling_params
            )
            
            # Log the chain-of-thought if present (for debugging, not user-facing)
            if analysis:
                print(f"[assistant] (channel: analysis) [INTERNAL COT - not shown to user]")
                print(f"{analysis[:500]}{'...' if len(analysis) > 500 else ''}\n")
            
            print(f"[assistant] (channel: {channel})\n{action}\n")
            
            # Add to history with channel info
            # Note: For Harmony, we drop 'analysis' channel content on subsequent turns
            # if the response ended with 'final'. The build_harmony_conversation handles this.
            history.append({"role": "user", "content": obs_for_llm})
            # If we have both analysis and final, add both to history so we can properly
            # drop analysis later when building the next conversation
            if analysis and channel == "final":
                history.append({"role": "assistant", "content": analysis, "channel": "analysis"})
            history.append({"role": "assistant", "content": action, "channel": channel})
        else:
            action = await get_llm_action(obs_for_llm, history, tokenizer_or_encoding, client, sampling_params)
            print(f"[assistant]\n{action}\n")
            
            # Use obs_for_llm in history to reflect what was actually sent to the model
            history.append({"role": "user", "content": obs_for_llm})
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
        # Use stop_tokens_for_assistant_actions() which returns only <|return|> and <|call|>
        # Do NOT use stop_tokens() which includes <|end|> - that marks the end of ONE message,
        # but the model outputs multiple messages (analysis channel -> final channel)
        # We also need to stop on </interact> for our interaction format
        harmony_stop_tokens = encoding.stop_tokens_for_assistant_actions()
        # Convert to strings for the sampling params (we'll also add </interact>)
        stop_sequences = ["</interact>"]
        
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=stop_sequences,
            stop_token_ids=harmony_stop_tokens,
        )
    else:
        print(f"Using standard model: {args.model}")
        tokenizer_or_encoding = AutoTokenizer.from_pretrained(args.model)
        
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=["</interact>"],
        )
    
    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(base_model=args.model)
    
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
            env, tokenizer_or_encoding, client, sampling_params, args.max_steps,
            fast_eval=args.fast_eval,
            eval_workers=args.eval_workers,
            eval_batch_size=args.eval_batch_size,
            eval_timeout_s=args.eval_timeout_s,
            use_harmony=use_harmony,
            reasoning_effort=args.reasoning_effort,
        )
        rewards.append(r)
    
    print(f"avg reward: {sum(rewards)/len(rewards):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
