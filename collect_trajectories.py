"""Collect multi-turn trajectories and save as HuggingFace dataset.

Usage:
    python collect_trajectories.py \
    --dataset anirudhb11/lcb_v6_feb_may_2025_formatted \
    --model openai/gpt-oss-120b \
    --backend vllm \
    --start-problem 0 \
    --num-problems 2 \
    --num-samples 4 \
    --max-turns 2 \
    --gpu-memory-utilization 0.75 \
    \
    --fast-eval \
    --eval-workers 16 \
    --eval-batch-size 8 \
    --eval-timeout-s 5.0 \
    --push-to-hub bicycleman15/temp

Multi-GPU (launches one vLLM server per GPU, shards prompts across them):
    python collect_trajectories.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --vllm-multi-gpu \
    --vllm-gpu-ids 0,1 \
    --num-problems 10 \
    --num-samples 8 \
    --max-turns 5 \
    \
    --fast-eval \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 1.0 \
    --push-to-hub bicycleman15/temp

For GPT-OSS models (uses Harmony format):
    python collect_trajectories.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model openai/gpt-oss-120b \
    --backend vllm \
    --num-problems 10 \
    --num-samples 8 \
    --max-turns 5 \
    --reasoning-effort medium \
    --fast-eval \
    --push-to-hub bicycleman15/temp

Resume from checkpoint (if previous run failed):
    python collect_trajectories.py \
        --resume-from checkpoints/20260117_143052 \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        ... (same args as original run)

Checkpoints are automatically saved after each generation round to:
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint.pkl
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint_info.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Optional

import requests
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

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

from checkpoint import CheckpointManager, get_checkpoint_dir
from intellect_env import IntellectCodeEnv, step_batch
from utils.gpu_keepalive import GPUKeepAlive
from utils.pass_at_k import compute_pass_at_k
from utils.vllm_multi_gpu import (
    resolve_vllm_gpu_ids,
    build_vllm_server_urls,
    launch_vllm_servers,
    wait_for_vllm_servers,
    register_vllm_shutdown,
)

# Backend imports (conditional)
try:
    import tinker
    from tinker import types as tinker_types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

try:
    from chota_tinker import (
        SamplingClient,
        ServerSamplingClient,
        MultiServerSamplingClient,
        SamplingParams,
        ModelInput,
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def render_trajectory(messages: list[dict], interactions: list[dict], question: str, reward: float, num_turns: int, terminated: bool, truncated: bool) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.2f} | Turns: {num_turns} | Terminated: {terminated} | Truncated: {truncated}")
    lines.append("=" * 80)
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")
    
    if interactions:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Code Interactions ({len(interactions)}):")
        for i, inter in enumerate(interactions):
            lines.append(f"  [{i+1}] Code:\n{inter['code']}")
            lines.append(f"  Output:\n{inter['output']}")
    
    return "\n".join(lines)


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

FINAL_PROMPT = """STOP. Do NOT use <interact> anymore. Your interaction budget is exhausted.

You MUST now output your final solution code wrapped in ```python``` code blocks.

Based on all the information and debugging you have done so far, write your best solution now.

Output ONLY the final ```python``` code block. No more <interact> blocks allowed."""

# For Harmony GPT-OSS models, use SYSTEM_PROMPT as DEVELOPER_INSTRUCTIONS
DEVELOPER_INSTRUCTIONS = SYSTEM_PROMPT


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


@dataclass
class RolloutState:
    """Track state of a single rollout for batched processing."""
    problem_index: int
    sample_index: int
    env: IntellectCodeEnv
    history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    obs: str = ""
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    interaction_timeout_count: int = 0  # Number of interactions that timed out
    eval_timeout_count: int = 0  # Number of test cases that timed out during final eval
    turn_wise_finish_reasons: list = field(default_factory=list)  # finish_reason per assistant turn (for HF columns)


def serialize_rollout_state(state: RolloutState) -> dict:
    """Serialize a RolloutState to a dictionary for checkpointing."""
    return {
        "problem_index": state.problem_index,
        "sample_index": state.sample_index,
        "history": [msg.copy() for msg in state.history],
        "messages": [msg.copy() for msg in state.messages],
        "interactions": [inter.copy() for inter in state.interactions],
        "total_reward": state.total_reward,
        "obs": state.obs,
        "done": state.done,
        "terminated": state.terminated,
        "truncated": state.truncated,
        "interaction_timeout_count": state.interaction_timeout_count,
        "eval_timeout_count": state.eval_timeout_count,
        "turn_wise_finish_reasons": list(state.turn_wise_finish_reasons),
        # Env-related data needed for reconstruction
        "question": state.env.question,
        "tests": state.env.tests,
        "current_turn": state.env.current_turn,
        "env_history": state.env.history.copy() if state.env.history else [],
        "has_interacted": state.env.has_interacted,
    }


def deserialize_rollout_state(data: dict, shared_dataset, args) -> RolloutState:
    """Deserialize a dictionary back to a RolloutState."""
    env = IntellectCodeEnv(
        system_prompt="",
        dataset_name=args.dataset,
        problem_index=data["problem_index"],
        max_turns=args.max_turns,
        dataset=shared_dataset,
    )
    env.reset()
    # Restore env state
    env.current_turn = data["current_turn"]
    env.history = data["env_history"]
    env.has_interacted = data["has_interacted"]
    
    state = RolloutState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        env=env,
        history=data["history"],
        messages=data["messages"],
        interactions=data["interactions"],
        total_reward=data["total_reward"],
        obs=data["obs"],
        done=data["done"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        interaction_timeout_count=data.get("interaction_timeout_count", 0),
        eval_timeout_count=data.get("eval_timeout_count", 0),
        turn_wise_finish_reasons=data.get("turn_wise_finish_reasons", []),
    )
    return state


def create_sampling_client(args):
    """Create sampling client based on backend choice."""
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        if args.vllm_multi_gpu:
            raise ValueError("--vllm-multi-gpu requires --backend vllm")
        service_client = tinker.ServiceClient()
        return service_client.create_sampling_client(base_model=args.model)
    else:  # vllm
        if not VLLM_AVAILABLE:
            raise ImportError("chota_tinker not installed. Install it or use --backend tinker")
        if args.vllm_multi_gpu:
            if args.vllm_server_url:
                raise ValueError("--vllm-server-url cannot be used with --vllm-multi-gpu")
            gpu_ids = resolve_vllm_gpu_ids(args)
            urls = build_vllm_server_urls(args, gpu_ids)
            print(f"Launching vLLM servers for GPUs: {', '.join(gpu_ids)}")
            processes = launch_vllm_servers(args, gpu_ids)
            register_vllm_shutdown(processes)
            wait_for_vllm_servers(urls, args.vllm_server_startup_timeout_s)
            return MultiServerSamplingClient(urls)
        if args.vllm_server_url:
            return ServerSamplingClient(args.vllm_server_url)
        else:
            kwargs = {"gpu_memory_utilization": args.gpu_memory_utilization}
            if getattr(args, "max_model_len", None) is not None:
                kwargs["max_model_len"] = args.max_model_len
            if getattr(args, "tensor_parallel_size", 1) > 1:
                kwargs["tensor_parallel_size"] = args.tensor_parallel_size
            return SamplingClient(args.model, **kwargs)


def create_sampling_params(args, backend: str, harmony_encoding=None):
    """Create sampling params for the chosen backend.
    
    Args:
        args: Command line arguments
        backend: 'tinker' or 'vllm'
        harmony_encoding: If using Harmony (GPT-OSS), pass the encoding to get stop tokens
    """
    stop_sequences = ["</interact>"]
    stop_token_ids = None
    
    # For Harmony models, add Harmony stop tokens
    if harmony_encoding is not None:
        # Use stop_tokens_for_assistant_actions() which returns only <|return|> and <|call|>
        # Do NOT use stop_tokens() which includes <|end|> - that marks the end of ONE message,
        # but the model outputs multiple messages (analysis channel -> final channel)
        stop_token_ids = harmony_encoding.stop_tokens_for_assistant_actions()
    
    if backend == "tinker":
        params = tinker_types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=stop_sequences,
        )
        if stop_token_ids:
            params = tinker_types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.95,
                stop=stop_sequences,
                stop_token_ids=stop_token_ids,
            )
        return params
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=stop_sequences,
            stop_token_ids=stop_token_ids,
        )


def build_prompt(state: RolloutState, tokenizer, max_turns: int) -> list[int]:
    """Build tokenized prompt from rollout state (standard HF tokenizer path).
    
    On the last turn (current_turn == max_turns - 1), appends FINAL_PROMPT
    to force the model to output final code.
    """
    is_last_turn = state.env.current_turn == max_turns - 1
    if is_last_turn:
        obs_for_prompt = f"{state.obs}\n\n{FINAL_PROMPT}" if state.obs else FINAL_PROMPT
    else:
        obs_for_prompt = state.obs
    
    messages = state.history + [{"role": "user", "content": obs_for_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


def build_prompt_harmony(state: RolloutState, encoding, max_turns: int) -> list[int]:
    """Build tokenized prompt from rollout state using Harmony encoding (GPT-OSS path).
    
    On the last turn (current_turn == max_turns - 1), appends FINAL_PROMPT
    to force the model to output final code.
    """
    is_last_turn = state.env.current_turn == max_turns - 1
    if is_last_turn:
        obs_for_prompt = f"{state.obs}\n\n{FINAL_PROMPT}" if state.obs else FINAL_PROMPT
    else:
        obs_for_prompt = state.obs
    
    conversation = build_harmony_conversation(state.history, obs_for_prompt, encoding)
    return encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)


def postprocess_response(response: str) -> str:
    """Postprocess LLM response."""
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    return response


def _truncated_by_token_limit(finish_reason: str | None) -> bool:
    """True if generation stopped due to max token limit."""
    return (finish_reason or "").lower() in ("length", "length_capped", "max_tokens")


def truncate_interaction_output(
    output_text: str,
    tokenizer_or_encoding,
    max_tokens: int,
    use_harmony: bool = False,
) -> str:
    """If output_text has more than max_tokens tokens, keep only the last max_tokens and prepend a notice.
    
    Args:
        output_text: The text to potentially truncate
        tokenizer_or_encoding: HF tokenizer (standard) or HarmonyEncoding (GPT-OSS)
        max_tokens: Maximum tokens to keep (from the end)
        use_harmony: If True, use HarmonyEncoding API; otherwise use HF tokenizer API
    """
    if not output_text or max_tokens <= 0:
        return output_text
    
    if use_harmony:
        # HarmonyEncoding API: disallowed_special=() to encode special token strings as regular text
        ids = tokenizer_or_encoding.encode(output_text, disallowed_special=())
    else:
        # HF tokenizer API
        ids = tokenizer_or_encoding.encode(output_text, add_special_tokens=False)
    
    if len(ids) <= max_tokens:
        return output_text
    
    keep = ids[-max_tokens:]
    
    if use_harmony:
        # HarmonyEncoding decode
        tail_text = tokenizer_or_encoding.decode(keep)
    else:
        # HF tokenizer decode
        tail_text = tokenizer_or_encoding.decode(keep, skip_special_tokens=False)
    
    return f"Output too long, showing only last {max_tokens} tokens.\n{tail_text}"


def maybe_truncate_obs(obs: str, tokenizer_or_encoding, max_interaction_output_tokens: Optional[int], use_harmony: bool = False) -> str:
    """If obs is <output>...</output> and body exceeds max_interaction_output_tokens, truncate to tail and prepend notice.
    
    Args:
        obs: The observation string (typically <output>...</output>)
        tokenizer_or_encoding: HF tokenizer (standard) or HarmonyEncoding (GPT-OSS)
        max_interaction_output_tokens: Maximum tokens to keep in the output body
        use_harmony: If True, use HarmonyEncoding API; otherwise use HF tokenizer API
    """
    if obs is None or not obs or max_interaction_output_tokens is None or max_interaction_output_tokens <= 0:
        return obs
    match = re.match(r"^<output>\n(.*)</output>\s*$", obs, re.DOTALL)
    if not match:
        return obs
    body = match.group(1)
    truncated = truncate_interaction_output(body, tokenizer_or_encoding, max_interaction_output_tokens, use_harmony=use_harmony)
    if truncated is body:
        return obs
    return f"<output>\n{truncated}</output>"


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params) -> list[tuple[list[int], Optional[str]]]:
    """Batch sample using tinker (via async gather). Returns list of (tokens, finish_reason)."""
    max_tokens = getattr(sampling_params, "max_tokens", None)

    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        seq = result.sequences[0]
        tokens = seq.tokens
        fr = getattr(seq, "finish_reason", None)
        if fr is None and max_tokens is not None and len(tokens) >= max_tokens:
            fr = "length"
        return (tokens, fr)

    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


async def sample_batch_tinker_harmony(client, prompts: list[list[int]], sampling_params, encoding) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """Batch sample using tinker with Harmony parsing.

    Returns list of (response_content, channel, analysis_content, finish_reason) tuples.
    """
    token_results = await sample_batch_tinker(client, prompts, sampling_params)

    parsed_results = []
    for tokens, finish_reason in token_results:
        content, channel, analysis = parse_harmony_response(tokens, encoding)
        # Postprocess: if stopped by </interact>, append it back
        if "<interact>" in content and "</interact>" not in content:
            content += "</interact>"
        parsed_results.append((content, channel, analysis, finish_reason))

    return parsed_results


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params, show_progress: bool = False) -> list[tuple[str, Optional[str]]]:
    """Batch sample using vLLM. Returns list of (text, finish_reason)."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    max_tokens = getattr(sampling_params, "max_tokens", None)
    # Pass show_progress for MultiServerSamplingClient (ignored by other clients)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        # Fallback for clients that don't support show_progress
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    out = []
    for r in results:
        seq = r.sequences[0]
        text = seq.text
        fr = getattr(seq, "finish_reason", None)
        if fr is None and max_tokens is not None and len(seq.tokens) >= max_tokens:
            fr = "length"
        out.append((text, fr))
    return out


def sample_batch_vllm_harmony(client, prompts: list[list[int]], sampling_params, encoding, show_progress: bool = False) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """Batch sample using vLLM with Harmony parsing.

    Returns list of (response_content, channel, analysis_content, finish_reason) tuples.
    """
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    max_tokens = getattr(sampling_params, "max_tokens", None)
    # Pass show_progress for MultiServerSamplingClient (ignored by other clients)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        # Fallback for clients that don't support show_progress
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)

    parsed_results = []
    for result in results:
        seq = result.sequences[0]
        tokens = seq.tokens
        fr = getattr(seq, "finish_reason", None)
        if fr is None and max_tokens is not None and len(tokens) >= max_tokens:
            fr = "length"
        content, channel, analysis = parse_harmony_response(tokens, encoding)
        # Postprocess: if stopped by </interact>, append it back
        if "<interact>" in content and "</interact>" not in content:
            content += "</interact>"
        parsed_results.append((content, channel, analysis, fr))

    return parsed_results


def run_batched_rollouts(
    args,
    client,
    tokenizer_or_encoding,
    sampling_params,
    checkpoint_manager: Optional[CheckpointManager] = None,
    use_harmony: bool = False,
) -> list[list[dict[str, Any]]]:
    """Run batched rollouts across all problems and samples.
    
    Args:
        args: Command line arguments
        client: Sampling client (tinker or vLLM)
        tokenizer_or_encoding: HF tokenizer (standard) or HarmonyEncoding (GPT-OSS)
        sampling_params: Sampling parameters
        checkpoint_manager: Optional checkpoint manager for resuming
        use_harmony: If True, use Harmony format for GPT-OSS models
    """
    # Load dataset ONCE before creating environments
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/") or args.dataset.startswith("anirudhb11/intellect_"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
    elif args.dataset.__contains__('lcb'):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="test")
    else:
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, "code", split="train")
    
    # Select slice based on start_problem and num_problems
    end_problem = min(args.start_problem + args.num_problems, len(full_dataset))
    shared_dataset = full_dataset.select(range(args.start_problem, end_problem))
    actual_num_problems = len(shared_dataset)
    print(f"Dataset loaded with {len(full_dataset)} total problems.")
    print(f"Selected slice: problems {args.start_problem} to {end_problem - 1} ({actual_num_problems} problems)")
    
    # Update args.num_problems to reflect actual slice size
    if actual_num_problems < args.num_problems:
        print(f"Warning: Requested {args.num_problems} problems but only {actual_num_problems} available in slice.")
        args.num_problems = actual_num_problems
    
    # For Harmony, build the system and developer content once
    if use_harmony:
        harmony_system_content = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort[args.reasoning_effort.upper()])
            .with_conversation_start_date(date.today().isoformat())
        )
        harmony_developer_content = (
            DeveloperContent.new()
            .with_instructions(DEVELOPER_INSTRUCTIONS)
        )
    
    # Track generation round for checkpointing
    generation_round = 0
    active_states: list[RolloutState] = []
    completed_states: list[RolloutState] = []
    skip_generation = False  # Flag to skip generation on first iteration after resume
    
    # Resume from checkpoint if available
    if checkpoint_manager and checkpoint_manager.has_checkpoint():
        print(f"\nResuming from checkpoint: {checkpoint_manager.checkpoint_dir}")
        checkpoint_data = checkpoint_manager.load()
        
        # Verify args match
        for warning in checkpoint_manager.verify_args({
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "dataset": args.dataset,
            "model": args.model,
        }):
            print(warning)
        
        # Restore states
        generation_round = checkpoint_data.current_round
        print(f"  Resuming from generation round {generation_round}")
        print(f"  Restoring {len(checkpoint_data.active_states_data)} active states...")
        print(f"  Restoring {len(checkpoint_data.completed_states_data)} completed states...")
        
        for state_data in checkpoint_data.active_states_data:
            state = deserialize_rollout_state(state_data, shared_dataset, args)
            active_states.append(state)
        
        for state_data in checkpoint_data.completed_states_data:
            state = deserialize_rollout_state(state_data, shared_dataset, args)
            completed_states.append(state)
        
        print(f"  Successfully restored {len(active_states)} active and {len(completed_states)} completed states.")
        
        # Skip generation on first iteration - checkpoint was saved after generation but before step_batch
        skip_generation = True
    else:
        # Initialize all rollout states from scratch
        for problem_idx in range(args.num_problems):
            for sample_idx in range(args.num_samples):
                env = IntellectCodeEnv(
                    system_prompt="",
                    dataset_name=args.dataset,
                    problem_index=problem_idx,
                    max_turns=args.max_turns,
                    dataset=shared_dataset,  # Pass pre-loaded dataset
                )
                obs, info = env.reset()
                
                if use_harmony:
                    # Harmony format: system content + developer content (instructions)
                    history = [
                        {"role": "system", "content": harmony_system_content},
                        {"role": "developer", "content": harmony_developer_content},
                    ]
                    # For messages (user-facing record), we store a simplified version
                    messages = [
                        {"role": "system", "content": f"[Harmony format] Reasoning effort: {args.reasoning_effort}"},
                        {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
                        {"role": "user", "content": obs},
                    ]
                else:
                    # Standard format: single system message with all instructions
                    history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
                
                state = RolloutState(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    env=env,
                    history=history,
                    messages=messages,
                    obs=obs,
                )
                active_states.append(state)
    
    problems_completed = set()
    # Count already completed problems from checkpoint
    for state in completed_states:
        problem_samples_done = sum(1 for s in completed_states if s.problem_index == state.problem_index)
        if problem_samples_done == args.num_samples:
            problems_completed.add(state.problem_index)
    
    pbar = tqdm(total=args.num_problems, desc="Problems completed", initial=len(problems_completed))
    
    while active_states:
        # Skip generation if resuming (checkpoint was saved after generation, before step_batch)
        if skip_generation:
            print(f"\n[Resuming round {generation_round}] Skipping generation, going directly to code execution...")
            # Extract the last assistant response from each state's history
            processed_responses = []
            for state in active_states:
                # The last message in history should be the assistant's response
                last_assistant_msg = None
                for msg in reversed(state.history):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg["content"]
                        break
                if last_assistant_msg is None:
                    raise ValueError(f"No assistant response found in history for state {state.problem_index}:{state.sample_index}")
                processed_responses.append(last_assistant_msg)
            skip_generation = False  # Only skip once
        else:
            generation_round += 1
            print(f"\n[Generation round {generation_round}] Processing {len(active_states)} active states...")
            
            # Build prompts for all active states
            # On the last turn, FINAL_PROMPT is appended to force final code output
            if use_harmony:
                prompts = [build_prompt_harmony(s, tokenizer_or_encoding, args.max_turns) for s in active_states]
            else:
                prompts = [build_prompt(s, tokenizer_or_encoding, args.max_turns) for s in active_states]
            
            # Batch sample
            if use_harmony:
                # Harmony path: parse tokens with Harmony encoding
                if args.backend == "tinker":
                    harmony_results = asyncio.run(sample_batch_tinker_harmony(
                        client, prompts, sampling_params, tokenizer_or_encoding
                    ))
                else:
                    harmony_results = sample_batch_vllm_harmony(
                        client, prompts, sampling_params, tokenizer_or_encoding, show_progress=args.vllm_multi_gpu
                    )
                # harmony_results is list of (content, channel, analysis, finish_reason)
                responses = harmony_results
            elif args.backend == "tinker":
                token_results = asyncio.run(sample_batch_tinker(client, prompts, sampling_params))
                responses = [(tokenizer_or_encoding.decode(tokens, skip_special_tokens=True), fr) for tokens, fr in token_results]
            else:
                responses = sample_batch_vllm(client, prompts, sampling_params, show_progress=args.vllm_multi_gpu)
            
            # Process responses and step environments
            processed_responses = []
            for i, state in enumerate(active_states):
                if use_harmony:
                    # Unpack Harmony result (content, channel, analysis, finish_reason)
                    response, channel, analysis, finish_reason = responses[i]
                else:
                    response, finish_reason = responses[i]
                    response = postprocess_response(response)
                    channel = None
                    analysis = None
                
                # Record finish_reason for this assistant turn (for HF columns)
                state.turn_wise_finish_reasons.append(finish_reason)
                
                # On the last turn, include FINAL_PROMPT in history to reflect what was sent
                is_last_turn = state.env.current_turn == args.max_turns - 1
                if is_last_turn:
                    obs_for_history = f"{state.obs}\n\n{FINAL_PROMPT}" if state.obs else FINAL_PROMPT
                    # Also update the last user message in messages to include FINAL_PROMPT
                    # (the obs was already appended to messages in the previous iteration)
                    if state.messages and state.messages[-1]["role"] == "user":
                        state.messages[-1]["content"] = obs_for_history
                else:
                    obs_for_history = state.obs
                
                # Update history and messages
                state.history.append({"role": "user", "content": obs_for_history})
                
                if use_harmony:
                    # For Harmony, track channel info in history
                    # If we have both analysis and final, add both to history
                    if analysis and channel == "final":
                        state.history.append({"role": "assistant", "content": analysis, "channel": "analysis"})
                    state.history.append({"role": "assistant", "content": response, "channel": channel})
                    
                    # For messages (user-facing record), include analysis as a note if present
                    if analysis:
                        state.messages.append({
                            "role": "assistant", 
                            "content": f"[Analysis (internal CoT)]\n{analysis}\n\n[Response (channel: {channel})]\n{response}"
                        })
                    else:
                        state.messages.append({"role": "assistant", "content": response})
                else:
                    state.history.append({"role": "assistant", "content": response})
                    state.messages.append({"role": "assistant", "content": response})
                
                processed_responses.append(response)

            # Save checkpoint BEFORE code execution (so we can retry if execution fails)
            if checkpoint_manager:
                checkpoint_manager.save(
                    active_states_data=[serialize_rollout_state(s) for s in active_states],
                    completed_states_data=[serialize_rollout_state(s) for s in completed_states],
                    current_round=generation_round,
                    total_rounds=args.max_turns * args.num_problems * args.num_samples,  # Rough estimate
                )

        still_active = []

        # Use GPUKeepAlive during code execution/evaluation to prevent SLURM idle GPU termination
        with GPUKeepAlive():
            if args.fast_eval:
                step_results = step_batch(
                    [s.env for s in active_states],
                    processed_responses,
                    eval_workers=args.eval_workers,
                    eval_batch_size=args.eval_batch_size,
                    eval_timeout_s=args.eval_timeout_s,
                    show_progress=True,
                )
            else:
                step_results = [s.env.step(r) for s, r in zip(active_states, processed_responses)]

        max_out_tokens = getattr(args, "max_interaction_output_tokens", None)
        for state, _response, (obs, reward, terminated, truncated, info) in zip(
            active_states, processed_responses, step_results
        ):
            
            # Step environment
            state.total_reward += reward
            state.terminated = terminated
            state.truncated = truncated
            
            # Track timeout counts from info
            if info.get("interaction_timed_out", False):
                state.interaction_timeout_count += 1
            if info.get("eval_timeout_count", 0) > 0:
                state.eval_timeout_count = info["eval_timeout_count"]
            
            # Truncate interaction output if over limit (guardrail against huge terminal output)
            if obs and max_out_tokens is not None and max_out_tokens > 0:
                obs = maybe_truncate_obs(obs, tokenizer_or_encoding, max_out_tokens, use_harmony=use_harmony)
            
            # Update interactions (truncation already applied to obs that goes into history)
            if state.env.history and len(state.env.history) > len(state.interactions):
                state.interactions = state.env.history.copy()
            
            if obs:
                state.messages.append({"role": "user", "content": obs})
            
            if terminated or truncated:
                state.done = True
                completed_states.append(state)
                
                # Check if all samples for this problem are done
                problem_samples_done = sum(
                    1 for s in completed_states if s.problem_index == state.problem_index
                )
                if problem_samples_done == args.num_samples and state.problem_index not in problems_completed:
                    problems_completed.add(state.problem_index)
                    pbar.update(1)
            else:
                state.obs = obs
                still_active.append(state)
        
        active_states = still_active
    
    pbar.close()
    
    # Organize results by problem
    all_trajectories: list[list[dict]] = [[] for _ in range(args.num_problems)]
    for state in completed_states:
        turn_wise_finish_reason = list(state.turn_wise_finish_reasons)
        turn_wise_truncated_by_token_limit = [_truncated_by_token_limit(fr) for fr in state.turn_wise_finish_reasons]
        num_assistant_messages_truncated = sum(turn_wise_truncated_by_token_limit)
        num_assistant_messages_not_truncated = len(turn_wise_truncated_by_token_limit) - num_assistant_messages_truncated
        traj = {
            "question": state.env.question,
            "messages": state.messages,
            "num_turns": state.env.current_turn,
            "final_reward": state.total_reward,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "interactions": state.interactions,
            "tests": state.env.tests,
            "interaction_timeout_count": state.interaction_timeout_count,
            "eval_timeout_count": state.eval_timeout_count,
            "turn_wise_finish_reason": turn_wise_finish_reason,
            "turn_wise_truncated_by_token_limit": turn_wise_truncated_by_token_limit,
            "num_assistant_messages_truncated": num_assistant_messages_truncated,
            "num_assistant_messages_not_truncated": num_assistant_messages_not_truncated,
        }
        all_trajectories[state.problem_index].append(traj)
    
    return all_trajectories

def main(args):
    print(f"=" * 60)
    print(f"Collecting trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    if is_gpt_oss_model(args.model):
        print(f"  Format: Harmony (GPT-OSS)")
        print(f"  Reasoning effort: {args.reasoning_effort}")
    print(f"  Problem range: {args.start_problem} to {args.start_problem + args.num_problems - 1} ({args.num_problems} problems)")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Output: {args.output_dir}")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print(f"=" * 60)
    
    # Setup checkpoint manager
    if args.resume_from:
        checkpoint_dir = args.resume_from
        print(f"\nResuming from checkpoint directory: {checkpoint_dir}")
    else:
        checkpoint_dir = get_checkpoint_dir()
        print(f"\nCheckpoint directory: {checkpoint_dir}")
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        args_dict={
            "dataset": args.dataset,
            "model": args.model,
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    )
    
    # Detect if we're using a GPT-OSS model that requires Harmony format
    use_harmony = is_gpt_oss_model(args.model)
    
    if use_harmony:
        print(f"\nDetected GPT-OSS model: {args.model}")
        print(f"Using Harmony format with reasoning effort: {args.reasoning_effort}")
        
        # Load Harmony encoding instead of HuggingFace tokenizer
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokenizer_or_encoding = encoding
        
        # Create sampling params with Harmony stop tokens
        sampling_params = create_sampling_params(args, args.backend, harmony_encoding=encoding)

    else:
        print(f"\nUsing standard model: {args.model}")
        tokenizer_or_encoding = AutoTokenizer.from_pretrained(args.model)
        sampling_params = create_sampling_params(args, args.backend)
    
    # Initialize client based on backend
    print(f"\nInitializing {args.backend} client...")
    sampling_client = create_sampling_client(args)
    
    print(f"\nCollecting trajectories for {args.num_problems} problems (batched)...")
    
    # Run batched rollouts
    all_trajectories = run_batched_rollouts(
        args=args,
        client=sampling_client,
        tokenizer_or_encoding=tokenizer_or_encoding,
        sampling_params=sampling_params,
        checkpoint_manager=checkpoint_manager,
        use_harmony=use_harmony,
    )
    
    # Flatten into rows (one per trajectory)
    rows = []
    all_results = []
    
    for problem_idx, problem_trajectories in enumerate(all_trajectories):
        problem_results = []
        
        for traj_idx, traj in enumerate(problem_trajectories):
            is_successful = traj["final_reward"] > 0
            problem_results.append(is_successful)
            
            rows.append({
                "problem_id": problem_idx,
                "trajectory_id": traj_idx,
                "question": traj["question"],
                "messages": json.dumps(traj["messages"]),
                "num_turns": traj["num_turns"],
                "final_reward": traj["final_reward"],
                "terminated": traj["terminated"],
                "truncated": traj["truncated"],
                "interactions": json.dumps(traj["interactions"]),
                "tests": json.dumps(traj["tests"]),
                "is_successful": is_successful,
                "interaction_timeout_count": traj["interaction_timeout_count"],
                "eval_timeout_count": traj["eval_timeout_count"],
                "turn_wise_finish_reason": traj["turn_wise_finish_reason"],
                "turn_wise_truncated_by_token_limit": traj["turn_wise_truncated_by_token_limit"],
                "num_assistant_messages_truncated": traj["num_assistant_messages_truncated"],
                "num_assistant_messages_not_truncated": traj["num_assistant_messages_not_truncated"],
                "rendered": render_trajectory(
                    traj["messages"], traj["interactions"], traj["question"],
                    traj["final_reward"], traj["num_turns"], traj["terminated"], traj["truncated"]
                ),
            })
        
        all_results.append(problem_results)
    
    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_2 = compute_pass_at_k(all_results, k=2)
    pass_at_4 = compute_pass_at_k(all_results, k=4)
    pass_at_8 = compute_pass_at_k(all_results, k=min(8, args.num_samples))
    
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {len(rows)}")
    print(f"  pass@1: {pass_at_1:.4f}")
    print(f"  pass@2: {pass_at_2:.4f}")
    print(f"  pass@4: {pass_at_4:.4f}")
    print(f"  pass@8: {pass_at_8:.4f}")
    print(f"{'=' * 60}")
    
    dataset = Dataset.from_list(rows)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "start_problem": args.start_problem,
        "num_problems": args.num_problems,
        "num_samples": args.num_samples,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "pass_at_1": pass_at_1,
        "pass_at_2": pass_at_2,
        "pass_at_4": pass_at_4,
        "pass_at_8": pass_at_8,
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved dataset to: {args.output_dir}")
    print(f"Saved metadata to: {metadata_path}")
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        **metadata,
        "num_successful_trajectories": sum(1 for r in rows if r["is_successful"]),
        "problems_solved": sum(1 for pr in all_results if any(pr)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_path}")
    
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return dataset, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect multi-turn trajectories for code problems")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL", 
                                 "anirudhb11/lcb_v6_feb_may_2025_formatted", "anirudhb11/lcb_v6_feb_may_2025_formatted_hardest_to_easiest", "anirudhb11/intellect_3_code_very_hard_top_400_hardest", "anirudhb11/qwen3_4b_instruct_top_400_hardest_interations_10_turns"])
    parser.add_argument("--start-problem", type=int, default=0,
                        help="Starting problem index for dataset slicing (default: 0)")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    # Checkpointing options
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. checkpoints/20260117_143052)")
    
    # Backend options
    parser.add_argument("--backend", type=str, default="vllm", choices=["tinker", "vllm"],
                        help="Inference backend: 'tinker' or 'vllm' (default: vllm)")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                        help="URL for vLLM server (e.g. http://localhost:8000). If not set, uses local vLLM.")
    parser.add_argument("--vllm-multi-gpu", action="store_true",
                        help="Launch one local vLLM server per GPU and shard prompts across them.")
    parser.add_argument("--vllm-gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for vLLM servers (default: all visible GPUs).")
    parser.add_argument("--vllm-server-base-port", type=int, default=8000,
                        help="Base port for vLLM servers; ports increment per GPU.")
    parser.add_argument("--vllm-server-host", type=str, default="127.0.0.1",
                        help="Host to bind vLLM servers (default: 127.0.0.1).")
    parser.add_argument("--vllm-server-startup-timeout-s", type=float, default=300.0,
                        help="Seconds to wait for vLLM servers to be ready.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for local vLLM or vLLM servers (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max sequence length for vLLM; lower than model default to reduce KV cache (e.g. 16384 or 131072)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1). Use >1 to shard large models across GPUs.")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 5.0)")
    parser.add_argument("--max-interaction-output-tokens", type=int, default=4000,
                        help="Truncate terminal output in <output> to this many tokens (keep tail); prepend 'Output too long, showing only last X tokens' to avoid context overflow (default: 4000).")
    
    # Harmony/GPT-OSS options
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-OSS models using Harmony format (default: medium)")
    
    args = parser.parse_args()
    main(args)
