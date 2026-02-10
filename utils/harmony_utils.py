"""Shared utilities for GPT-OSS models using Harmony format.

This module provides common functionality for working with OpenAI's Harmony
format used by GPT-OSS models.
"""

from typing import Optional

# Harmony library imports - re-exported for convenience
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

# Re-export all Harmony imports for convenience
__all__ = [
    # Detection
    "is_gpt_oss_model",
    # Parsing
    "parse_harmony_response",
    # Re-exported from openai_harmony
    "load_harmony_encoding",
    "HarmonyEncodingName",
    "HarmonyRole",
    "HarmonyMessage",
    "Conversation",
    "DeveloperContent",
    "SystemContent",
    "ReasoningEffort",
]


def is_gpt_oss_model(model_name: str) -> bool:
    """Check if the model is a GPT-OSS model that requires Harmony format."""
    return "gpt-oss" in model_name.lower()


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
