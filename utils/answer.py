"""Answer extraction and checking utilities using math-verify."""

from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig


def check_answer(response: str, target: str) -> bool:
    """
    Check if the response contains the correct answer using math-verify.
    
    Uses robust mathematical expression parsing and comparison from math-verify,
    which handles LaTeX, symbolic expressions, sets, intervals, and more.
    
    Args:
        response: The model's response text
        target: The expected answer (gold)
        
    Returns:
        True if the answer is correct, False otherwise
    """
    try:
        # Parse gold answer (typically cleaner format)
        gold_parsed = parse(
            target,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]
        )
        
        # Parse model response with more lenient config for model outputs
        answer_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,  # Prioritize \boxed{} answers
                ),
                ExprExtractionConfig()
            ]
        )
        
        # Verify: order matters - gold first, then answer
        return verify(gold_parsed, answer_parsed)
    except Exception:
        # Fallback to simple string comparison if parsing fails
        return _simple_check(response, target)


def _simple_check(response: str, target: str) -> bool:
    """Simple fallback answer checking."""
    target_normalized = _normalize(target)
    response_normalized = _normalize(response)
    return target_normalized in response_normalized


def _normalize(text: str) -> str:
    """Normalize text for simple comparison."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = text.replace(" ", "")
    text = text.replace("\\,", "")
    return text
