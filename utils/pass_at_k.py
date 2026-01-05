"""Pass@K evaluation utilities."""


def compute_pass_at_k(results: list[list[bool]], k: int) -> float:
    """
    Compute pass@k metric.
    
    Args:
        results: List of lists, where each inner list contains boolean results
                 for each sample of a problem (True if correct, False otherwise)
        k: Number of samples to consider for pass@k
    
    Returns:
        pass@k score (fraction of problems solved with k attempts)
    """
    num_problems = len(results)
    num_passed = 0
    
    for problem_results in results:
        # A problem passes if any of the first k samples is correct
        if any(problem_results[:k]):
            num_passed += 1
    
    return num_passed / num_problems
