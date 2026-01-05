# load a dataset and evaluate pass@K
# we use tinker for sampling
# for now use this dataset: https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-RL
# dataset = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")

import asyncio

import tinker
from tinker import types
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.pass_at_k import compute_pass_at_k
from utils.answer import check_answer


async def evaluate_problem(
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    problem: str,
    target: str,
    num_samples: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> list[bool]:
    """
    Evaluate a single problem with multiple samples.
    
    Returns:
        List of booleans indicating if each sample was correct
    """
    # Create prompt - use chat template if available, otherwise simple format
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": problem}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        # Fallback for base models without chat template
        prompt_text = f"Solve this problem step-by-step: {problem}\n\n Put the final answer within \\boxed{{}}\n"
    
    input_ids = tokenizer(prompt_text)["input_ids"]
    model_input = types.ModelInput.from_ints(input_ids)
    
    # Sampling parameters
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    
    # Sample from the model
    response = await sampling_client.sample_async(
        prompt=model_input,
        sampling_params=params,
        num_samples=num_samples,
    )
    
    # Check each sample
    results = []
    responses = []
    for seq in response.sequences:
        decoded = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        is_correct = check_answer(decoded, target)
        results.append(is_correct)
        responses.append(decoded)
    
    return results, responses, prompt_text


async def run_evaluation(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    num_problems: int = 100,
    num_samples: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.7,
):
    """
    Run pass@K evaluation on a math dataset.
    
    Args:
        model_name: The model to evaluate
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration/subset
        num_problems: Number of problems to evaluate
        num_samples: Number of samples per problem (K for pass@K)
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
    """
    print(f"Loading model: {model_name}")
    
    # Initialize tinker
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    dataset = load_dataset(dataset_name, dataset_config, split="train")
    
    # Limit to num_problems
    dataset = dataset.select(range(min(num_problems, len(dataset))))
    
    print(f"Evaluating {len(dataset)} problems with {num_samples} samples each...")
    
    # Evaluate all problems
    all_results = []
    examples_to_show = []  # Store a few examples for display
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        problem = example["question"]
        target = example["answer"]
        
        results, responses, prompt_text = await evaluate_problem(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            problem=problem,
            target=target,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        all_results.append(results)
        
        # Store first few examples for display
        if len(examples_to_show) < 3:
            examples_to_show.append({
                "problem": problem,
                "prompt": prompt_text,
                "target": target,
                "response": responses[0],  # First sample
                "correct": results[0],
            })
    
    # Compute final metrics
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Problems evaluated: {len(dataset)}")
    print(f"Samples per problem: {num_samples}")
    print()
    
    for k in [1, 2, 4, 8]:
        if k <= num_samples:
            score = compute_pass_at_k(all_results, k=k)
            print(f"pass@{k}: {score:.2%}")
    
    # Display example prompts and responses
    print("\n" + "=" * 50)
    print("EXAMPLE OUTPUTS")
    print("=" * 50)
    
    for i, ex in enumerate(examples_to_show):
        print(f"\n--- Example {i+1} ---\n")
        print(f"{ex['prompt']}")
        print(f"{ex['response']}")
        print(f"\n\n[TARGET]\n{ex['target']}")
        print(f"\n[CORRECT] {ex['correct']}")
        print("-" * 50)
    
    return all_results


if __name__ == "__main__":
    # Configuration

    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name = "meta-llama/Llama-3.2-3B"
    model_name = "openai/gpt-oss-20b"

    dataset_name = "PrimeIntellect/INTELLECT-3-RL"
    dataset_config = "math"

    num_problems = 5  # Start with fewer for testing
    num_samples = 8
    max_tokens = 2048
    temperature = 0.6

    asyncio.run(run_evaluation(
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        num_problems=num_problems,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
    ))