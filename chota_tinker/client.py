"""Sampling client using vLLM for fast inference."""

from vllm import LLM
from vllm import SamplingParams as VLLMSamplingParams

from .types import ModelInput, SamplingParams, SamplingResult, Sequence


class SamplingClient:
    """Minimal sampling client wrapping vLLM."""

    def __init__(self, model_name: str, **vllm_kwargs):
        """
        Initialize the sampling client.

        Args:
            model_name: HuggingFace model name or path
            **vllm_kwargs: Additional kwargs passed to vLLM LLM()
                e.g. tensor_parallel_size, gpu_memory_utilization, etc.
        """
        self.model_name = model_name
        self.llm = LLM(model=model_name, **vllm_kwargs)

    def sample(
        self,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> SamplingResult:
        """
        Sample from the model.

        Args:
            prompt: Input tokens as ModelInput
            sampling_params: Sampling parameters
            num_samples: Number of samples to generate

        Returns:
            SamplingResult with generated sequences
        """
        # Convert to vLLM sampling params
        vllm_params = VLLMSamplingParams(
            max_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k if sampling_params.top_k > 0 else -1,
            stop=sampling_params.stop,
            n=num_samples,
        )

        # Generate using vLLM
        outputs = self.llm.generate(
            prompt_token_ids=[prompt.token_ids],
            sampling_params=vllm_params,
        )

        # Convert outputs to our format
        sequences = []
        for output in outputs[0].outputs:
            seq = Sequence(
                tokens=output.token_ids,
                text=output.text,
            )
            sequences.append(seq)

        return SamplingResult(sequences=sequences)

    def sample_batch(
        self,
        prompts: list[ModelInput],
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> list[SamplingResult]:
        """
        Sample from the model for multiple prompts (batched).

        Args:
            prompts: List of input tokens as ModelInput
            sampling_params: Sampling parameters
            num_samples: Number of samples per prompt

        Returns:
            List of SamplingResult, one per prompt
        """
        vllm_params = VLLMSamplingParams(
            max_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k if sampling_params.top_k > 0 else -1,
            stop=sampling_params.stop,
            n=num_samples,
        )

        # Generate using vLLM (batched)
        outputs = self.llm.generate(
            prompt_token_ids=[p.token_ids for p in prompts],
            sampling_params=vllm_params,
        )

        # Convert outputs to our format
        results = []
        for output in outputs:
            sequences = [
                Sequence(tokens=o.token_ids, text=o.text)
                for o in output.outputs
            ]
            results.append(SamplingResult(sequences=sequences))

        return results

