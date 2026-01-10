"""Sampling client using vLLM for fast inference."""

import requests
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


class ServerSamplingClient:
    """Sampling client that connects to a running vLLM server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Connect to a vLLM server.

        Start server with: vllm serve <model_name> --port 8000

        Args:
            base_url: URL of the vLLM server
        """
        self.base_url = base_url.rstrip("/")

    def sample(
        self,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> SamplingResult:
        """Sample from the model via the server."""
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": prompt.token_ids,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "n": num_samples,
                "stop": sampling_params.stop,
            },
        )
        response.raise_for_status()
        data = response.json()

        sequences = []
        for choice in data["choices"]:
            logprobs = choice.get("logprobs") or {}
            seq = Sequence(
                tokens=logprobs.get("tokens", []),
                text=choice["text"],
            )
            sequences.append(seq)

        return SamplingResult(sequences=sequences)

    def sample_batch(
        self,
        prompts: list[ModelInput],
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> list[SamplingResult]:
        """Sample from the model for multiple prompts."""
        # vLLM server supports batch via list of prompts
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": [p.token_ids for p in prompts],
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "n": num_samples,
                "stop": sampling_params.stop,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Group choices by prompt index
        results = [[] for _ in prompts]
        for choice in data["choices"]:
            idx = choice["index"] // num_samples
            seq = Sequence(
                tokens=choice.get("logprobs", {}).get("tokens", []),
                text=choice["text"],
            )
            results[idx].append(seq)

        return [SamplingResult(sequences=seqs) for seqs in results]

