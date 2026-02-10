"""Sampling client using vLLM for fast inference."""

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
import torch
from vllm import LLM
from vllm import SamplingParams as VLLMSamplingParams
from vllm.inputs import TokensPrompt

from .types import ModelInput, SamplingParams, SamplingResult, Sequence

DEFAULT_RETRY_STATUSES = {429, 500, 502, 503, 504}


class SamplingClient:
    """Minimal sampling client wrapping vLLM with sleep/wake support."""

    def __init__(self, model_name: str, **vllm_kwargs):
        """
        Initialize the sampling client.

        Args:
            model_name: HuggingFace model name or path
            **vllm_kwargs: Additional kwargs passed to vLLM LLM()
                e.g. tensor_parallel_size, gpu_memory_utilization, etc.
        """
        self.model_name = model_name
        # Enable prefix caching by default for multi-turn efficiency
        vllm_kwargs.setdefault("enable_prefix_caching", True)
        # Enable sleep mode for memory management during training
        vllm_kwargs.setdefault("enable_sleep_mode", True)
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
            stop_token_ids=sampling_params.stop_token_ids,
            n=num_samples,
        )

        # Generate using vLLM
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prompt.token_ids)],
            sampling_params=vllm_params,
        )

        # Convert outputs to our format
        sequences = []
        for output in outputs[0].outputs:
            seq = Sequence(
                tokens=output.token_ids,
                text=output.text,
                finish_reason=getattr(output, "finish_reason", None),
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
            stop_token_ids=sampling_params.stop_token_ids,
            n=num_samples,
        )

        # Generate using vLLM (batched)
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=p.token_ids) for p in prompts],
            sampling_params=vllm_params,
        )

        # Convert outputs to our format
        results = []
        for output in outputs:
            sequences = [
                Sequence(
                    tokens=o.token_ids,
                    text=o.text,
                    finish_reason=getattr(o, "finish_reason", None),
                )
                for o in output.outputs
            ]
            results.append(SamplingResult(sequences=sequences))

        return results

    def sleep(self, level: int = 1):
        """
        Put vLLM to sleep to free GPU memory for training.
        
        Args:
            level: Sleep level (1 = offload KV cache, higher = more aggressive)
        """
        self.llm.sleep(level=level)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)  # Allow memory to settle

    def wake_up(self):
        """Wake up vLLM from sleep for inference."""
        self.llm.wake_up()

    def load_weights(self, state_dict: dict, checkpoint_dir: str = "/tmp/vllm_weight_sync"):
        """
        Load new weights into the vLLM model.
        
        For vLLM V1 with sleep mode, this uses apply_model to access the
        internal model's load_weights method.
        
        Args:
            state_dict: Model state dict from training model
            checkpoint_dir: Not used (kept for API compatibility)
        """
        # Convert state_dict items to a list for pickling (needed for multiprocess)
        weights_list = list(state_dict.items())
        
        def _load_weights(model):
            """Load weights into the model (runs in worker process)."""
            model.load_weights(weights_list)
        
        # Use apply_model to access model in V1 engine (works across processes)
        try:
            self.llm.llm_engine.apply_model(_load_weights)
        except AttributeError:
            # Fallback for older vLLM versions with model_executor
            try:
                model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                model.load_weights(weights_list)
            except AttributeError:
                try:
                    model = self.llm.llm_engine.model_executor.driver_worker.model_runner.get_model()
                    model.load_weights(weights_list)
                except AttributeError:
                    raise RuntimeError(
                        "Could not access vLLM internal model. "
                        "This may be due to vLLM version incompatibility."
                    )


class ServerSamplingClient:
    """Sampling client that connects to a running vLLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_s: float = 120.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        retry_statuses: Optional[set[int]] = None,
    ):
        """
        Connect to a vLLM server.

        Start server with: vllm serve <model_name> --port 8000

        Args:
            base_url: URL of the vLLM server
            timeout_s: Request timeout in seconds
            max_retries: Number of retry attempts for transient failures
            retry_backoff_s: Base backoff in seconds (exponential per attempt)
            retry_statuses: HTTP statuses to retry
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.retry_statuses = retry_statuses or DEFAULT_RETRY_STATUSES

    def _post_json(self, path: str, payload: dict) -> dict:
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    timeout=self.timeout_s,
                )
                if response.status_code >= 400:
                    if (
                        response.status_code in self.retry_statuses
                        and attempt < self.max_retries
                    ):
                        time.sleep(self.retry_backoff_s * (2 ** attempt))
                        continue
                    response.raise_for_status()

                try:
                    return response.json()
                except ValueError as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_s * (2 ** attempt))
                        continue
                    break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                break

        raise RuntimeError(
            f"vLLM request failed after {self.max_retries + 1} attempts: {self.base_url}{path}"
        ) from last_exc

    def sample(
        self,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> SamplingResult:
        """Sample from the model via the server."""
        request_data = {
            "prompt": prompt.token_ids,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "n": num_samples,
            "stop": sampling_params.stop,
        }
        if sampling_params.stop_token_ids:
            request_data["stop_token_ids"] = sampling_params.stop_token_ids
        data = self._post_json("/v1/completions", request_data)

        sequences = []
        for choice in data["choices"]:
            logprobs = choice.get("logprobs") or {}
            seq = Sequence(
                tokens=logprobs.get("tokens", []),
                text=choice["text"],
                finish_reason=choice.get("finish_reason"),
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
        request_data = {
            "prompt": [p.token_ids for p in prompts],
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "n": num_samples,
            "stop": sampling_params.stop,
        }
        if sampling_params.stop_token_ids:
            request_data["stop_token_ids"] = sampling_params.stop_token_ids
        data = self._post_json("/v1/completions", request_data)

        # Group choices by prompt index
        results = [[] for _ in prompts]
        for choice in data["choices"]:
            idx = choice["index"] // num_samples
            logprobs = choice.get("logprobs") or {}
            if not isinstance(logprobs, dict):
                logprobs = {}
            seq = Sequence(
                tokens=logprobs.get("tokens", []),
                text=choice["text"],
                finish_reason=choice.get("finish_reason"),
            )
            results[idx].append(seq)

        return [SamplingResult(sequences=seqs) for seqs in results]


class MultiServerSamplingClient:
    """Sampling client that shards requests across multiple vLLM servers."""

    def __init__(self, base_urls: list[str], max_workers: Optional[int] = None):
        if not base_urls:
            raise ValueError("base_urls must be a non-empty list")
        self.clients = [ServerSamplingClient(url) for url in base_urls]
        self.max_workers = max_workers or len(self.clients)
        self._rr_index = 0

    def _next_client(self) -> ServerSamplingClient:
        client = self.clients[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(self.clients)
        return client

    def sample(
        self,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> SamplingResult:
        """Sample from the next server (round-robin)."""
        client = self._next_client()
        return client.sample(prompt, sampling_params, num_samples=num_samples)

    def sample_batch(
        self,
        prompts: list[ModelInput],
        sampling_params: SamplingParams,
        num_samples: int = 1,
        show_progress: bool = False,
    ) -> list[SamplingResult]:
        """Shard prompts across servers and recombine results in order.
        
        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            num_samples: Number of samples per prompt
            show_progress: If True, display a tqdm progress bar (updates per GPU batch)
        """
        if not prompts:
            return []

        num_clients = len(self.clients)
        buckets: list[list[ModelInput]] = [[] for _ in range(num_clients)]
        index_buckets: list[list[int]] = [[] for _ in range(num_clients)]

        for idx, prompt in enumerate(prompts):
            bucket_idx = idx % num_clients
            buckets[bucket_idx].append(prompt)
            index_buckets[bucket_idx].append(idx)

        results: list[Optional[SamplingResult]] = [None] * len(prompts)

        def _run_batch(client: ServerSamplingClient, batch, indices):
            if not batch:
                return indices, []
            return indices, client.sample_batch(
                batch, sampling_params, num_samples=num_samples
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for client, batch, indices in zip(self.clients, buckets, index_buckets):
                if batch:
                    futures.append(
                        executor.submit(_run_batch, client, batch, indices)
                    )

            if show_progress:
                from concurrent.futures import as_completed
                from tqdm import tqdm
                pbar = tqdm(total=len(prompts), desc="Sampling", unit="prompts")
                for future in as_completed(futures):
                    indices, batch_results = future.result()
                    for idx, result in zip(indices, batch_results):
                        results[idx] = result
                    pbar.update(len(indices))
                pbar.close()
            else:
                for future in futures:
                    indices, batch_results = future.result()
                    for idx, result in zip(indices, batch_results):
                        results[idx] = result

        if any(r is None for r in results):
            raise RuntimeError("MultiServerSamplingClient received incomplete results")
        return [r for r in results if r is not None]


