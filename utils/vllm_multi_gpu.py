"""Utilities for multi-GPU vLLM server management."""

import atexit
import os
import signal
import subprocess
import time
from typing import Optional

import requests


def _parse_csv(value: Optional[str]) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_vllm_gpu_ids(args) -> list[str]:
    """Resolve GPU IDs for multi-GPU vLLM servers.
    
    Priority:
    1. --vllm-gpu-ids argument
    2. CUDA_VISIBLE_DEVICES environment variable
    3. Auto-detect via torch.cuda.device_count()
    """
    if args.vllm_gpu_ids:
        return _parse_csv(args.vllm_gpu_ids)

    visible = _parse_csv(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible:
        return visible

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to auto-detect GPUs; please set --vllm-gpu-ids"
        ) from exc

    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError(
            "No CUDA devices detected; please set --vllm-gpu-ids or CUDA_VISIBLE_DEVICES"
        )
    return [str(i) for i in range(count)]


def _client_host_for_server(host: str) -> str:
    """Convert server bind address to client-usable address."""
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def build_vllm_server_urls(args, gpu_ids: list[str]) -> list[str]:
    """Build list of URLs for vLLM servers, one per GPU."""
    host = _client_host_for_server(args.vllm_server_host)
    return [
        f"http://{host}:{args.vllm_server_base_port + idx}"
        for idx in range(len(gpu_ids))
    ]


def launch_vllm_servers(args, gpu_ids: list[str]) -> list[subprocess.Popen]:
    """Launch one vLLM server per GPU.
    
    Each server is launched with CUDA_VISIBLE_DEVICES set to a single GPU,
    and listens on a unique port (base_port + index).
    """
    processes = []
    for idx, gpu_id in enumerate(gpu_ids):
        port = args.vllm_server_base_port + idx
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            "vllm",
            "serve",
            args.model,
            "--host",
            args.vllm_server_host,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        # Add max-model-len if specified (used by budget_forcing)
        if hasattr(args, 'max_model_len') and args.max_model_len:
            cmd.extend(["--max-model-len", str(args.max_model_len)])
        processes.append(subprocess.Popen(cmd, env=env))
    return processes


def wait_for_vllm_servers(urls: list[str], timeout_s: float) -> None:
    """Wait until all vLLM servers respond to /v1/models.
    
    Args:
        urls: List of server URLs to wait for
        timeout_s: Maximum time to wait in seconds
        
    Raises:
        RuntimeError: If servers don't respond within timeout
    """
    deadline = time.monotonic() + timeout_s
    remaining = set(urls)
    while remaining and time.monotonic() < deadline:
        for url in list(remaining):
            try:
                resp = requests.get(f"{url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    remaining.remove(url)
            except requests.RequestException:
                pass
        if remaining:
            time.sleep(1.0)
    if remaining:
        raise RuntimeError(
            f"Timed out waiting for vLLM servers: {', '.join(sorted(remaining))}"
        )


def shutdown_vllm_servers(processes: list[subprocess.Popen]) -> None:
    """Terminate vLLM server processes gracefully."""
    for proc in processes:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def register_vllm_shutdown(processes: list[subprocess.Popen]) -> None:
    """Register cleanup handlers to shutdown vLLM servers on exit.
    
    Registers both atexit handler and signal handlers for SIGINT/SIGTERM.
    """
    atexit.register(shutdown_vllm_servers, processes)

    def _handle_signal(signum, _frame):
        shutdown_vllm_servers(processes)
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
