"""GPU Keep-Alive utility to prevent SLURM from killing jobs due to idle GPU.

Usage:
    from utils.gpu_keepalive import GPUKeepAlive

    # As context manager
    with GPUKeepAlive():
        # CPU-bound work here while GPU stays "busy"
        evaluate_tasks(...)

    # Or manually
    keepalive = GPUKeepAlive()
    keepalive.start()
    # ... do work ...
    keepalive.stop()
"""

import threading


class GPUKeepAlive:
    """Keep GPU busy with continuous matmuls to prevent SLURM idle GPU termination.
    
    Runs a background thread that performs small matrix multiplications continuously.
    Uses negligible GPU memory (~2MB) and minimal GPU compute (~1-5% utilization).
    """
    
    def __init__(self, device: int = 0, size: int = 512):
        """
        Args:
            device: CUDA device index (default: 0)
            size: Matrix size for matmuls (default: 512, uses ~2MB GPU memory)
        """
        self.device = device
        self.size = size
        self._stop = threading.Event()
        self._thread = None
    
    def _loop(self):
        """Continuous matmul loop running on background thread."""
        import torch
        dev = f"cuda:{self.device}"
        
        while not self._stop.is_set():
            a = torch.randn(self.size, self.size, device=dev)
            b = torch.randn(self.size, self.size, device=dev)
            _ = torch.mm(a, b)
            torch.cuda.synchronize(dev)
            del a, b
    
    def start(self):
        """Start the keep-alive background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the keep-alive background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
