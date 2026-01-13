#!/usr/bin/env python3
"""
GPU Idle Monitor Script
Monitors GPU utilization and memory usage, with options to:
- Check if GPUs are idle
- Wait until GPUs become idle
- Continuously monitor GPU status
- Occupy GPUs with high utilization and memory usage

Usage:
python gpu_idle.py occupy --memory 20

"""

import subprocess
import time
import argparse
import sys


def get_gpu_stats():
    """Get GPU utilization and memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = [p.strip() for p in line.split(",")]
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization": float(parts[2]) if parts[2] != "[N/A]" else 0,
                    "memory_used": float(parts[3]),
                    "memory_total": float(parts[4]),
                    "temperature": float(parts[5]) if parts[5] != "[N/A]" else 0,
                })
        return gpus
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        sys.exit(1)


def is_gpu_idle(gpu, util_threshold=5, mem_threshold=500):
    """Check if a GPU is considered idle based on thresholds."""
    return gpu["utilization"] <= util_threshold and gpu["memory_used"] <= mem_threshold


def print_gpu_status(gpus, util_threshold=5, mem_threshold=500):
    """Print formatted GPU status."""
    print("\n" + "=" * 80)
    print(f"{'GPU':<5} {'Name':<25} {'Util %':<10} {'Memory (MB)':<20} {'Temp °C':<10} {'Status':<10}")
    print("=" * 80)
    
    for gpu in gpus:
        mem_str = f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}"
        idle = is_gpu_idle(gpu, util_threshold, mem_threshold)
        status = "🟢 IDLE" if idle else "🔴 BUSY"
        
        print(f"{gpu['index']:<5} {gpu['name']:<25} {gpu['utilization']:<10.1f} {mem_str:<20} {gpu['temperature']:<10.0f} {status:<10}")
    
    print("=" * 80)


def wait_for_idle(gpu_indices=None, util_threshold=5, mem_threshold=500, check_interval=10, timeout=None):
    """Wait until specified GPUs become idle."""
    start_time = time.time()
    
    print(f"Waiting for GPUs to become idle (util <= {util_threshold}%, mem <= {mem_threshold}MB)...")
    
    while True:
        gpus = get_gpu_stats()
        
        if gpu_indices:
            target_gpus = [g for g in gpus if g["index"] in gpu_indices]
        else:
            target_gpus = gpus
        
        all_idle = all(is_gpu_idle(g, util_threshold, mem_threshold) for g in target_gpus)
        
        if all_idle:
            print("\n✅ All target GPUs are now idle!")
            return True
        
        if timeout and (time.time() - start_time) > timeout:
            print(f"\n⏰ Timeout reached after {timeout} seconds.")
            return False
        
        elapsed = time.time() - start_time
        busy_gpus = [g["index"] for g in target_gpus if not is_gpu_idle(g, util_threshold, mem_threshold)]
        print(f"\r[{elapsed:.0f}s] Waiting... Busy GPUs: {busy_gpus}", end="", flush=True)
        
        time.sleep(check_interval)


def monitor(interval=2, util_threshold=5, mem_threshold=500):
    """Continuously monitor GPU status."""
    print("Monitoring GPUs (Ctrl+C to stop)...")
    
    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")
            print(f"GPU Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            gpus = get_gpu_stats()
            print_gpu_status(gpus, util_threshold, mem_threshold)
            
            idle_count = sum(1 for g in gpus if is_gpu_idle(g, util_threshold, mem_threshold))
            print(f"\nIdle: {idle_count}/{len(gpus)} GPUs")
            print(f"\nRefreshing every {interval}s... (Ctrl+C to stop)")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def occupy_gpus(gpu_indices=None, memory_gb=10, duration=None):
    """Occupy GPUs with high memory and utilization."""
    try:
        import torch
    except ImportError:
        print("PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    
    if gpu_indices is None:
        gpu_indices = list(range(num_gpus))
    
    print(f"🔥 Occupying GPUs: {gpu_indices}")
    print(f"📦 Target memory per GPU: {memory_gb} GB")
    print(f"⏱️  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print("\nPress Ctrl+C to stop\n")
    
    # Allocate memory on each GPU
    tensors = []
    for idx in gpu_indices:
        if idx >= num_gpus:
            print(f"⚠️  GPU {idx} not available (only {num_gpus} GPUs found)")
            continue
        
        device = torch.device(f"cuda:{idx}")
        # Allocate memory (each float32 is 4 bytes)
        num_elements = int(memory_gb * 1024 * 1024 * 1024 / 4)
        try:
            tensor = torch.randn(num_elements, device=device)
            tensors.append((idx, tensor, device))
            allocated_gb = tensor.element_size() * tensor.numel() / (1024**3)
            print(f"✅ GPU {idx}: Allocated {allocated_gb:.2f} GB")
        except RuntimeError as e:
            print(f"❌ GPU {idx}: Failed to allocate - {e}")
    
    if not tensors:
        print("No GPUs were successfully occupied!")
        sys.exit(1)
    
    print(f"\n🔄 Running compute loops to keep utilization high...")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            for idx, tensor, device in tensors:
                # Perform operations to keep GPU busy
                size = min(8192, int(len(tensor) ** 0.5))
                a = tensor[:size*size].reshape(size, size)
                _ = torch.matmul(a, a)
                _ = torch.sin(tensor)
                _ = tensor * tensor
            
            iteration += 1
            elapsed = time.time() - start_time
            
            if iteration % 10 == 0:
                print(f"\r⚡ Running... {elapsed:.0f}s elapsed, {iteration} iterations", end="", flush=True)
            
            if duration and elapsed >= duration:
                print(f"\n\n✅ Duration of {duration}s reached. Stopping.")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user.")
    
    # Cleanup
    print("🧹 Releasing GPU memory...")
    del tensors
    torch.cuda.empty_cache()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="GPU Idle Monitor")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show current GPU status")
    status_parser.add_argument("--util-threshold", type=float, default=5, help="Utilization threshold for idle (default: 5%%)")
    status_parser.add_argument("--mem-threshold", type=float, default=500, help="Memory threshold for idle in MB (default: 500)")
    
    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait until GPUs become idle")
    wait_parser.add_argument("--gpus", type=int, nargs="+", help="GPU indices to wait for (default: all)")
    wait_parser.add_argument("--util-threshold", type=float, default=5, help="Utilization threshold for idle (default: 5%%)")
    wait_parser.add_argument("--mem-threshold", type=float, default=500, help="Memory threshold for idle in MB (default: 500)")
    wait_parser.add_argument("--interval", type=float, default=10, help="Check interval in seconds (default: 10)")
    wait_parser.add_argument("--timeout", type=float, help="Timeout in seconds (default: no timeout)")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuously monitor GPUs")
    monitor_parser.add_argument("--interval", type=float, default=2, help="Refresh interval in seconds (default: 2)")
    monitor_parser.add_argument("--util-threshold", type=float, default=5, help="Utilization threshold for idle (default: 5%%)")
    monitor_parser.add_argument("--mem-threshold", type=float, default=500, help="Memory threshold for idle in MB (default: 500)")
    
    # Check command (returns exit code)
    check_parser = subparsers.add_parser("check", help="Check if GPUs are idle (exit 0 if idle, 1 if busy)")
    check_parser.add_argument("--gpus", type=int, nargs="+", help="GPU indices to check (default: all)")
    check_parser.add_argument("--util-threshold", type=float, default=5, help="Utilization threshold for idle (default: 5%%)")
    check_parser.add_argument("--mem-threshold", type=float, default=500, help="Memory threshold for idle in MB (default: 500)")
    
    # Occupy command (burn GPUs)
    occupy_parser = subparsers.add_parser("occupy", help="Occupy GPUs with high utilization and memory")
    occupy_parser.add_argument("--gpus", type=int, nargs="+", help="GPU indices to occupy (default: all)")
    occupy_parser.add_argument("--memory", type=float, default=10, help="Memory to allocate per GPU in GB (default: 10)")
    occupy_parser.add_argument("--duration", type=float, help="Duration in seconds (default: infinite)")
    
    args = parser.parse_args()
    
    if args.command is None or args.command == "status":
        gpus = get_gpu_stats()
        ut = getattr(args, "util_threshold", 5)
        mt = getattr(args, "mem_threshold", 500)
        print_gpu_status(gpus, ut, mt)
        
    elif args.command == "wait":
        success = wait_for_idle(
            gpu_indices=args.gpus,
            util_threshold=args.util_threshold,
            mem_threshold=args.mem_threshold,
            check_interval=args.interval,
            timeout=args.timeout,
        )
        sys.exit(0 if success else 1)
        
    elif args.command == "monitor":
        monitor(
            interval=args.interval,
            util_threshold=args.util_threshold,
            mem_threshold=args.mem_threshold,
        )
        
    elif args.command == "check":
        gpus = get_gpu_stats()
        if args.gpus:
            target_gpus = [g for g in gpus if g["index"] in args.gpus]
        else:
            target_gpus = gpus
        
        all_idle = all(is_gpu_idle(g, args.util_threshold, args.mem_threshold) for g in target_gpus)
        
        if all_idle:
            print("✅ All target GPUs are idle")
            sys.exit(0)
        else:
            busy = [g["index"] for g in target_gpus if not is_gpu_idle(g, args.util_threshold, args.mem_threshold)]
            print(f"🔴 Busy GPUs: {busy}")
            sys.exit(1)
    
    elif args.command == "occupy":
        occupy_gpus(
            gpu_indices=args.gpus,
            memory_gb=args.memory,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
