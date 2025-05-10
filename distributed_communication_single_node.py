import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [0.5, 1, 10, 50, 100, 500, 1024]
NUM_WARMUP = 5
NUM_ITERS = 10

def setup(rank, world_size, backend, device_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type == "cuda":
        torch.cuda.set_device(rank)

def benchmark_all_reduce(rank, world_size, backend, device_type):
    setup(rank, world_size, backend, device_type)
    device = f"{device_type}:{rank}" if device_type == "cuda" else "cpu"

    results = []

    for size_mb in SIZES_MB:
        num_elements = int((size_mb * 1024 * 1024) / 4)
        tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

        for _ in range(NUM_WARMUP):
            dist.all_reduce(tensor, async_op=False)
            if device_type == "cuda":
                torch.cuda.synchronize()

        start = time.time()
        for _ in range(NUM_ITERS):
            dist.all_reduce(tensor, async_op=False)
            if device_type == "cuda":
                torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / NUM_ITERS
        print(f"Rank {rank} | Size: {size_mb}MB | Time: {avg_time:.6f}s")
        results.append((size_mb, avg_time))

    dist.destroy_process_group()

def run(backend, device_type, world_size):
    mp.spawn(
        benchmark_all_reduce,
        args=(world_size, backend, device_type),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    run("nccl", "cuda", 2)  # Change backend/device/world_size as needed
