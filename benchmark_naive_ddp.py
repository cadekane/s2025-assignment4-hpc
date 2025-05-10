import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cs336_basics.model import BasicsTransformerLM

# Define model sizes
MODEL_SIZES = {
    "small":   {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium":  {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":   {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":      {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":    {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def benchmark_ddp(rank, world_size, model_size="small", batch_size=8, seq_len=512, steps=5):
    setup(rank, world_size)

    config = MODEL_SIZES[model_size]
    device = torch.device(f"cuda:{rank}")

    model = BasicsTransformerLM(
        vocab_size=50257,
        context_length=seq_len,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        attn_pdrop=0.1,
        residual_pdrop=0.1
    ).to(device)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    vocab_size = 50257
    iter_times = []
    comm_times = []

    for step in range(steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()

        torch.cuda.synchronize()
        start_iter = time.time()

        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        torch.cuda.synchronize()
        start_comm = time.time()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

        torch.cuda.synchronize()
        comm_time = time.time() - start_comm

        optimizer.step()

        torch.cuda.synchronize()
        iter_time = time.time() - start_iter

        iter_times.append(iter_time)
        comm_times.append(comm_time)

        if rank == 0:
            print(f"[Step {step}] Iter Time: {iter_time:.4f}s, Comm Time: {comm_time:.4f}s")

    if rank == 0:
        print("--- Benchmark Summary ---")
        print(f"Avg Iter Time: {np.mean(iter_times):.4f}s")
        print(f"Avg Comm Time: {np.mean(comm_times):.4f}s")
        print(f"Comm/Iter Ratio: {np.mean(comm_times)/np.mean(iter_times):.2%}")

    cleanup()

def run_all():
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    mp.spawn(benchmark_ddp, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    run_all()
