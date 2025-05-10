import torch
import timeit
import numpy as np
import torch.nn as nn
import argparse
from cs336_basics.model import RMSNorm
from tests.adapters import TritonRMSNormAutogradFunction

# Define a new module to wrap Triton RMSNorm
class TritonRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))  # Optional: for consistency with other layers
        self.eps = eps
    
    def forward(self, x):
        # Call the Triton autograd function in forward pass
        return TritonRMSNormAutogradFunction.apply(x, self.weight, self.bias, self.eps)

# Dictionary of normalization layers
NORMALIZATION_LAYERS = {
    "layernorm": lambda d_model: nn.LayerNorm(d_model),
    "rmsnorm": lambda d_model: RMSNorm(d_model),
    "triton_rmsnorm": lambda d_model: TritonRMSNorm(d_model),  # Add Triton RMSNorm here
}

def benchmark_norm(norm_type="layernorm", d_model=768, batch_size=8, seq_len=512, steps=5, warmup=1, backward=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create normalization layer
    norm = NORMALIZATION_LAYERS[norm_type](d_model).to(device)
    times = []

    # Warm-up phase
    for _ in range(warmup):
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=backward)
        dy = torch.randn_like(x) if backward else None
        norm.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        out = norm(x)
        if backward:
            out.backward(dy)

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Benchmark phase
    for _ in range(steps):
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=backward)
        dy = torch.randn_like(x) if backward else None
        norm.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

        start = timeit.default_timer()
        out = norm(x)
        if backward:
            out.backward(dy)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        end = timeit.default_timer()

        times.append(end - start)

    avg_time = np.mean(times)
    std_dev = np.std(times)
    kind = "forward+backward" if backward else "forward"
    print(f"[{norm_type.upper()}] {kind} — Avg: {avg_time:.6f}s, Std Dev: {std_dev:.6f}s")
    return avg_time, std_dev

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_type", choices=NORMALIZATION_LAYERS.keys(), default="layernorm")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    benchmark_norm(
        norm_type=args.norm_type,
        d_model=args.d_model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        warmup=args.warmup,
        backward=args.backward
    )
