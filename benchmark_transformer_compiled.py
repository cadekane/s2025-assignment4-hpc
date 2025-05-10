import torch
import timeit
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim

from cs336_basics.model import BasicsTransformerLM

def benchmark_model(d_model=768, d_ff=3072, num_layers=12, num_heads=12,
                    batch_size=8, seq_len=512, vocab_size=50257, steps=5,
                    warmup=1, compile_model=False, backward=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BasicsTransformerLM(
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        context_length=seq_len
    ).to(device)

    if compile_model:
        model = torch.compile(model)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    def generate_batch():
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        return x, y

    # Warm-up runs
    for _ in range(warmup):
        x, y = generate_batch()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        if backward:
            loss.backward()
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Timed runs
    times = []
    for _ in range(steps):
        x, y = generate_batch()
        optimizer.zero_grad(set_to_none=True)

        start = timeit.default_timer()
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        if backward:
            loss.backward()
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        end = timeit.default_timer()
        times.append(end - start)

    avg_time = np.mean(times)
    std_dev = np.std(times)
    kind = "forward+backward+optimizer" if backward else "forward"
    compile_status = "compiled" if compile_model else "vanilla"
    print(f"[{compile_status.upper()}] d_model={d_model} — {kind} — Avg: {avg_time:.6f}s, Std Dev: {std_dev:.6f}s")
    return avg_time, std_dev

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    benchmark_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        steps=args.steps,
        warmup=args.warmup,
        compile_model=args.compile,
        backward=args.backward
    )
