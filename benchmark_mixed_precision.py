import argparse # Got to finish xl on a 32GB GPU with this script
import torch
import timeit
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

def benchmark(model_size="small", batch_size=8, seq_len=512, steps=5, warmup=1, backward=False, mixed_precision=False):
    config = MODEL_SIZES[model_size]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    autocast = torch.cuda.amp.autocast if mixed_precision and device == "cuda" else nullcontext
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == "cuda" else None

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

    dummy_input = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    dummy_target = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters())
    times = []

    # Warm-up steps
    for _ in range(warmup):
        with autocast():
            output = model(dummy_input)
            if backward:
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), dummy_target.view(-1))

        if backward:
            scaler.scale(loss).backward() if scaler else loss.backward()
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        del output
        if backward:
            del loss

    # Timed steps
    for _ in range(steps):
        start = timeit.default_timer()
        with autocast():
            output = model(dummy_input)
            if backward:
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), dummy_target.view(-1))

        if backward:
            scaler.scale(loss).backward() if scaler else loss.backward()
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        end = timeit.default_timer()
        times.append(end - start)

        del output
        if backward:
            del loss

    avg_time = np.mean(times)
    std_dev = np.std(times)
    kind = "forward+backward" if backward else "forward"
    print(f"[{model_size}] {kind} pass — Avg: {avg_time:.4f}s, Std Dev: {std_dev:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, choices=MODEL_SIZES.keys(), default="small")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision with autocast")
    args = parser.parse_args()

    benchmark(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        warmup=args.warmup,
        backward=args.backward,
        mixed_precision=args.mixed_precision
    )
