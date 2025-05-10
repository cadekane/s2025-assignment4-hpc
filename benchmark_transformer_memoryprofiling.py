import torch
import timeit
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from torch.profiler import profile, record_function, ProfilerActivity
import os
import traceback

# Define model sizes
MODEL_SIZES = {
    "small":   {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium":  {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":   {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":      {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":    {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def run_step(model, inputs, optimizer, enable_backward, device):
    with record_function('forward_pass'):
        output = model(inputs)
    
    if enable_backward:
        with record_function('backward_pass'):
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), inputs.view(-1))
            loss.backward()

        with record_function('optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return output

def benchmark(model_size="small", batch_size=8, seq_len=512, steps=5, warmup=1, backward=False, profile_mode=False):
    config = MODEL_SIZES[model_size]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Verify CUDA is available when profiling
    if profile_mode and device != "cuda":
        print("Warning: Memory profiling requires CUDA. Switching to regular timing.")
        profile_mode = False

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

    # Simple memory tracking functions
    def get_mem_info():
        if device == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated() / (1024 ** 2),  # MB
                "cached": torch.cuda.memory_reserved() / (1024 ** 2),      # MB
                "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            }
        return {"allocated": 0, "cached": 0, "max_allocated": 0}

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Get initial memory state
    initial_mem = get_mem_info()
    print(f"Initial memory state: {initial_mem}")

    # Warm-up steps
    for _ in range(warmup):
        output = model(dummy_input)
        if backward:
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), dummy_target.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()

    # Memory after warm-up
    warmup_mem = get_mem_info()
    print(f"Memory after warm-up: {warmup_mem}")

    # PROFILING BLOCK!
    if profile_mode:
        try:
            # Force some allocations
            torch.cuda.empty_cache()
            
            # Enable memory history recording
            print("Enabling memory history recording...")
            try:
                # Try the newer API first
                torch.cuda.memory._record_memory_history(enabled=True, max_entries=100000)
                print("Using newer memory history recording API")
            except TypeError:
                # Fall back to the legacy API
                try:
                    torch.cuda.memory._record_memory_history(max_entries=100000)
                    print("Using legacy memory history recording API")
                except Exception as e:
                    print(f"Warning: Could not enable memory history recording: {e}")
            
            # Force allocations to make sure memory events are recorded
            print("Creating dummy tensors to force memory events...")
            dummy_tensors = [torch.randn(1000, 1000, device=device) for _ in range(5)]
            torch.cuda.synchronize()
            
            memory_stats = []
            mem_snapshots = []
            
            # Enable profiler with memory tracking
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
            ) as prof:
                for step in range(steps):
                    print(f"Running profiling step {step+1}/{steps}")
                    
                    # Capture memory before step
                    mem_before = get_mem_info()
                    
                    # Run the model step
                    run_step(model, dummy_input, optimizer, backward, device)
                    
                    # Force CUDA synchronization
                    torch.cuda.synchronize()
                    
                    # Capture memory after step
                    mem_after = get_mem_info()
                    mem_snapshot = {
                        "step": step + 1,
                        "before": mem_before,
                        "after": mem_after,
                        "diff": {k: mem_after[k] - mem_before[k] for k in mem_before}
                    }
                    mem_snapshots.append(mem_snapshot)
                    print(f"Step {step+1} memory change: {mem_snapshot['diff']}")
                    
                    prof.step()
            
            print("Profiling completed. Generating reports...")
            
            # Export Chrome trace
            try:
                prof.export_chrome_trace("cuda_trace.json")
                print("Chrome trace exported to cuda_trace.json")
            except Exception as e:
                print(f"Chrome trace export error: {e}")
            
            # Export memory timeline
            try:
                print("Trying to export memory timeline...")
                # Get the function signature to check if it takes a device parameter
                import inspect
                export_timeline_params = inspect.signature(prof.export_memory_timeline).parameters
                
                # Check if it accepts 'device' parameter
                if 'device' in export_timeline_params:
                    prof.export_memory_timeline("memory_timeline.html", device)
                else:
                    prof.export_memory_timeline("memory_timeline.html")
                
                if os.path.exists("memory_timeline.html"):
                    print(f"Memory timeline exported successfully to memory_timeline.html")
                    filesize = os.path.getsize("memory_timeline.html") / 1024  # KB
                    print(f"File size: {filesize:.2f} KB")
                else:
                    print("Warning: memory_timeline.html was not created")
            except Exception as e:
                print(f"Memory timeline export error: {e}")
                traceback.print_exc()
            
            # Export memory snapshot
            try:
                print("Exporting memory snapshot...")
                torch.cuda.memory._dump_snapshot("xl_memory_snapshot.pickle")
                
                if os.path.exists("memory_snapshot.pickle"):
                    print(f"Memory snapshot exported successfully to memory_snapshot.pickle")
                    filesize = os.path.getsize("memory_snapshot.pickle") / 1024  # KB
                    print(f"File size: {filesize:.2f} KB")
                    print("You can visualize this file at https://pytorch.org/memory_viz")
                else:
                    print("Warning: memory_snapshot.pickle was not created")
            except Exception as e:
                print(f"Memory snapshot export error: {e}")
                traceback.print_exc()
            
            # Print memory stats table
            print("\nMemory Statistics by Step:")
            headers = ["Step", "Before (MB)", "After (MB)", "Change (MB)"]
            print(f"{headers[0]:<6} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15}")
            print("-" * 55)
            for snapshot in mem_snapshots:
                print(f"{snapshot['step']:<6} {snapshot['before']['allocated']:<15.2f} "
                      f"{snapshot['after']['allocated']:<15.2f} {snapshot['diff']['allocated']:<15.2f}")
            
            # Print peak memory usage
            final_mem = get_mem_info()
            print(f"\nPeak memory usage: {final_mem['max_allocated']:.2f} MB")
            
            # Print key averages from profiler
            print("\nTop operations by CUDA time:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
        except Exception as e:
            print(f"Profiling error: {e}")
            traceback.print_exc()
        finally:
            # Clean up memory recording
            try:
                # Try the newer API first to disable recording
                torch.cuda.memory._record_memory_history(enabled=False)
                print("Disabled memory history recording with newer API")
            except TypeError:
                # Fall back to the legacy API
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                    print("Disabled memory history recording with legacy API")
                except Exception as e:
                    print(f"Warning: Could not disable memory history recording: {e}")

    # Timed steps
    else:
        times = []
        for _ in range(steps):
            start = timeit.default_timer()
            output = model(dummy_input)
            if backward:
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), dummy_target.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)

        avg_time = np.mean(times)
        std_dev = np.std(times)
        kind = "forward+backward" if backward else "forward"
        print(f"[{model_size}] {kind} pass — Avg: {avg_time:.4f}s, Std Dev: {std_dev:.4f}s")
        
        # Final memory stats for timed version
        final_mem = get_mem_info()
        print(f"Final memory stats: {final_mem}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_SIZES.keys(), help="Size of the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up steps")
    parser.add_argument("--backward", action="store_true", help="Enable backward pass")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()
    
    benchmark(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        warmup=args.warmup,
        backward=args.backward,
        profile_mode=args.profile
    )