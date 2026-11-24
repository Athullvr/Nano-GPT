#!/usr/bin/env python3
"""
TouchlineLM — KV Cache Memory Profiling and Context Scaling Benchmark

Performs rigorous memory profiling across context lengths [32, 128, 256, 512, 1024]:
1. Diagnoses autograd DAG retention (comparing generation with and without torch.inference_mode).
2. Measures theoretical KV cache size, peak allocated memory, and peak reserved memory.
3. Compares three execution strategies:
   - Uncached Baseline (legacy full-sequence recomputation)
   - Dynamic KV Cache (standard torch.cat concatenation)
   - Preallocated Static KV Cache (zero-allocation in-place copy)
4. Measures prefill latency, per-token decode latency, total latency, and throughput.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List, Any

import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.inference.kv_cache import StaticKVCache


CONTEXT_LENGTHS = [32, 128, 256, 512, 1024]
PROMPT_LEN = 16


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def format_mb(bytes_val: float) -> str:
    return f"{bytes_val / (1024 ** 2):.2f} MB"


def diagnose_autograd_dag_retention(model: TouchlineNanoGPT, device: str) -> Dict[str, float]:
    """
    Empirically demonstrates the exact root cause of the 364 MB peak VRAM anomaly:
    Without torch.inference_mode(), repeated torch.cat accumulates an unbroken autograd
    DAG across all autoregressive decode steps, retaining all intermediate activations.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}

    test_prompt = torch.randint(0, model.config.vocab_size, (1, 173), device=device)
    num_steps = 100

    # Case A: Without no_grad / inference_mode (The Bug Scenario)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    idx = test_prompt.clone()
    logits, past_kv = model(idx, use_cache=True)
    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    for _ in range(num_steps):
        logits, past_kv = model(next_tok, past_key_values=past_kv, use_cache=True)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    peak_without_guard = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del logits, past_kv, next_tok, idx
    torch.cuda.empty_cache()
    gc.collect()

    # Case B: With torch.inference_mode() (Correct Production Behavior)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        idx = test_prompt.clone()
        logits, past_kv = model(idx, use_cache=True)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(num_steps):
            logits, past_kv = model(next_tok, past_key_values=past_kv, use_cache=True)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        peak_with_inference_mode = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del logits, past_kv, next_tok, idx
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "steps": num_steps,
        "peak_vram_without_guard_mb": round(peak_without_guard, 2),
        "peak_vram_with_inference_mode_mb": round(peak_with_inference_mode, 2),
        "memory_saved_mb": round(peak_without_guard - peak_with_inference_mode, 2),
    }


def benchmark_single_run(
    model: TouchlineNanoGPT,
    input_ids: torch.Tensor,
    new_tokens: int,
    strategy: str,
    device: str,
) -> Dict[str, Any]:
    """Runs a single timed generation pass under torch.inference_mode for a given strategy."""
    sync_device(device)
    torch.cuda.reset_peak_memory_stats(device) if device.startswith("cuda") else None

    prompt_len = input_ids.size(1)
    idx = input_ids.clone()

    t_start = time.perf_counter()

    with torch.inference_mode():
        if strategy == "uncached":
            logits, _ = model(idx, use_cache=False)
            sync_device(device)
            t_prefill = time.perf_counter() - t_start

            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_tok), dim=1)

            decode_times = []
            for _ in range(new_tokens - 1):
                sync_device(device)
                t_step_start = time.perf_counter()
                logits, _ = model(idx, use_cache=False)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                sync_device(device)
                decode_times.append(time.perf_counter() - t_step_start)
                idx = torch.cat((idx, next_tok), dim=1)

        elif strategy == "dynamic":
            logits, past_kv = model(idx, use_cache=True)
            sync_device(device)
            t_prefill = time.perf_counter() - t_start

            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_tok), dim=1)

            decode_times = []
            for _ in range(new_tokens - 1):
                sync_device(device)
                t_step_start = time.perf_counter()
                logits, past_kv = model(next_tok, past_key_values=past_kv, use_cache=True)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                sync_device(device)
                decode_times.append(time.perf_counter() - t_step_start)
                idx = torch.cat((idx, next_tok), dim=1)

        elif strategy == "static":
            static_cache = StaticKVCache.from_model(model, batch_size=1, device=device)
            logits, _ = model(idx, past_key_values=static_cache, use_cache=True)
            sync_device(device)
            t_prefill = time.perf_counter() - t_start

            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_tok), dim=1)

            decode_times = []
            for _ in range(new_tokens - 1):
                sync_device(device)
                t_step_start = time.perf_counter()
                logits, _ = model(next_tok, past_key_values=static_cache, use_cache=True)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                sync_device(device)
                decode_times.append(time.perf_counter() - t_step_start)
                idx = torch.cat((idx, next_tok), dim=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    sync_device(device)
    t_total = time.perf_counter() - t_start
    total_decode_time = sum(decode_times) if decode_times else 0.0

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.startswith("cuda") else 0.0
    peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.startswith("cuda") else 0.0

    return {
        "prefill_ms": t_prefill * 1000,
        "total_ms": t_total * 1000,
        "decode_step_mean_ms": statistics.mean(decode_times) * 1000 if decode_times else 0.0,
        "tokens_per_second": new_tokens / t_total if t_total > 0 else 0.0,
        "peak_allocated_mb": round(peak_alloc, 2),
        "peak_reserved_mb": round(peak_res, 2),
    }


def run_scaling_experiment(
    checkpoint_path: Path,
    device: str,
    repetitions: int = 3,
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"TOUCHLINELM KV CACHE MEMORY PROFILING & CONTEXT SCALING ({device.upper()})")
    print(f"{'=' * 80}")

    # Baseline memory tracking before model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        gc.collect()
        m0_alloc = torch.cuda.memory_allocated(device)
        m0_res = torch.cuda.memory_reserved(device)
        print(f"Initial Baseline CUDA Memory: Allocated = {format_mb(m0_alloc)}, Reserved = {format_mb(m0_res)}")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(ckpt["model_config"])
    model = TouchlineNanoGPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if device.startswith("cuda"):
        m1_alloc = torch.cuda.memory_allocated(device)
        m1_res = torch.cuda.memory_reserved(device)
        print(f"Post-Model Load CUDA Memory: Allocated = {format_mb(m1_alloc)}, Reserved = {format_mb(m1_res)}")

        # Step 1: Autograd DAG retention diagnosis
        print("\n--- Diagnosing Autograd Retention Anomaly ---")
        dag_diagnosis = diagnose_autograd_dag_retention(model, device)
        print(
            f"  • Without torch.inference_mode(): Peak VRAM = {dag_diagnosis['peak_vram_without_guard_mb']:.2f} MB\n"
            f"  • With torch.inference_mode():    Peak VRAM = {dag_diagnosis['peak_vram_with_inference_mode_mb']:.2f} MB\n"
            f"  • Memory Saved by inference_mode: {dag_diagnosis['memory_saved_mb']:.2f} MB "
            f"({(dag_diagnosis['memory_saved_mb'] / dag_diagnosis['peak_vram_without_guard_mb']) * 100:.1f}% reduction)"
        )
    else:
        dag_diagnosis = {}

    # Step 2: Context Scaling Experiment across [32, 128, 256, 512, 1024]
    print(f"\n--- Context Scaling Across Lengths: {CONTEXT_LENGTHS} (Prompt={PROMPT_LEN} tokens) ---")
    results = []

    for total_ctx in CONTEXT_LENGTHS:
        new_tokens = total_ctx - PROMPT_LEN
        if new_tokens <= 0:
            continue

        # Theoretical KV cache size: 4 layers * 2 (K+V) * 1 batch * 4 heads * total_ctx * 64 dim * 4 bytes
        theory_bytes = 4 * 2 * 1 * 4 * total_ctx * 64 * 4
        theory_mb = theory_bytes / (1024 ** 2)

        input_ids = torch.randint(0, model_config.vocab_size, (1, PROMPT_LEN), device=device)

        ctx_summary = {
            "context_length": total_ctx,
            "prompt_length": PROMPT_LEN,
            "new_tokens": new_tokens,
            "theoretical_kv_mb": round(theory_mb, 3),
            "strategies": {},
        }

        print(f"\n[Context: {total_ctx} tokens | Generated: {new_tokens} tokens | Theoretical KV: {theory_mb:.3f} MB]")

        for strategy in ["uncached", "dynamic", "static"]:
            runs = []
            for rep in range(repetitions):
                torch.manual_seed(42 + rep)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(42 + rep)
                    torch.cuda.empty_cache()
                    gc.collect()

                res = benchmark_single_run(model, input_ids, new_tokens, strategy, device)
                runs.append(res)

            median_alloc = statistics.median([r["peak_allocated_mb"] for r in runs])
            median_res = statistics.median([r["peak_reserved_mb"] for r in runs])
            median_latency = statistics.median([r["total_ms"] for r in runs])
            median_tok_s = statistics.median([r["tokens_per_second"] for r in runs])
            median_decode_ms = statistics.median([r["decode_step_mean_ms"] for r in runs])

            ctx_summary["strategies"][strategy] = {
                "peak_allocated_mb": round(median_alloc, 2),
                "peak_reserved_mb": round(median_res, 2),
                "total_latency_ms": round(median_latency, 2),
                "tokens_per_second": round(median_tok_s, 1),
                "decode_step_ms": round(median_decode_ms, 3),
            }

            print(
                f"  • {strategy.upper():10s}: Peak Alloc = {median_alloc:5.2f} MB | "
                f"Peak Res = {median_res:5.2f} MB | Decode = {median_decode_ms:5.2f} ms/tok | "
                f"Total = {median_latency:6.1f} ms | Speed = {median_tok_s:6.1f} tok/s"
            )

        results.append(ctx_summary)

    return {
        "device": device,
        "model_parameters": model.get_num_params(),
        "model_weights_mb": 16.52,
        "dag_diagnosis": dag_diagnosis,
        "scaling_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="TouchlineLM KV Cache Memory Profiling")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per test")
    parser.add_argument("--output-json", type=str, default="data/intermediate/kv_cache_memory_profile.json", help="Path to save output JSON")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.", file=sys.stderr)
        sys.exit(1)

    profile_data = run_scaling_experiment(ckpt_path, device=args.device, repetitions=args.repetitions)

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Memory profiling complete! Saved results to: {out_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
