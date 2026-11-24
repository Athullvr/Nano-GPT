#!/usr/bin/env python3
"""
TouchlineLM — Rigorous Inference Baseline Benchmark Script

Measures and profiles baseline (non-KV cached) inference performance across:
1. Model loading latency.
2. Prompt prefill latency (Time-To-First-Token / TTFT).
3. Per-token decode latency distribution (mean, median, std, p95).
4. Total generation latency and throughput (tokens/second).
5. Peak memory allocation (CPU RAM / CUDA VRAM).

Runs on both CPU and CUDA with warm-up iterations and multiple repetitions.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer


BENCHMARK_PROMPTS = [
    {
        "id": "short_prompt",
        "name": "Match Narrative (Short, ~32 prompt tokens -> 100 new tokens)",
        "prompt": (
            "<|narrative|> # Match Report: Argentina vs France\n"
            "**Competition Stage**: Final | **Date**: 2022-12-18\n"
            "**Final Scoreline**: Argentina"
        ),
        "target_tokens": 100,
    },
    {
        "id": "medium_prompt",
        "name": "Player Dossier (Medium, ~60 prompt tokens -> 150 new tokens)",
        "prompt": (
            "<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini\n"
            "**Position**: Right Wing | **Team**: Argentina\n\n"
            "## 1. Distribution and Ball Progression\n"
            "Operating in possession, the player executed"
        ),
        "target_tokens": 150,
    },
    {
        "id": "long_prompt",
        "name": "Tactical Macro Analysis (Long, ~120 prompt tokens -> 200 new tokens)",
        "prompt": (
            "<|narrative|> # Match Report: Croatia vs Brazil\n"
            "**Competition Stage**: Quarter-finals | **Date**: 2022-12-09\n"
            "**Final Scoreline**: Croatia 1 - 1 Brazil (Croatia won on penalties)\n\n"
            "## 1. Macroscopic Match Summary\n"
            "A high-stakes tactical battle unfolded in Education City Stadium. "
            "Brazil sought territorial dominance through progressive carries and dynamic wide overloads, "
            "whereas Croatia absorbed pressure via a disciplined defensive mid-block and elite midfield retention.\n\n"
            "### Question 1: How did the expected goals (xG) metrics compare to the actual match outcome?"
        ),
        "target_tokens": 200,
    },
]


def sync_device(device: str) -> None:
    """Synchronizes device if CUDA to guarantee accurate wall-clock timing."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def benchmark_model_loading(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    device: str,
    repetitions: int = 5,
) -> Dict[str, Any]:
    """
    Measures isolated model loading time from disk across multiple repetitions.
    Includes checkpoint deserialization, model instantiation, state_dict loading, and device transfer.
    """
    times = []
    print(f"\n[1/3] Benchmarking Model Loading on {device.upper()} ({repetitions} repetitions)...")

    for rep in range(repetitions):
        sync_device(device)
        t0 = time.perf_counter()

        # Step A: Load checkpoint on CPU
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Step B: Model config
        model_config = ModelConfig.from_dict(ckpt["model_config"])

        # Step C: Instantiate and load state dict
        model = TouchlineNanoGPT(model_config)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        sync_device(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  • Run {rep + 1}/{repetitions}: {elapsed * 1000:.2f} ms")

        # Cleanup memory between runs
        del model
        del ckpt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "repetitions": repetitions,
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0.0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def benchmark_single_generation(
    model: TouchlineNanoGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Executes a single generation run, measuring prompt prefill and token-by-token decode latencies
    with optional per-layer KV caching.
    """
    idx = input_ids.clone()
    prompt_len = idx.size(1)

    # 1. Measure Prompt Prefill Latency (Time-To-First-Token)
    sync_device(device)
    t_prefill_start = time.perf_counter()

    if use_cache:
        logits, past_key_values = model(idx, use_cache=True)
    else:
        logits, _ = model(idx, use_cache=False)
        past_key_values = None

    logits_step = logits[:, -1, :].clone() / max(temperature, 1e-5)

    # Top-k filtering
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
        logits_step[logits_step < v[:, [-1]]] = -float("Inf")

    # Top-p filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits_step, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits_step[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits_step, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    sync_device(device)
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start

    idx = torch.cat((idx, next_token), dim=1)

    # 2. Measure Token-by-Token Decode Latencies
    decode_step_times = []

    for _ in range(max_new_tokens - 1):
        sync_device(device)
        t_decode_start = time.perf_counter()

        if use_cache:
            logits, past_key_values = model(
                next_token, past_key_values=past_key_values, use_cache=True
            )
        else:
            if idx.size(1) > model.config.max_seq_len:
                idx_cond = idx[:, -model.config.max_seq_len:]
            else:
                idx_cond = idx
            logits, _ = model(idx_cond, use_cache=False)

        logits_step = logits[:, -1, :].clone() / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
            logits_step[logits_step < v[:, [-1]]] = -float("Inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_step, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits_step[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        sync_device(device)
        t_decode_end = time.perf_counter()

        decode_step_times.append(t_decode_end - t_decode_start)
        idx = torch.cat((idx, next_token), dim=1)

    total_decode_time = sum(decode_step_times)
    total_generation_time = prefill_time + total_decode_time
    tokens_per_second = max_new_tokens / total_generation_time if total_generation_time > 0 else 0.0
    decode_tok_per_sec = (max_new_tokens - 1) / total_decode_time if total_decode_time > 0 else 0.0

    first_10_decode_ms = statistics.mean(decode_step_times[:10]) * 1000 if len(decode_step_times) >= 10 else 0.0
    last_10_decode_ms = statistics.mean(decode_step_times[-10:]) * 1000 if len(decode_step_times) >= 10 else 0.0

    return {
        "use_cache": use_cache,
        "prompt_tokens": prompt_len,
        "new_tokens": max_new_tokens,
        "prefill_latency_ms": prefill_time * 1000,
        "total_decode_latency_ms": total_decode_time * 1000,
        "total_generation_latency_ms": total_generation_time * 1000,
        "mean_decode_step_ms": statistics.mean(decode_step_times) * 1000,
        "median_decode_step_ms": statistics.median(decode_step_times) * 1000,
        "std_decode_step_ms": statistics.stdev(decode_step_times) * 1000 if len(decode_step_times) > 1 else 0.0,
        "first_10_decode_ms": first_10_decode_ms,
        "last_10_decode_ms": last_10_decode_ms,
        "tokens_per_second": tokens_per_second,
        "decode_tokens_per_second": decode_tok_per_sec,
    }


def run_prompt_benchmark(
    model: TouchlineNanoGPT,
    input_ids: torch.Tensor,
    target_tokens: int,
    device: str,
    repetitions: int,
    use_cache: bool,
) -> Dict[str, Any]:
    """Runs multiple repetitions for a given prompt and cache configuration."""
    runs = []
    peak_vram_mb = 0.0

    for rep in range(repetitions):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(1000 + rep)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(1000 + rep)

        metrics = benchmark_single_generation(
            model=model,
            input_ids=input_ids,
            max_new_tokens=target_tokens,
            device=device,
            use_cache=use_cache,
        )

        if device.startswith("cuda"):
            vram_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_vram_mb = max(peak_vram_mb, vram_allocated)

        runs.append(metrics)

    prefill_times = [r["prefill_latency_ms"] for r in runs]
    decode_step_means = [r["mean_decode_step_ms"] for r in runs]
    first_10_decode_times = [r["first_10_decode_ms"] for r in runs]
    last_10_decode_times = [r["last_10_decode_ms"] for r in runs]
    total_latencies = [r["total_generation_latency_ms"] for r in runs]
    tok_speeds = [r["tokens_per_second"] for r in runs]
    decode_tok_speeds = [r["decode_tokens_per_second"] for r in runs]

    return {
        "use_cache": use_cache,
        "repetitions": repetitions,
        "peak_vram_mb": round(peak_vram_mb, 2) if device.startswith("cuda") else None,
        "prefill_ms": {
            "median": round(statistics.median(prefill_times), 2),
            "mean": round(statistics.mean(prefill_times), 2),
            "std": round(statistics.stdev(prefill_times), 2) if len(prefill_times) > 1 else 0.0,
            "min": round(min(prefill_times), 2),
            "max": round(max(prefill_times), 2),
        },
        "decode_per_token_ms": {
            "median": round(statistics.median(decode_step_means), 3),
            "mean": round(statistics.mean(decode_step_means), 3),
            "std": round(statistics.stdev(decode_step_means), 3) if len(decode_step_means) > 1 else 0.0,
            "first_10_mean": round(statistics.mean(first_10_decode_times), 3),
            "last_10_mean": round(statistics.mean(last_10_decode_times), 3),
        },
        "total_latency_ms": {
            "median": round(statistics.median(total_latencies), 2),
            "mean": round(statistics.mean(total_latencies), 2),
            "std": round(statistics.stdev(total_latencies), 2) if len(total_latencies) > 1 else 0.0,
        },
        "tokens_per_second": {
            "median": round(statistics.median(tok_speeds), 1),
            "mean": round(statistics.mean(tok_speeds), 1),
            "std": round(statistics.stdev(tok_speeds), 1) if len(tok_speeds) > 1 else 0.0,
        },
        "decode_tokens_per_second": {
            "median": round(statistics.median(decode_tok_speeds), 1),
            "mean": round(statistics.mean(decode_tok_speeds), 1),
            "std": round(statistics.stdev(decode_tok_speeds), 1) if len(decode_tok_speeds) > 1 else 0.0,
        },
    }


def run_benchmark_suite(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    device: str,
    mode: str = "both",
    warmup_runs: int = 3,
    repetitions: int = 5,
) -> Dict[str, Any]:
    """
    Executes benchmark suite comparing cached vs uncached inference across all prompts and repetitions.
    """
    print(f"\n{'=' * 75}")
    print(f"TOUCHLINELM INFERENCE COMPARATIVE BENCHMARK ({device.upper()} | Mode: {mode.upper()})")
    print(f"{'=' * 75}")

    # 1. Model Loading Benchmark
    loading_stats = benchmark_model_loading(
        checkpoint_path=checkpoint_path,
        tokenizer_dir=tokenizer_dir,
        device=device,
        repetitions=repetitions,
    )

    # 2. Load model instance for generation benchmarking
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(ckpt["model_config"])
    model = TouchlineNanoGPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    tokenizer = TouchlineBPETokenizer.load(tokenizer_dir)

    # 3. Warm-up runs to prime PyTorch / CUDA kernel compilation
    print(f"\n[2/3] Executing {warmup_runs} warm-up iterations on {device.upper()}...")
    warmup_ids = torch.tensor(
        [tokenizer.encode("Warmup prompt for TouchlineLM inference profiling", allowed_special="all")],
        dtype=torch.long,
        device=device,
    )
    for _ in range(warmup_runs):
        _ = benchmark_single_generation(
            model=model,
            input_ids=warmup_ids,
            max_new_tokens=25,
            device=device,
            use_cache=True,
        )
        _ = benchmark_single_generation(
            model=model,
            input_ids=warmup_ids,
            max_new_tokens=25,
            device=device,
            use_cache=False,
        )
    print("  • Warm-up complete.")

    # 4. Prompt Benchmarks
    print(f"\n[3/3] Benchmarking Generation Across Prompts ({repetitions} repetitions each)...")
    prompt_results = []

    for prompt_info in BENCHMARK_PROMPTS:
        p_id = prompt_info["id"]
        p_name = prompt_info["name"]
        prompt_text = prompt_info["prompt"]
        target_tokens = prompt_info["target_tokens"]

        token_ids = tokenizer.encode(prompt_text, allowed_special="all")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        prompt_len = len(token_ids)

        print(f"\n--- Testing: {p_name} ---")
        print(f"    Prompt Length: {prompt_len} tokens | Target Generation: {target_tokens} tokens")

        result_entry = {
            "id": p_id,
            "name": p_name,
            "prompt_tokens": prompt_len,
            "new_tokens": target_tokens,
        }

        # Benchmark Uncached if requested
        if mode in ("both", "uncached"):
            print("  [Profiling Uncached Baseline...]")
            uncached_stats = run_prompt_benchmark(
                model=model,
                input_ids=input_ids,
                target_tokens=target_tokens,
                device=device,
                repetitions=repetitions,
                use_cache=False,
            )
            result_entry["uncached"] = uncached_stats
            print(
                f"    Uncached: Decode = {uncached_stats['decode_per_token_ms']['median']:.2f} ms/tok "
                f"(first 10: {uncached_stats['decode_per_token_ms']['first_10_mean']:.2f} ms, "
                f"last 10: {uncached_stats['decode_per_token_ms']['last_10_mean']:.2f} ms) | "
                f"Total = {uncached_stats['total_latency_ms']['median']:.1f} ms | "
                f"Speed = {uncached_stats['tokens_per_second']['median']:.1f} tok/s"
            )

        # Benchmark Cached if requested
        if mode in ("both", "cached"):
            print("  [Profiling KV-Cached Inference...]")
            cached_stats = run_prompt_benchmark(
                model=model,
                input_ids=input_ids,
                target_tokens=target_tokens,
                device=device,
                repetitions=repetitions,
                use_cache=True,
            )
            result_entry["cached"] = cached_stats
            print(
                f"    KV-Cached: Decode = {cached_stats['decode_per_token_ms']['median']:.2f} ms/tok "
                f"(first 10: {cached_stats['decode_per_token_ms']['first_10_mean']:.2f} ms, "
                f"last 10: {cached_stats['decode_per_token_ms']['last_10_mean']:.2f} ms) | "
                f"Total = {cached_stats['total_latency_ms']['median']:.1f} ms | "
                f"Speed = {cached_stats['tokens_per_second']['median']:.1f} tok/s"
            )

        # Compute speedup if both were measured
        if "uncached" in result_entry and "cached" in result_entry:
            uncached_total = result_entry["uncached"]["total_latency_ms"]["median"]
            cached_total = result_entry["cached"]["total_latency_ms"]["median"]
            uncached_decode = result_entry["uncached"]["decode_per_token_ms"]["median"]
            cached_decode = result_entry["cached"]["decode_per_token_ms"]["median"]

            speedup_total = uncached_total / cached_total if cached_total > 0 else 0.0
            speedup_decode = uncached_decode / cached_decode if cached_decode > 0 else 0.0

            result_entry["speedup"] = {
                "total_speedup": round(speedup_total, 2),
                "decode_speedup": round(speedup_decode, 2),
            }
            print(f"    >>> SPEEDUP: Total = {speedup_total:.2f}x | Decode = {speedup_decode:.2f}x")

        prompt_results.append(result_entry)

    return {
        "device": device,
        "mode": mode,
        "loading": loading_stats,
        "prompts": prompt_results,
    }


def main():
    parser = argparse.ArgumentParser(description="TouchlineLM Inference Comparative Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer-dir", type=str, default="configs/tokenizer", help="Tokenizer directory")
    parser.add_argument("--device", type=str, default="both", choices=["cpu", "cuda", "both"], help="Device to benchmark")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "cached", "uncached"], help="Benchmark mode")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of repetitions per prompt")
    parser.add_argument("--output-json", type=str, default="data/intermediate/inference_kv_cache_comparison.json", help="Path to save JSON results")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    tok_dir = Path(args.tokenizer_dir)

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    devices_to_test = []
    if args.device == "both":
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("Error: CUDA requested but not available.", file=sys.stderr)
            sys.exit(1)
        devices_to_test = ["cuda"]
    else:
        devices_to_test = ["cpu"]

    suite_results = {}
    for dev in devices_to_test:
        suite_results[dev] = run_benchmark_suite(
            checkpoint_path=ckpt_path,
            tokenizer_dir=tok_dir,
            device=dev,
            mode=args.mode,
            repetitions=args.repetitions,
        )

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    print(f"\n{'=' * 75}")
    print(f"Benchmark run complete! Results persisted to: {out_file}")
    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    main()
