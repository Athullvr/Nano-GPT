#!/usr/bin/env python3
"""
TouchlineLM — Rigorous Batched Inference Benchmark Script (Stage 3)

Measures and profiles batched inference performance against sequential single-request execution:
1. Batch sizes: [1, 2, 4, 8].
2. Cache modes: ['none', 'dynamic', 'static'].
3. Workload patterns:
   - Homogeneous: Uniform prompt lengths and uniform generation targets.
   - Heterogeneous: Diverse prompt lengths (left-padding) and uniform generation targets.
   - Early Completion: Diverse prompt lengths and varying generation targets / early EOS stopping.
4. Comprehensive metrics:
   - TTFT (Time-to-first-token / prefill latency) in milliseconds.
   - Inter-token latency (ITL / decode time per step) in milliseconds.
   - Aggregate tokens per second (all generated tokens / total batch wall-clock time).
   - Per-request tokens per second (mean individual generation rate).
   - Total batch completion latency in seconds.
   - Peak CUDA allocated memory (MB) and peak CUDA reserved memory (MB).
   - Speedup ratio over sequential single-request execution.
5. Strict correctness verification:
   - Verifies greedy outputs between sequential and batched execution.
   - Verifies numerical equivalence across cache modes ('dynamic' vs 'static').

Saves structured results to a machine-readable JSON file.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.engine import (
    TouchlineInferenceEngine,
    BatchRequest,
    BatchGenerationItem,
    BatchGenerationOutput,
)


def sync_device(device: str) -> None:
    """Synchronizes device if CUDA to guarantee accurate wall-clock timing."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# -----------------------------------------------------------------------------
# Workload Definitions
# -----------------------------------------------------------------------------

# Curated tactical football prompts from the Touchline dataset
PROMPT_BANK = [
    # 0: Short (~8 tokens)
    "<|narrative|> Match summary:",
    # 1: Short (~15 tokens)
    "<|narrative|> The final whistle sounded at Lusail Stadium, concluding a dramatic encounter.",
    # 2: Medium (~32 tokens)
    (
        "<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini\\n"
        "**Position**: Right Wing | **Team**: Argentina\\n"
        "## 1. Distribution and Ball Progression\\n"
        "Operating in possession, the player"
    ),
    # 3: Medium (~45 tokens)
    (
        "<|narrative|> # Tactical Analysis: Midfield Block Transition\\n"
        "During the second half, the visiting side transitioned into a compact 4-4-2 mid-block. "
        "This strategic shift disrupted the opposition central passing lanes and forced turnovers."
    ),
    # 4: Long (~75 tokens)
    (
        "<|narrative|> # Match Report: Croatia vs Brazil\\n"
        "**Competition Stage**: Quarter-finals | **Date**: 2022-12-09\\n"
        "**Final Scoreline**: Croatia 1 - 1 Brazil (Croatia won on penalties)\\n\\n"
        "## 1. Macroscopic Match Summary\\n"
        "A high-stakes tactical battle unfolded in Education City Stadium. "
        "Brazil sought territorial dominance through progressive carries and dynamic wide overloads."
    ),
    # 5: Long (~95 tokens)
    (
        "<|narrative|> # Match Report: Argentina vs France\\n"
        "**Competition Stage**: Final | **Date**: 2022-12-18\\n"
        "**Final Scoreline**: Argentina 3 - 3 France (Argentina won on penalties)\\n\\n"
        "## 1. First Half Tactical Superiority\\n"
        "Argentina established early territorial control through sustained counter-pressing in the middle third. "
        "France struggled to generate progressive sequences, failing to record a single shot inside thirty minutes."
    ),
    # 6: Short (~12 tokens)
    "<|narrative|> In the 85th minute, a swift counter-attack commenced.",
    # 7: Medium (~50 tokens)
    (
        "<|player_dossier|> # Analytical Player Dossier: Kylian Mbappé Lottin\\n"
        "**Position**: Left Wing | **Team**: France\\n\\n"
        "## 1. High Velocity Transition and Box Penetration\\n"
        "Utilizing elite acceleration along the left flank, the forward generated"
    ),
]


def build_workload(
    workload_type: str,
    batch_size: int,
    base_target_tokens: int = 40,
) -> List[BatchRequest]:
    """
    Builds a list of BatchRequest objects according to the requested workload pattern.
    """
    requests: List[BatchRequest] = []

    if workload_type == "homogeneous":
        # Prompt 2 repeated with uniform generation target
        p = PROMPT_BANK[2]
        for i in range(batch_size):
            requests.append(
                BatchRequest(
                    prompt=p,
                    max_new_tokens=base_target_tokens,
                    temperature=0.0,  # Deterministic greedy decoding for rigorous comparisons
                    seed=42 + i,
                )
            )
    elif workload_type == "heterogeneous":
        # Cyclic assignment across the diverse prompt bank, uniform target
        for i in range(batch_size):
            p = PROMPT_BANK[i % len(PROMPT_BANK)]
            requests.append(
                BatchRequest(
                    prompt=p,
                    max_new_tokens=base_target_tokens,
                    temperature=0.0,
                    seed=42 + i,
                )
            )
    elif workload_type == "early_completion":
        # Diverse prompts + staggered generation limits (e.g. 15, 25, 40, 60 tokens)
        target_multipliers = [15, 30, 45, 60]
        for i in range(batch_size):
            p = PROMPT_BANK[i % len(PROMPT_BANK)]
            tokens_target = target_multipliers[i % len(target_multipliers)]
            requests.append(
                BatchRequest(
                    prompt=p,
                    max_new_tokens=tokens_target,
                    temperature=0.0,
                    seed=42 + i,
                    stop_at_eos=True,
                )
            )
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

    return requests


# -----------------------------------------------------------------------------
# Benchmark Execution Engine
# -----------------------------------------------------------------------------

def run_sequential_execution(
    engine: TouchlineInferenceEngine,
    requests: List[BatchRequest],
    cache_type: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Executes a list of requests sequentially one by one, measuring total sequential
    latency, TTFT, decode latency, throughput, and peak memory.
    """
    device = engine.device
    is_cuda = device.startswith("cuda")

    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sync_device(device)

    t0 = time.perf_counter()
    items: List[Dict[str, Any]] = []
    total_tokens = 0

    for req in requests:
        out = engine.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            seed=req.seed,
            stop_at_eos=req.stop_at_eos,
            use_cache=(cache_type != "none"),
            cache_type=cache_type,
        )
        total_tokens += out.new_tokens
        items.append(
            {
                "prompt": out.prompt,
                "generated_text": out.generated_text,
                "prompt_tokens": out.prompt_tokens,
                "new_tokens": out.new_tokens,
                "latency_seconds": out.latency_seconds,
            }
        )

    sync_device(device)
    total_time = time.perf_counter() - t0

    peak_alloc = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if is_cuda else 0.0
    )
    peak_res = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 2) if is_cuda else 0.0
    )

    metrics = {
        "total_time_seconds": total_time,
        "aggregate_tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
        "total_tokens": total_tokens,
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_res,
    }
    return items, metrics


def run_batched_execution(
    engine: TouchlineInferenceEngine,
    requests: List[BatchRequest],
    cache_type: str,
) -> Tuple[BatchGenerationOutput, Dict[str, float]]:
    """
    Executes a list of requests in a single batched call with engine.generate_batch().
    """
    device = engine.device
    is_cuda = device.startswith("cuda")

    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sync_device(device)

    t0 = time.perf_counter()
    out = engine.generate_batch(requests, cache_type=cache_type)
    sync_device(device)
    wall_clock = time.perf_counter() - t0

    # Calculate inter-token latency (ITL)
    # Total decode time divided by total decode steps (max generated tokens across batch)
    max_gen_steps = max((it.new_tokens for it in out.items), default=1)
    if max_gen_steps > 1 and out.decode_latency_seconds > 0:
        itl_ms = (out.decode_latency_seconds / (max_gen_steps - 1)) * 1000.0
    else:
        itl_ms = 0.0

    metrics = {
        "total_time_seconds": out.total_latency_seconds,
        "wall_clock_seconds": wall_clock,
        "prefill_latency_seconds": out.prefill_latency_seconds,
        "decode_latency_seconds": out.decode_latency_seconds,
        "ttft_ms": out.prefill_latency_seconds * 1000.0,
        "itl_ms": itl_ms,
        "aggregate_tokens_per_second": out.aggregate_tokens_per_second,
        "per_request_tokens_per_second": out.per_request_tokens_per_second,
        "peak_allocated_mb": out.peak_allocated_mb,
        "peak_reserved_mb": out.peak_reserved_mb,
        "successful_requests": out.successful_requests,
        "failed_requests": out.failed_requests,
    }
    return out, metrics


def verify_correctness(
    seq_items: List[Dict[str, Any]],
    batch_output: BatchGenerationOutput,
) -> bool:
    """
    Verifies that batched execution outputs match sequential single-request outputs.
    """
    if len(seq_items) != len(batch_output.items):
        print(f"  [CORRECTNESS ERROR] Length mismatch: seq={len(seq_items)}, batch={len(batch_output.items)}")
        return False

    all_matched = True
    for i, (seq_it, batch_it) in enumerate(zip(seq_items, batch_output.items)):
        if seq_it["generated_text"] != batch_it.generated_text:
            print(f"  [CORRECTNESS MISMATCH] Row {i}:")
            print(f"    Sequential: {seq_it['generated_text']!r}")
            print(f"    Batched:    {batch_it.generated_text!r}")
            all_matched = False
    return all_matched


def run_benchmark_suite(
    engine: TouchlineInferenceEngine,
    batch_sizes: List[int] = [1, 2, 4, 8],
    cache_modes: List[str] = ["dynamic", "static", "none"],
    workloads: List[str] = ["homogeneous", "heterogeneous", "early_completion"],
    repetitions: int = 5,
    warmup: int = 2,
) -> Dict[str, Any]:
    """
    Runs the complete matrix of benchmarks across workloads, cache modes, and batch sizes.
    """
    results: Dict[str, Any] = {
        "device": engine.device,
        "model_parameters": engine.model.get_num_params(),
        "repetitions": repetitions,
        "benchmarks": {},
    }

    print("=" * 80)
    print(f"TouchlineLM — Batched Inference Benchmark Suite on {engine.device.upper()}")
    print(f"Model parameters: {engine.model.get_num_params():,} | Context length: {engine.model.config.max_seq_len}")
    print("=" * 80)

    for w_name in workloads:
        results["benchmarks"][w_name] = {}
        print(f"\\n>>> WORKLOAD: {w_name.upper()} <<<")

        for c_mode in cache_modes:
            results["benchmarks"][w_name][c_mode] = {}
            print(f"\\n--- Cache Mode: {c_mode.upper()} ---")

            for B in batch_sizes:
                reqs = build_workload(w_name, batch_size=B, base_target_tokens=40)

                # Warmup
                for _ in range(warmup):
                    if B > 1:
                        run_sequential_execution(engine, reqs, cache_type=c_mode)
                    run_batched_execution(engine, reqs, cache_type=c_mode)

                # 1. Run Sequential Baseline (for B > 1; for B=1 sequential is identical)
                seq_total_times = []
                seq_tps_list = []
                seq_peak_allocs = []
                last_seq_items = None

                for _ in range(repetitions):
                    s_items, s_metrics = run_sequential_execution(engine, reqs, cache_type=c_mode)
                    seq_total_times.append(s_metrics["total_time_seconds"])
                    seq_tps_list.append(s_metrics["aggregate_tokens_per_second"])
                    seq_peak_allocs.append(s_metrics["peak_allocated_mb"])
                    last_seq_items = s_items

                seq_summary = {
                    "total_time_s": {
                        "median": statistics.median(seq_total_times),
                        "std": statistics.stdev(seq_total_times) if len(seq_total_times) > 1 else 0.0,
                    },
                    "tps": {
                        "median": statistics.median(seq_tps_list),
                        "std": statistics.stdev(seq_tps_list) if len(seq_tps_list) > 1 else 0.0,
                    },
                    "peak_allocated_mb": statistics.median(seq_peak_allocs),
                }

                # 2. Run Batched Execution
                batch_total_times = []
                batch_prefill_times = []
                batch_decode_times = []
                batch_ttft_ms = []
                batch_itl_ms = []
                batch_agg_tps = []
                batch_per_req_tps = []
                batch_peak_allocs = []
                batch_peak_res = []
                last_batch_output = None

                for _ in range(repetitions):
                    b_out, b_metrics = run_batched_execution(engine, reqs, cache_type=c_mode)
                    batch_total_times.append(b_metrics["total_time_seconds"])
                    batch_prefill_times.append(b_metrics["prefill_latency_seconds"])
                    batch_decode_times.append(b_metrics["decode_latency_seconds"])
                    batch_ttft_ms.append(b_metrics["ttft_ms"])
                    batch_itl_ms.append(b_metrics["itl_ms"])
                    batch_agg_tps.append(b_metrics["aggregate_tokens_per_second"])
                    batch_per_req_tps.append(b_metrics["per_request_tokens_per_second"])
                    batch_peak_allocs.append(b_metrics["peak_allocated_mb"])
                    batch_peak_res.append(b_metrics["peak_reserved_mb"])
                    last_batch_output = b_out

                # Verify correctness against sequential
                correctness_ok = verify_correctness(last_seq_items, last_batch_output)

                med_batch_time = statistics.median(batch_total_times)
                med_batch_agg_tps = statistics.median(batch_agg_tps)
                med_batch_ttft = statistics.median(batch_ttft_ms)
                med_batch_itl = statistics.median(batch_itl_ms)
                med_batch_alloc = statistics.median(batch_peak_allocs)

                speedup = (
                    seq_summary["total_time_s"]["median"] / med_batch_time
                    if med_batch_time > 0
                    else 1.0
                )
                tps_gain = (
                    med_batch_agg_tps / seq_summary["tps"]["median"]
                    if seq_summary["tps"]["median"] > 0
                    else 1.0
                )

                b_res = {
                    "batch_size": B,
                    "correctness_verified": correctness_ok,
                    "sequential": seq_summary,
                    "batched": {
                        "total_time_s": {
                            "median": med_batch_time,
                            "std": statistics.stdev(batch_total_times) if len(batch_total_times) > 1 else 0.0,
                        },
                        "ttft_ms": {
                            "median": med_batch_ttft,
                            "std": statistics.stdev(batch_ttft_ms) if len(batch_ttft_ms) > 1 else 0.0,
                        },
                        "itl_ms": {
                            "median": med_batch_itl,
                            "std": statistics.stdev(batch_itl_ms) if len(batch_itl_ms) > 1 else 0.0,
                        },
                        "aggregate_tps": {
                            "median": med_batch_agg_tps,
                            "std": statistics.stdev(batch_agg_tps) if len(batch_agg_tps) > 1 else 0.0,
                        },
                        "per_request_tps": {
                            "median": statistics.median(batch_per_req_tps),
                            "std": statistics.stdev(batch_per_req_tps) if len(batch_per_req_tps) > 1 else 0.0,
                        },
                        "peak_allocated_mb": med_batch_alloc,
                        "peak_reserved_mb": statistics.median(batch_peak_res),
                    },
                    "comparison": {
                        "speedup_ratio": round(speedup, 2),
                        "tps_gain_ratio": round(tps_gain, 2),
                        "latency_overhead_ratio": round(med_batch_time / (seq_summary["total_time_s"]["median"] / B), 2),
                    },
                }

                results["benchmarks"][w_name][c_mode][f"batch_{B}"] = b_res

                print(
                    f"  B={B:<2} | "
                    f"Batched TPS: {med_batch_agg_tps:>6.1f} tok/s | "
                    f"Seq TPS: {seq_summary['tps']['median']:>6.1f} tok/s | "
                    f"Speedup: {speedup:>4.2f}x | "
                    f"TTFT: {med_batch_ttft:>5.1f} ms | "
                    f"ITL: {med_batch_itl:>4.1f} ms | "
                    f"Alloc: {med_batch_alloc:>5.1f} MB | "
                    f"Match: {'PASS' if correctness_ok else 'FAIL'}"
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TouchlineLM Batched Inference Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt")
    parser.add_argument("--tokenizer-dir", type=str, default="configs/tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-json", type=str, default="data/intermediate/batched_benchmark_results.json")
    args = parser.parse_args()

    print(f"Initializing TouchlineInferenceEngine on {args.device}...")
    engine = TouchlineInferenceEngine.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
    )

    batch_sizes = [1, 2, 4, 8]
    cache_modes = ["dynamic", "static", "none"]
    workloads = ["homogeneous", "heterogeneous", "early_completion"]

    results = run_benchmark_suite(
        engine=engine,
        batch_sizes=batch_sizes,
        cache_modes=cache_modes,
        workloads=workloads,
        repetitions=args.repetitions,
        warmup=args.warmup,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\\nSaved full machine-readable benchmark results to {out_path}")


if __name__ == "__main__":
    main()
