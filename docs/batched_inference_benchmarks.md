# TouchlineLM — Batched Inference Benchmark & Audit Report (Phase 3B Stage 3)

## 1. Executive Summary & Audit Overview

This document presents the audited empirical benchmarks for batched autoregressive inference on **TouchlineLM** (4.2M parameters, FP32, $n_{\text{layers}}=4, d_{\text{model}}=256, n_{\text{heads}}=4, d_{\text{head}}=64, \text{context}=1024$).
Benchmarks were executed on local hardware (**NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM, CUDA 12.8**) using:
[`scripts/benchmark_batched_inference.py`](file:///home/athulvr/Projects/Nano Gpt/scripts/benchmark_batched_inference.py).

Raw machine-readable results are preserved in [`data/intermediate/batched_benchmark_results.json`](file:///home/athulvr/Projects/Nano Gpt/data/intermediate/batched_benchmark_results.json).
Audited and reconciled speedup calculations across all baselines are stored in [`data/intermediate/batched_benchmark_results_audited.json`](file:///home/athulvr/Projects/Nano Gpt/data/intermediate/batched_benchmark_results_audited.json).

### Audit Findings on Speedup Calculation

An audit of the reported homogeneous numbers:
- $B=2$: $1,387.7\text{ tok/s}$
- $B=4$: $2,693.2\text{ tok/s}$
- $B=8$: $4,482.3\text{ tok/s}$
revealed why they did not equal simple divisions against $B=1$ batched throughput ($736.4\text{ tok/s}$):
1. **The Benchmark Baseline is Iso-Workload Sequential Serving**:
   The benchmark measured the exact sequential time $T_{\text{seq}}(B)$ required to serve that **exact same set of $B$ requests** one by one.
   - At $B=2$, sequential execution required $0.0833\text{ s}$ ($959.9\text{ tok/s}$), while batched execution required $0.0576\text{ s}$ ($1,387.7\text{ tok/s}$). The true time speedup is:
     $$\text{Speedup} = \frac{0.0833\text{ s}}{0.0576\text{ s}} = \mathbf{1.45\times}$$
   - At $B=8$, sequential execution required $0.3198\text{ s}$ ($1,009.7\text{ tok/s}$), while batched execution required $0.0719\text{ s}$ ($4,482.3\text{ tok/s}$). The true time speedup is:
     $$\text{Speedup} = \frac{0.3198\text{ s}}{0.0719\text{ s}} = \mathbf{4.44\times}$$
2. **If Normalized Against the Single-Stream $B=1$ Baseline**:
   - Relative to the **$B=1$ sequential stream** ($946.8\text{ tok/s}$):
     $B=2$ throughput is **$1.47\times$**, $B=4$ is **$2.84\times$**, and $B=8$ is **$4.73\times$**.
   - Relative to the **$B=1$ batched stream** ($736.4\text{ tok/s}$, which includes batch left-padding and tensor routing overhead):
     $B=2$ throughput is **$1.88\times$**, $B=4$ is **$3.66\times$**, and $B=8$ is **$6.09\times$**.

Both baselines are mathematically sound and are reported side-by-side in Section 3 to ensure total transparency.

---

## 2. Rigorous Metric & Timing Definitions

To eliminate ambiguity between throughput, prefill latency, and inter-token decode latency, each metric is defined with explicit timing boundaries:

```
Batch Execution Timeline (Total Latency: T_total)
|---------------------------------------------------------------------------------------|
| Prefill Pass (TTFT) |                   Decode Loop (T_decode)                        |
|   t_0 ----------> t_1|  t_1 -----------------------------------------------------> t_N |
|  Prompt tokenization |  Step 1       Step 2       ...                       Step S-1  |
|  & Prefill Forward   |  Decode       Decode                                 Decode    |
```

1. **Total Batch Completion Latency ($T_{\text{total}}$)**:
   $$T_{\text{total}} = t_{\text{end}} - t_{\text{start}}$$
   Measured via `time.perf_counter()` synchronized with `torch.cuda.synchronize()`. Spans input tokenization, left-padding tensor construction, prefill forward pass, all decode steps until `finished.all()`, and string token decoding.
2. **Time to First Token (TTFT / Prefill Latency)**:
   $$\text{TTFT} = t_{\text{prefill\_end}} - t_{\text{prefill\_start}}$$
   The wall-clock duration of the initial full-sequence forward pass over $P_{\max}$ prompt tokens producing the first token prediction logits. Measured in milliseconds.
3. **Inter-Token Latency (ITL / Decode Latency per Step)**:
   $$\text{ITL} = \frac{T_{\text{decode}}}{S - 1} = \frac{t_{\text{end}} - t_{\text{prefill\_end}}}{\max(N_1, \dots, N_B) - 1}$$
   The average wall-clock latency per incremental decoding step across all batch rows. Measured in milliseconds.
4. **Aggregate Output Throughput**:
   $$\text{Aggregate TPS} = \frac{\sum_{b=1}^B N_b}{T_{\text{total}}}$$
   The total number of newly generated valid tokens across all batch rows divided by total batch completion time.
5. **Per-Request Output Throughput**:
   $$\text{Per-Req TPS} = \frac{1}{B} \sum_{b=1}^B \frac{N_b}{T_{\text{total}}}$$
   The average generation rate experienced across individual requests during batch execution.
6. **Iso-Workload Speedup**:
   $$\text{Speedup}_{\text{iso}} = \frac{T_{\text{seq}}(B)}{T_{\text{batch}}(B)} = \frac{\text{Aggregate TPS}_{\text{batch}}(B)}{\text{Aggregate TPS}_{\text{seq}}(B)}$$
   The speedup achieved when serving the exact same $B$ requests batched in parallel versus one after another.
7. **Single-Stream Expansion Ratio**:
   $$\text{Expansion}_{\text{seq}(1)} = \frac{\text{Aggregate TPS}_{\text{batch}}(B)}{\text{Aggregate TPS}_{\text{seq}}(B=1)}, \quad \text{Expansion}_{\text{batch}(1)} = \frac{\text{Aggregate TPS}_{\text{batch}}(B)}{\text{Aggregate TPS}_{\text{batch}}(B=1)}$$

---

## 3. Audited Benchmark Results

All runs were performed with 2 warm-up iterations and 5 measured repetitions using deterministic greedy decoding (`temperature=0.0`) on CUDA. Peak memory was reset prior to each timed invocation.

### 3.1 Workload A: Homogeneous (Uniform Prompts ~32 tok -> 40 new tok)

| Cache Mode | Batch Size ($B$) | Batched TPS (tok/s) | Seq TPS (tok/s) | **Iso-Workload Speedup** | **Speedup vs Seq $B=1$** | **Speedup vs Batch $B=1$** | TTFT (ms) | ITL (ms) | Peak Alloc (MB) | Peak Res (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dynamic** | 1 | 715.6 | 934.3 | **0.77x** | 0.77x | 1.00x | 1.9 | 1.4 | 26.6 | 36.0 |
| | 2 | 1,473.3 | 1,011.9 | **1.46x** | 1.58x | 2.06x | 1.7 | 1.3 | 28.5 | 36.0 |
| | 4 | 2,616.2 | 1,001.6 | **2.61x** | 2.80x | 3.66x | 2.1 | 1.5 | 32.3 | 40.0 |
| | 8 | 4,369.5 | 998.7 | **4.37x** | 4.68x | 6.11x | 2.2 | 1.7 | 40.0 | 46.0 |
| **Static** | 1 | 736.4 | 946.8 | **0.78x** | 0.78x | 1.00x | 2.0 | 1.3 | 33.4 | 42.0 |
| | 2 | 1,387.7 | 959.9 | **1.45x** | 1.47x | 1.88x | 2.2 | 1.4 | 42.1 | 54.0 |
| | 4 | 2,693.2 | 1,009.4 | **2.67x** | 2.84x | 3.66x | 1.8 | 1.4 | 59.5 | 76.0 |
| | 8 | 4,482.3 | 1,009.7 | **4.44x** | 4.73x | 6.09x | 2.0 | 1.7 | 94.3 | 160.0 |
| **None** (uncached) | 1 | 710.5 | 991.4 | **0.72x** | 0.72x | 1.00x | 1.5 | 1.4 | 28.7 | 40.0 |
| | 2 | 1,324.4 | 1,001.2 | **1.32x** | 1.34x | 1.86x | 1.5 | 1.5 | 31.2 | 40.0 |
| | 4 | 2,419.8 | 1,017.7 | **2.38x** | 2.44x | 3.41x | 1.7 | 1.6 | 38.0 | 48.0 |
| | 8 | 3,803.0 | 1,024.5 | **3.71x** | 3.84x | 5.35x | 1.8 | 2.0 | 52.0 | 60.0 |

---

### 3.2 Workload B: Heterogeneous (Prompts 8–95 tok -> 40 new tok)

| Cache Mode | Batch Size ($B$) | Batched TPS (tok/s) | Seq TPS (tok/s) | **Iso-Workload Speedup** | **Speedup vs Seq $B=1$** | **Speedup vs Batch $B=1$** | TTFT (ms) | ITL (ms) | Peak Alloc (MB) | Peak Res (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dynamic** | 1 | 784.3 | 1,025.0 | **0.77x** | 0.77x | 1.00x | 1.4 | 1.3 | 25.5 | 36.0 |
| | 2 | 1,270.6 | 1,025.1 | **1.24x** | 1.24x | 1.62x | 1.6 | 1.3 | 27.9 | 36.0 |
| | 4 | 2,501.6 | 1,019.9 | **2.45x** | 2.44x | 3.19x | 2.0 | 1.4 | 37.9 | 50.0 |
| | 8 | 4,210.4 | 1,018.8 | **4.13x** | 4.11x | 5.37x | 3.8 | 1.7 | 68.0 | 80.0 |
| **Static** | 1 | 737.1 | 963.7 | **0.76x** | 0.76x | 1.00x | 1.8 | 1.3 | 32.8 | 42.0 |
| | 2 | 1,259.5 | 979.5 | **1.29x** | 1.31x | 1.71x | 1.7 | 1.3 | 41.7 | 54.0 |
| | 4 | 2,499.9 | 979.9 | **2.55x** | 2.59x | 3.39x | 1.9 | 1.4 | 62.5 | 76.0 |
| | 8 | 4,169.1 | 986.2 | **4.23x** | 4.33x | 5.66x | 3.9 | 1.7 | 109.5 | 160.0 |
| **None** (uncached) | 1 | 697.5 | 979.5 | **0.71x** | 0.71x | 1.00x | 1.7 | 1.4 | 26.3 | 38.0 |
| | 2 | 1,115.0 | 990.0 | **1.13x** | 1.14x | 1.60x | 1.8 | 1.5 | 30.2 | 40.0 |
| | 4 | 2,047.0 | 1,001.9 | **2.04x** | 2.09x | 2.93x | 2.0 | 1.8 | 46.2 | 58.0 |
| | 8 | 1,903.8 | 1,007.5 | **1.89x** | 1.94x | 2.73x | 3.2 | 4.0 | 94.3 | 120.0 |

---

### 3.3 Workload C: Early Completion (Prompts 8–95 tok -> 15–60 new tok)

| Cache Mode | Batch Size ($B$) | Batched TPS (tok/s) | Seq TPS (tok/s) | **Iso-Workload Speedup** | **Speedup vs Seq $B=1$** | **Speedup vs Batch $B=1$** | TTFT (ms) | ITL (ms) | Peak Alloc (MB) | Peak Res (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dynamic** | 1 | 706.3 | 918.3 | **0.77x** | 0.77x | 1.00x | 1.7 | 1.3 | 25.1 | 36.0 |
| | 2 | 1,109.8 | 973.5 | **1.14x** | 1.21x | 1.57x | 1.9 | 1.3 | 27.5 | 36.0 |
| | 4 | 1,706.7 | 1,002.3 | **1.70x** | 1.86x | 2.42x | 2.2 | 1.4 | 39.1 | 50.0 |
| | 8 | 2,966.2 | 1,004.9 | **2.95x** | 3.23x | 4.20x | 3.8 | 1.6 | 70.8 | 80.0 |
| **Static** | 1 | 695.4 | 881.8 | **0.79x** | 0.79x | 1.00x | 1.8 | 1.3 | 32.8 | 42.0 |
| | 2 | 1,104.0 | 947.0 | **1.17x** | 1.25x | 1.59x | 2.0 | 1.3 | 41.7 | 54.0 |
| | 4 | 1,720.3 | 994.3 | **1.73x** | 1.95x | 2.47x | 2.1 | 1.4 | 62.5 | 76.0 |
| | 8 | 2,928.8 | 981.5 | **2.98x** | 3.32x | 4.21x | 3.9 | 1.6 | 109.5 | 160.0 |
| **None** (uncached) | 1 | 667.1 | 949.1 | **0.70x** | 0.70x | 1.00x | 1.8 | 1.4 | 25.4 | 38.0 |
| | 2 | 986.5 | 965.5 | **1.02x** | 1.04x | 1.48x | 1.8 | 1.5 | 29.7 | 40.0 |
| | 4 | 1,345.0 | 994.9 | **1.35x** | 1.42x | 2.02x | 2.0 | 1.8 | 48.9 | 62.0 |
| | 8 | 1,166.3 | 1,009.2 | **1.16x** | 1.23x | 1.75x | 3.3 | 4.2 | 99.0 | 120.0 |

---

## 4. In-Depth Architectural Insights

### 4.1 The $B=1$ Batching Penalty ($0.77\times$)
Notice that when running a single request through `generate_batch()` ($B=1$), throughput is $\sim 736\text{ tok/s}$, whereas `generate()` achieves $\sim 946\text{ tok/s}$ ($0.78\times$ relative performance).
- **Cause**: `generate_batch()` executes defensive generalized serving logic:
  1. It builds and broadcasts a 4D `attn_mask` `[B, 1, 1, T]` into PyTorch SDPA on every step.
  2. It generates and tracks explicit per-token `position_ids` `[B, 1]`.
  3. It applies independent `torch.multinomial` sampling and per-request lifecycle checks (`finished[b]`).
  In contrast, single-request `generate()` passes `attn_mask=None` and `position_ids=None`, allowing PyTorch's C++ SDPA dispatch to select optimized FlashAttention kernel branches with scalar offset caching.
- **Serving Implication**: In a real inference server, batch sizes $> 1$ amortize this routing overhead immediately ($B=2$ yields $1,387\text{ tok/s}$).

### 4.2 Analysis of the Early-Completion Penalty (Execution Bubbles)
In Workload C ($B=8$), aggregate throughput drops from **$4,169\text{ tok/s}$** to **$2,928\text{ tok/s}$** (a **$29.7\%$ throughput degradation**).
- **Cause**: Static lockstep batching forces all rows to execute until the slowest request finishes ($60$ tokens).
  - Row 0 finished at step 15, generating 45 wasted "pad steps".
  - Row 1 finished at step 30, generating 30 wasted "pad steps".
  - Over steps 31–60, the GPU continued to compute key/value projections and attention for finished rows that were simply emitting pad tokens.
- **Architectural Motivation for Continuous Batching**:
  This $29.7\%$ drop is the exact theoretical motivation for **continuous batching** (iteration-level scheduling). With continuous batching, completed rows are immediately evicted and new incoming requests are dynamically inserted into the empty slots, restoring full compute density.

### 4.3 Why Uncached Inference Collapses at $B=8$
Without KV caching (`none`), every decode step recomputes the entire prefix sequence:
- At $B=1$, the context is short enough that recomputation overhead is low ($697\text{ tok/s}$).
- At $B=8$, the attention computation grows quadratically: $\mathcal{O}(B \times T^2)$.
- In Workload B, ITL ballooned from $1.7\text{ ms}$ (cached) to **$4.0\text{ ms}$** (uncached), and throughput degraded by **$54\%$** (from $4,169\text{ tok/s}$ down to $1,903\text{ tok/s}$).
- This confirms that KV caching is strictly mandatory for multi-request serving.

---

## 5. Summary of Correctness & Fairness Audits

1. **Prompt & Generation Invariance**: Verified that sequential and batched loops execute identical string prompts, target token limits, and random seeds.
2. **Deterministic Output Parity**: All 17 unit and integration tests in `tests/test_batched_inference.py` pass. Every generated token string under greedy batched execution matches sequential generation token-for-token.
3. **Hardware Isolation**: Each benchmark repetition executed `torch.cuda.empty_cache()`, `gc.collect()`, `torch.cuda.reset_peak_memory_stats()`, and explicit `torch.cuda.synchronize()` boundaries.
4. **Reproducibility**: Run:
   ```bash
   .venv/bin/python3 scripts/benchmark_batched_inference.py --device cuda --repetitions 5 --warmup 2
   ```
