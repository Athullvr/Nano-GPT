# TouchlineLM Phase 3A — KV Cache Memory Profiling & Optimization Report

This document delivers the technical investigation, root-cause diagnosis, memory audit, context scaling experiments, and architectural analysis for the Key-Value (KV) cache memory footprint in **TouchlineLM** (`TouchlineNanoGPT`, 4,198,912 parameters, FP32, 4 layers, 4 heads, $d=256$, max context length 1,024).

---

## 1. Executive Summary & Root-Cause Diagnosis

### 1.1 The Anomaly
In Phase 2 benchmarking, dynamic KV-cached inference on an NVIDIA RTX 4050 GPU reported a peak VRAM allocation of **363.97 MB** during 200-token generation, whereas the uncached baseline reported **84.51 MB**. This contradicted theoretical expectations: the entire 1,024-token KV cache at FP32 is only **8.00 MB** ($4 \text{ layers} \times 2 \times 1 \text{ batch} \times 4 \text{ heads} \times 1024 \times 64 \times 4\text{ bytes}$).

### 1.2 The Root Cause: Autograd DAG Retention
A meticulous memory audit revealed that the benchmark loop in `scripts/benchmark_inference.py` executed `model(idx)` without wrapping the call in `torch.inference_mode()` (or `torch.no_grad()`). Although `model.eval()` was set, all model parameters still possessed `requires_grad=True`.

The structural divergence between cached and uncached generation produced the memory disparity:
1. **Uncached Baseline**: At each decoding step $t$, `logits, _ = model(idx)` was called. Because `logits` was reassigned each step without preserving references to prior steps, PyTorch's autograd engine automatically released the previous step's backward graph. Peak memory was bounded by a single step's forward graph (~84 MB).
2. **Dynamic KV Cache**: At each decoding step $t$, `past_key_values` from step $t-1$ was passed into step $t$, where `torch.cat((k_past, k), dim=2)` linked the new key to the historical tensor. Because `k_past` carried gradient tracking from step $t-1$, `torch.cat` constructed an **unbroken, 200-step computational DAG (`CatBackward0`) linking every decode step together**. This forced PyTorch to retain **all 200 steps of intermediate forward activations across all 4 Transformer layers simultaneously in GPU memory**.

```
Autograd Graph Chaining Bug (Without inference_mode):

Step 1: [k_1] ---> (CatBackward0) ---> [k_1, k_2]
                     ^                      |
Step 2: [k_2] -------|                      v
                                       (CatBackward0) ---> [k_1, k_2, k_3] ... (Retains all activations!)
```

### 1.3 The Empirical Proof
Running the identical 173-prompt $\to$ 100-step generation on the same RTX 4050 GPU:

| Execution Mode | Autograd Graph Status | Peak Allocated VRAM | Reduction |
| :--- | :--- | :---: | :---: |
| **Without Guard** (Original Benchmark) | 100-step backward DAG retained | **170.67 MB** | Baseline |
| **With `torch.inference_mode()`** | Autograd graph disabled | **30.08 MB** | **-140.60 MB (-82.4%)** |

For 200 steps, peak allocated memory dropped from **366.08 MB down to 35.43 MB** (an **89.5% memory reduction**).

> [!IMPORTANT]
> Under proper `torch.inference_mode()`, dynamic KV-cached inference does **not** consume more memory than uncached inference. At full context ($T = 1,024$), dynamic KV caching consumes **40.66 MB**, which is **16.96 MB LESS** than the uncached baseline (**57.62 MB**).

---

## 2. Granular Memory Lifecycle Audit

Using `torch.cuda.memory_allocated()` (exact tensor memory) and `torch.cuda.memory_reserved()` (PyTorch caching allocator pool) under `torch.inference_mode()`:

| Lifecycle Phase | Allocated Memory | Reserved Memory | Underlying Tensor Contents |
| :--- | :---: | :---: | :--- |
| **1. PyTorch CUDA Init** | 0.00 MB | 0.00 MB | Uninitialized device state |
| **2. Model Load & Weights** | 16.52 MB | 34.00 MB | 4,198,912 parameters (FP32) + LayerNorm/Embedding buffers |
| **3. Prefill (173 tokens)** | 20.58 MB | 54.00 MB | Model weights + $Q, K, V$ projections + SDPA attention scratchpad |
| **4. Decode Step 100** | 18.25 MB | 44.00 MB | Model weights + active KV cache (1.07 MB) + 1-token decode activation |
| **5. Decode Step 200** | 19.44 MB | 44.00 MB | Model weights + active KV cache (2.14 MB) + 1-token decode activation |
| **Peak During Generation** | **35.43 MB** | **62.00 MB** | Maximum transient allocation during initial prefill projection |

### 2.1 Allocated vs. Reserved Memory
- **Allocated Memory**: Strictly represents memory actively holding PyTorch tensor data. Peak allocation during KV-cached generation is ~35 MB, of which 16.5 MB is the model weights, ~3 MB is the KV cache, and ~15 MB is transient activation workspace during prefill.
- **Reserved Memory**: The PyTorch caching allocator requests memory from the CUDA driver in 2 MB / 20 MB chunks to avoid calling expensive `cudaMalloc`/`cudaFree` on every tensor creation. While reserved memory reaches 52–62 MB, this memory is available for immediate re-use by subsequent operations.

---

## 3. Cache Allocation Analysis: Dynamic (`torch.cat`) vs. Preallocated

### 3.1 Overhead of Repeated `torch.cat`
In dynamic caching, each single-token decode step concatenates `k_past` of length $t$ with new key `k` of length 1:
```python
k = torch.cat((k_past, k), dim=2)
v = torch.cat((v_past, v), dim=2)
```
- **Copy Volume**: To generate $N$ tokens after a prompt of length $P$, total elements copied per tensor is:
  $$\sum_{t=P}^{P+N-1} t = \mathcal{O}(N \cdot P + N^2)$$
- **Allocator Requests**: Generating 1,000 tokens across 4 layers generates $2 \times 4 \times 1,000 = 8,000$ tensor allocations and deallocations.
- **Transient Memory**: During concatenation, both the old tensor (size $t$) and new tensor (size $t+1$) exist simultaneously in memory until the old tensor is released.

### 3.2 Preallocated Static KV Cache (`StaticKVCache`)
To evaluate zero-allocation caching, [`src/inference/kv_cache.py`](file:///home/athulvr/Projects/Nano Gpt/src/inference/kv_cache.py) implements `StaticKVCache`:
- Preallocates maximum-context buffers:
  `k_cache: [B, n_heads, max_seq_len, head_dim]` (8.00 MB total for 4 layers).
- Newly projected keys/values are copied in-place via `.narrow(2, pos, 1).copy_(k)`.
- Attention views are sliced dynamically with zero copies: `k[:, :, :pos, :]`.
- **Zero reallocations** occur during the entire decoding loop.

### 3.3 Microbenchmark: Tensor Update Mechanisms (500 Steps)

| Mechanism | Implementation | Total Update Time | Relative Speed | Allocations |
| :--- | :--- | :---: | :---: | :---: |
| **Dynamic `torch.cat`** | `torch.cat((k_past, new_k), dim=2)` | 8.02 ms | 1.00× | 500 new buffers |
| **Python Slice Assign** | `k_pre[:, :, pos:pos+1, :] = new_k` | 14.81 ms | 0.54× (slower) | 0 |
| **In-place `narrow.copy_`** | `k_pre.narrow(2, pos, 1).copy_(new_k)` | **4.99 ms** | **1.61× (faster)** | **0** |

`narrow().copy_()` avoids Python strided slice indexing overhead, executing in-place tensor copies 3× faster than Python slice assignment and 38% faster than `torch.cat`.

---

## 4. Context Scaling Experiment (32 to 1024 Tokens)

Measured using [`scripts/profile_kv_cache_memory.py`](file:///home/athulvr/Projects/Nano Gpt/scripts/profile_kv_cache_memory.py) on NVIDIA RTX 4050 Laptop GPU (Prompt = 16 tokens):

| Total Context | Generated Tokens | Theoretical KV Cache | Strategy | Peak Allocated Memory | Peak Reserved Memory | Decode Latency (ms/tok) | Total Latency (ms) | Throughput (tok/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32 tokens** | 16 | 0.250 MB | **Uncached** | 25.63 MB | 36.00 MB | 0.85 ms | 14.2 ms | 1,122.9 tok/s |
| | | | **Dynamic KV** | **25.17 MB** | **34.00 MB** | **0.83 ms** | **13.9 ms** | **1,150.9 tok/s** |
| | | | **Static KV** | 32.91 MB | 42.00 MB | 0.85 ms | 15.7 ms | 1,019.2 tok/s |
| **128 tokens** | 112 | 1.000 MB | **Uncached** | 28.80 MB | 38.00 MB | 0.83 ms | 95.4 ms | 1,174.1 tok/s |
| | | | **Dynamic KV** | **26.65 MB** | **36.00 MB** | **0.81 ms** | **92.4 ms** | **1,212.3 tok/s** |
| | | | **Static KV** | 32.91 MB | 42.00 MB | **0.81 ms** | 92.8 ms | 1,206.4 tok/s |
| **256 tokens** | 240 | 2.000 MB | **Uncached** | 33.37 MB | 58.00 MB | 0.84 ms | 206.2 ms | 1,163.9 tok/s |
| | | | **Dynamic KV** | **28.65 MB** | **38.00 MB** | **0.79 ms** | **193.4 ms** | **1,240.9 tok/s** |
| | | | **Static KV** | 32.91 MB | 42.00 MB | **0.79 ms** | 194.1 ms | 1,236.6 tok/s |
| **512 tokens** | 496 | 4.000 MB | **Uncached** | 41.18 MB | 78.00 MB | 1.11 ms | 557.9 ms | 889.0 tok/s |
| | | | **Dynamic KV** | **32.66 MB** | 44.00 MB | **0.79 ms** | 399.7 ms | 1,240.9 tok/s |
| | | | **Static KV** | 32.91 MB | **42.00 MB** | **0.79 ms** | **399.0 ms** | **1,243.3 tok/s** |
| **1024 tokens** | 1,008 | 8.000 MB | **Uncached** | 57.62 MB | 84.00 MB | 1.88 ms | 1,902.5 ms | 529.8 tok/s |
| | | | **Dynamic KV** | 40.66 MB | 52.00 MB | 0.81 ms | 831.0 ms | 1,213.0 tok/s |
| | | | **Static KV** | **32.91 MB** | **42.00 MB** | **0.80 ms** | **823.4 ms** | **1,224.2 tok/s** |

---

## 5. Architectural Evaluation: Does Evidence Support Preallocation?

```
Peak Allocated Memory Scaling Across Context Length:

VRAM (MB)
  60 |                                                  * (Uncached: 57.62 MB)
  50 |
  40 |                                            * (Dynamic: 40.66 MB)
  30 |   *========================================* (Static: 32.91 MB constant)
  20 |   * (Dynamic: 25.17 MB)
  10 |
   0 +--------------------------------------------------
         32           128          256          512         1024 Context (Tokens)
```

### 5.1 When Dynamic Cache is Superior
1. **Short-to-Medium Contexts ($T \le 512$)**:
   Dynamic cache allocates memory on-demand. At 32 tokens, it allocates only 25.17 MB vs. 32.91 MB for Static Cache (a **7.74 MB saving**).
2. **Variable Sequence Lengths**:
   Does not require committing to a maximum context length up front.
3. **Execution Throughput**:
   On a 4.2M parameter model with batch size 1, dynamic caching achieves **1,213–1,240 tokens/second**, matching static caching throughput. Because single-token key/value updates are only 1 KB ($1 \times 4 \times 1 \times 64 \times 4\text{ bytes}$), GPU memory bandwidth easily handles `torch.cat`.

### 5.2 When Static Preallocated Cache is Superior
1. **Full Context Window ($T \to 1,024$)**:
   Static cache strictly bounds memory allocation at **32.91 MB constant**, saving 7.75 MB over Dynamic Cache (40.66 MB) and 24.71 MB over Uncached (57.62 MB).
2. **Deterministic, Zero-Allocation Guarantees**:
   Zero `cudaMalloc` calls during decoding, eliminating allocator fragmentation.
3. **Serving & Graph Capture**:
   Essential for CUDA Graph capture (`torch.cuda.make_graphed_callables`), TensorRT-LLM, or C++ LibTorch deployments where dynamic buffer reshaping invalidates execution graphs.

### 5.3 Recommendation
- **Default for Local Interactive CLI**: Retain **Dynamic KV Cache** (`cache_type="dynamic"`) as the primary path for flexible, on-demand memory scaling.
- **Production Option**: Provide **Static Preallocated Cache** (`cache_type="static"`) for bounded memory deployments and long-sequence batch serving.

---

## 6. Correctness & Test Verification

All 68 tests pass across the repository:
```
============================== 68 passed in 9.52s ==============================
```

### Verified Test Properties:
1. **Logit Equivalence**: Next-token logits generated by `StaticKVCache` match `DynamicKVCache` within `torch.allclose(atol=1e-5)` across all decoding steps.
2. **Greedy Sequence Invariance**: Greedy decoding ($\text{temperature} = 0.0$) yields **100% identical token strings** across Uncached, Dynamic Cache, and Static Cache paths.
3. **Capacity Boundaries**: `StaticKVCache` gracefully enforces context boundaries at $T = 1,024$.
4. **Buffer Reset**: `StaticKVCache.reset()` allows re-using preallocated buffers for multiple prompts with zero memory leaks.
