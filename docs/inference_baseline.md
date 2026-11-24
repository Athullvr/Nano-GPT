# TouchlineLM — Inference Baseline Audit & Performance Profiling

**Date**: 2026-11-25  
**Model**: `TouchlineNanoGPT` (4,198,912 parameters, 4 layers, 4 heads, $d=256$, context length 1024, vocab 4096)  
**Checkpoint**: [`checkpoints/final_model.pt`](file:///home/athulvr/Projects/Nano Gpt/checkpoints/final_model.pt)  
**Hardware Platforms**:
- **CPU**: AMD Ryzen / Intel x86_64 CPU (Linux 6.18, PyTorch 2.15.0+cu132)
- **CUDA GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB dedicated GDDR6 VRAM)

---

## 1. Executive Summary & Purpose

Before introducing inference-time optimizations such as key-value (KV) caching, quantization, or kernel fusion, we must establish a rigorous, mathematically sound, and empirically verified baseline.

This report audits the execution path of the current TouchlineLM generation loop, analyzes the exact computational costs of autoregressive decoding without state caching, verifies that RoPE and the Transformer architecture can support KV caching without changing model weights or checkpoint compatibility, and presents empirical benchmark measurements collected using a standardized timing methodology.

---

## 2. Audit of Existing Generation & Forward Execution Path

### 2.1 Existing Execution Flow in `src/inference/engine.py`

In [`src/inference/engine.py`](file:///home/athulvr/Projects/Nano Gpt/src/inference/engine.py#L196-L245), the generation loop currently operates as follows:

```python
for _ in range(max_new_tokens):
    # 1. Slice context to window
    idx_cond = idx[:, -self.model.config.max_seq_len:]
    
    # 2. Complete model forward pass on the entire sequence
    logits, _ = self.model(idx_cond)
    
    # 3. Discard all positions except the last
    logits_step = logits[:, -1, :].clone()
    
    # 4. Sample next token
    idx_next = sample(logits_step, temperature, top_k, top_p)
    
    # 5. Append new token to sequence
    idx = torch.cat((idx, idx_next), dim=1)
```

Inside [`TouchlineNanoGPT.forward()`](file:///home/athulvr/Projects/Nano Gpt/src/model/model.py#L102-L137):
1. **Token Embeddings**: Projects all $T$ tokens from discrete token IDs into dense representations: $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$.
2. **RoPE Computation**: Calls `self.rope(x, seq_len=T)`, slicing $\cos$ and $\sin$ tables for all positions $0, 1, \dots, T-1$.
3. **Layer Forward Passes** (Across all 4 Transformer blocks):
   - Computes linear projections for all $T$ positions: $Q = X W_q$, $K = X W_k$, $V = X W_v \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$.
   - Rotates all $Q$ and $K$ vectors across positions $0 \dots T-1$ via `apply_rotary_emb`.
   - Computes full causal Scaled Dot-Product Attention:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M_{\text{causal}}\right) V$$
     This computes a complete $T \times T$ lower-triangular attention score matrix in each head.
   - Passes all $T$ token representations through FeedForward MLPs ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
4. **Final LayerNorm & LM Head**: Applies LayerNorm to all $T$ tokens, then projects all $T$ hidden vectors through `lm_head` ($W_{\text{tied}}^T$) to yield logits of shape $[B, T, V]$.
5. **Logit Truncation**: Positions $0 \dots T-2$ are immediately discarded; only `logits[:, -1, :]` is retained.

---

### 2.2 Proof of Full-Sequence Recomputation ($\mathcal{O}(T^2)$ Complexity)

Let prompt length be $P$, and let the number of generated tokens be $N$.

At decoding step $t \in \{0, 1, \dots, N-1\}$, the input sequence length is $T_t = P + t$:
- At step $0$ (Prefill): The model processes $P$ tokens.
- At step $1$: The model re-processes $P + 1$ tokens.
- At step $2$: The model re-processes $P + 2$ tokens.
- At step $N-1$: The model re-processes $P + N - 1$ tokens.

#### Cumulative Computational Cost:
$$\text{Total Tokens Processed} = \sum_{t=0}^{N-1} (P + t) = N \cdot P + \frac{N(N - 1)}{2} = \mathcal{O}(N \cdot P + N^2)$$

For example, generating $N=200$ tokens from a prompt of $P=173$ tokens:
$$\text{Tokens Processed} = 200 \times 173 + \frac{200 \times 199}{2} = 34,600 + 19,900 = 54,500\text{ token forward passes}$$

With per-layer KV caching:
- Step 0 processes $P$ tokens once to populate the cache.
- Steps $1 \dots N-1$ process strictly **1 token each**:
$$\text{Tokens Processed (with KV Cache)} = P + (N - 1) = 173 + 199 = 372\text{ token forward passes}$$

**Ratio**: The non-KV-cached baseline performs **$146.5\times$ more token evaluations** in the Transformer layers than a KV-cached implementation for this prompt.

---

## 3. Audit of Benchmark Timing Methodology

In naive benchmarking, end-to-end timing often introduces substantial confounding errors:

1. **Conflating TTFT and TPOT**:
   - **Time-To-First-Token (TTFT)** measures prompt prefill latency. Prefill is compute-bound, performing large matrix multiplications across all $P$ prompt tokens in parallel.
   - **Time-Per-Output-Token (TPOT)** measures autoregressive decode latency. Decode is memory-bandwidth-bound, requiring repeated memory transfers of weights for each single token.
   - Combining these into a single "tokens per second" metric obscures prompt processing bottlenecks and misrepresents decode scaling.
2. **Model Loading Contamination**:
   - Loading includes disk read, torch deserialization (`torch.load`), state dict validation, and GPU VRAM allocation. If included in generation throughput, short generation runs appear artificially slow.
3. **Asynchronous CUDA Execution**:
   - CUDA kernel launches in PyTorch are non-blocking. Measuring latency with `time.perf_counter()` without calling `torch.cuda.synchronize()` before and after each phase records only kernel queueing times rather than actual hardware execution.
4. **Kernel Warm-up & Variance**:
   - Initial forward passes trigger PyTorch allocator initialization, cuBLAS autotuning, and CUDA context overhead. Profiling without dedicated warm-up iterations inflates measurements for the first query.
5. **Memory Telemetry**:
   - Without tracking `torch.cuda.max_memory_allocated()`, activation memory inflation from quadratic sequence recomputation is hidden.

### Standardized Profiling Protocol Implemented in `scripts/benchmark_inference.py`:
- **Model Loading**: Measured independently across 5 fresh loads from disk.
- **Warm-up**: 3 full warm-up generation cycles executed prior to data collection.
- **Explicit Synchronization**: `torch.cuda.synchronize()` executed before and after prefill and decode sections.
- **Statistical Aggregation**: 5 measured repetitions per prompt; reports Median, Mean, Standard Deviation, Min, and Max.
- **Memory Tracking**: Peak VRAM recorded via `torch.cuda.max_memory_allocated()`.

---

## 4. Empirical Baseline Benchmark Results

Measurements collected using [`scripts/benchmark_inference.py`](file:///home/athulvr/Projects/Nano Gpt/scripts/benchmark_inference.py) on the final trained checkpoint ([`checkpoints/final_model.pt`](file:///home/athulvr/Projects/Nano Gpt/checkpoints/final_model.pt)).

### 4.1 Model Loading Latency (Isolated)

| Platform | Device | Repetitions | Median Loading Time | Mean $\pm$ Std | Min / Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU** | x86_64 | 5 | **53.34 ms** | $55.49 \pm 7.38\text{ ms}$ | $49.66\text{ ms} / 67.42\text{ ms}$ |
| **CUDA** | RTX 4050 | 5 | **45.34 ms** | $45.80 \pm 0.83\text{ ms}$ | $44.94\text{ ms} / 46.75\text{ ms}$ |

*Finding*: Checkpoint deserialization and model initialization require under 55 ms on both devices.

---

### 4.2 Generation Benchmark: CPU (Baseline Without KV Cache)

Tested across three fixed prompts with 5 repetitions each:
- **Short**: Match Narrative ($P=32$ prompt tokens, $N=100$ generated tokens)
- **Medium**: Player Dossier ($P=39$ prompt tokens, $N=150$ generated tokens)
- **Long**: Tactical Macro Analysis ($P=173$ prompt tokens, $N=200$ generated tokens)

| Prompt Category | Prompt $P$ | New $N$ | Prefill (TTFT) Median $\pm$ Std | Decode Per Token Median $\pm$ Std | Total Generation Median $\pm$ Std | Throughput (tok/s) Median $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (Match Narrative)** | 32 | 100 | $4.63 \pm 0.94\text{ ms}$ | $6.25 \pm 0.91\text{ ms}$ | $623.30 \pm 91.32\text{ ms}$ | **$160.4 \pm 30.6\text{ tok/s}$** |
| **Medium (Player Dossier)** | 39 | 150 | $3.45 \pm 0.28\text{ ms}$ | $5.63 \pm 0.35\text{ ms}$ | $842.26 \pm 52.84\text{ ms}$ | **$178.1 \pm 10.4\text{ tok/s}$** |
| **Long (Tactical Macro)** | 173 | 200 | $7.94 \pm 2.01\text{ ms}$ | $11.28 \pm 0.40\text{ ms}$ | $2,251.66 \pm 81.93\text{ ms}$ | **$88.8 \pm 3.0\text{ tok/s}$** |

#### Critical Finding on CPU:
On the long prompt ($P=173, N=200$), decode latency per token increases from **$5.63\text{ ms}$ up to $11.28\text{ ms}$** (+100.4% increase per token), cutting throughput in half from **178 tok/s down to 88.8 tok/s**. This directly demonstrates the computational burden of recomputing self-attention over the expanding 373-token sequence at every step.

---

### 4.3 Generation Benchmark: CUDA (Baseline Without KV Cache)

Tested on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) with 5 repetitions:

| Prompt Category | Prompt $P$ | New $N$ | Prefill (TTFT) Median $\pm$ Std | Decode Per Token Median $\pm$ Std | Total Generation Median $\pm$ Std | Throughput (tok/s) Median $\pm$ Std | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (Match Narrative)** | 32 | 100 | $1.48 \pm 0.07\text{ ms}$ | $1.48 \pm 0.03\text{ ms}$ | $148.31 \pm 2.69\text{ ms}$ | **$674.3 \pm 11.8\text{ tok/s}$** | 45.75 MB |
| **Medium (Player Dossier)** | 39 | 150 | $1.51 \pm 0.12\text{ ms}$ | $1.49 \pm 0.02\text{ ms}$ | $223.01 \pm 2.96\text{ ms}$ | **$672.6 \pm 8.8\text{ tok/s}$** | 54.87 MB |
| **Long (Tactical Macro)** | 173 | 200 | $1.48 \pm 0.09\text{ ms}$ | $1.59 \pm 0.02\text{ ms}$ | $318.63 \pm 2.94\text{ ms}$ | **$627.7 \pm 5.7\text{ tok/s}$** | 84.51 MB |

#### Critical Findings on CUDA:
1. **GPU Saturation Masking**: Because the model is compact (4.2M parameters, batch size 1), GPU compute cores are underutilized at small sequence lengths ($T < 100$), so quadratic recomputation latency appears muted on GPU compared to CPU ($1.48\text{ ms}$ vs $1.59\text{ ms}$ per token).
2. **Peak Memory Growth**: Peak VRAM increases from **45.75 MB to 84.51 MB** (+84.7%) because the model allocates activation memory for $T \times T$ attention maps and full intermediate tensors across all 4 layers at every step.

---

## 5. Architectural Feasibility for Per-Layer KV Caching

Can TouchlineLM support KV caching without changing model weights or invalidating the trained checkpoint?

### 5.1 Weight Invariance
KV caching modifies **only the runtime activation flow** in the attention layers. No new weights are introduced, and no existing weight matrices change shape. The checkpoint [`checkpoints/final_model.pt`](file:///home/athulvr/Projects/Nano Gpt/checkpoints/final_model.pt) will load into a KV-cached model with `strict=True` with zero parameter alterations.

### 5.2 Rotary Position Embedding (RoPE) Compatibility
In [`src/model/rope.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/rope.py), rotary embeddings are defined as:
$$R(x, m) = x \odot \cos(m \theta) + \text{rotate\_half}(x) \odot \sin(m \theta)$$

The key mathematical property of RoPE is that the attention score between position $m$ and position $n$ depends **only on their relative distance $m - n$**:
$$\langle R(q, m), R(k, n) \rangle = q^T R_{n-m} k$$

In autoregressive decoding:
1. All past keys $k_0, k_1, \dots, k_{m-1}$ were already rotated with their respective positions $0, 1, \dots, m-1$ and can be cached **in their rotated form** ($k_{\text{rot}}$).
2. When a new single token arrives at position $m$:
   - The query $q_m$ is rotated using position $m$: $q_{\text{rot}, m} = R(q_m, m)$.
   - The key $k_m$ is rotated using position $m$: $k_{\text{rot}, m} = R(k_m, m)$.
   - The new rotated key is concatenated to the cached keys:
     $$K_{\text{all}} = [K_{\text{cache}}, k_{\text{rot}, m}] \in \mathbb{R}^{B \times H \times (m+1) \times d_{\text{head}}}$$
     $$V_{\text{all}} = [V_{\text{cache}}, v_m] \in \mathbb{R}^{B \times H \times (m+1) \times d_{\text{head}}}$$
3. The attention computation $q_{\text{rot}, m} K_{\text{all}}^T$ evaluates the exact same dot products as the full sequence computation:
   $$\langle R(q_m, m), R(k_j, j) \rangle \quad \forall j \le m$$

*Conclusion*: RoPE is 100% mathematically compatible with KV caching. The only requirement is that `RotaryEmbedding` must support slicing $\cos$ and $\sin$ tables for an arbitrary position offset ($m$).

### 5.3 PyTorch Scaled Dot-Product Attention (SDPA) Compatibility
In [`src/model/attention.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/attention.py#L76-L88):
- During full sequence prefill: $Q \in \mathbb{R}^{B \times H \times P \times D}$, $K \in \mathbb{R}^{B \times H \times P \times D}$, $V \in \mathbb{R}^{B \times H \times P \times D}$, using `is_causal=True`.
- During single-token decode with KV cache:
  $$Q \in \mathbb{R}^{B \times H \times 1 \times D}$$
  $$K_{\text{cached}} \in \mathbb{R}^{B \times H \times (m+1) \times D}$$
  $$V_{\text{cached}} \in \mathbb{R}^{B \times H \times (m+1) \times D}$$
  Because $Q$ has length 1 and is located at position $m$, it is allowed to attend to all keys $j \le m$. No lower-triangular causal mask is required!
  Calling PyTorch's `F.scaled_dot_product_attention(q, k_cached, v_cached, is_causal=False)` works natively and efficiently without custom CUDA kernels.

---

## 6. Blueprint of Changes Required to Implement KV Caching

When Phase 2 of inference optimization begins, the following non-breaking changes will be required:

| Module | Required Change | Backward Compatibility Guarantee |
| :--- | :--- | :--- |
| [`src/model/rope.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/rope.py) | Add `position_offset: int = 0` to `RotaryEmbedding.forward()` to slice $\cos/\sin$ at `[pos : pos + seq_len]`. | Default `position_offset=0` preserves existing behavior. |
| [`src/model/attention.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/attention.py) | Accept optional `past_key_value: Optional[Tuple[Tensor, Tensor]] = None`. If provided, concatenate new $K, V$ to past keys/values and return updated `(k_cached, v_cached)`. | If `past_key_value=None`, executes standard causal attention. |
| [`src/model/block.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/block.py) | Accept and pass `past_key_value` into `self.attn`, returning updated tuple along with block output. | Existing signature preserved via optional arguments. |
| [`src/model/model.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/model.py) | Add `past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None` and `use_cache: bool = False` to `forward()`. Return `(logits, past_key_values)`. | When `use_cache=False`, forward pass returns `(logits, loss)` as before. |
| [`src/inference/engine.py`](file:///home/athulvr/Projects/Nano Gpt/src/inference/engine.py) | Implement dual-phase generation loop: (1) Prefill pass over prompt with `use_cache=True` storing initial KV cache; (2) Autoregressive loop passing single token $x_t \in \mathbb{R}^{B \times 1}$ with updated `past_key_values`. | Existing CLI flags and public API signatures remain identical. |

---

## 7. Baseline Summary & Comparison Checklist

| Performance Metric | Non-KV Baseline (CPU) | Non-KV Baseline (CUDA) | Projected KV-Cached Target |
| :--- | :---: | :---: | :---: |
| **Model Loading** | 53.3 ms | 45.3 ms | 45.3 ms (Unchanged) |
| **Short (100 tok) Decode/Tok** | 6.25 ms/tok | 1.48 ms/tok | ~1.10–1.20 ms/tok (CUDA) |
| **Long (200 tok) Decode/Tok** | **11.28 ms/tok** | **1.59 ms/tok** | **~1.10–1.20 ms/tok** (Flat $\mathcal{O}(1)$) |
| **Long Prompt Throughput** | 88.8 tok/s | 627.7 tok/s | Projected **>800 tok/s** (CUDA) |
| **Computational Complexity** | $\mathcal{O}(N \cdot P + N^2)$ | $\mathcal{O}(N \cdot P + N^2)$ | **$\mathcal{O}(P + N)$** (Linear) |
| **Peak CUDA VRAM (Long)** | — | **84.51 MB** | **~25–35 MB** |

The baseline is established, reproducible via `scripts/benchmark_inference.py`, and recorded in `data/intermediate/inference_baseline_metrics.json`.
