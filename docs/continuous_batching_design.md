# TouchlineLM — Continuous Batching Architecture Design (Phase 3C)

## 1. Executive Summary & Objective

In Phase 3B, TouchlineLM implemented and benchmarked **static lockstep batching**, achieving a **$4.44\times$ throughput speedup** over sequential execution on the local NVIDIA GeForce RTX 4050 Laptop GPU ($4,482.3\text{ tok/s}$ at $B=8$).
However, the benchmark audit revealed an intrinsic serving bottleneck: **the early-completion execution bubble**.
When requests in a static batch have different prompt lengths, differing generation targets, or trigger early EOS termination (Workload C), finished requests must idle as padding tokens while active requests finish decoding. At $B=8$, this idle padding induced a **$29.7\%$ throughput degradation** ($4,169\text{ tok/s} \rightarrow 2,928\text{ tok/s}$).

The objective of **Phase 3C** is to design a minimal, production-grade **Continuous Batching Scheduler** (iteration-level scheduling) and a dedicated **Slot-Allocated KV Cache** for TouchlineLM ($4,198,912$ parameters, FP32, $n_{\text{layers}}=4, d_{\text{model}}=256, n_{\text{heads}}=4, d_{\text{head}}=64, \text{context}=1024$).

Continuous batching eliminates execution bubbles by:
1. Evaluating request state at **every individual autoregressive decode iteration**.
2. Evicting finished requests immediately upon emitting an EOS token or reaching `max_new_tokens`.
3. Admitting waiting requests from a FIFO queue into newly freed batch slots as soon as capacity opens, without waiting for the rest of the batch to finish.
4. Managing independent KV cache allocations and sequence lengths per active slot.

```
Static Lockstep Batching (Wasted Bubbles):
Req 0: [P][D][D][D][D][EOS] . . . . . . . . . . . . . . . . . . . (IDLE PADDING BUBBLE)
Req 1: [P][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D][EOS] (Longest Request)
Req 2: [P][D][D][D][D][D][D][D][EOS] . . . . . . . . . . . . . . . (IDLE PADDING BUBBLE)

Continuous Batching (Zero Bubbles):
Req 0: [P][D][D][D][D][EOS] | Req 3 Admitted: [P][D][D][D][D][D]...
Req 1: [P][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D][EOS] | Req 4 Admitted...
Req 2: [P][D][D][D][D][D][D][D][EOS] | Req 5 Admitted: [P][D][D]...
```

---

## 2. Memory Accounting: Exact Binary (MiB) vs. Decimal (MB)

A single full-context ($T=1024$) FP32 Key-Value cache for TouchlineLM occupies:
$$\text{KV Bytes per Token per Layer} = 2 \times n_{\text{heads}} \times d_{\text{head}} \times \text{sizeof}(\text{float32}) = 2 \times 4 \times 64 \times 4 = 2,048\text{ bytes (2 KiB)}$$
$$\text{KV Bytes per Token (all 4 layers)} = 4 \times 2,048 = 8,192\text{ bytes (8 KiB)}$$
$$\text{Full Context ($1024$ tokens) per Request} = 1024 \times 8,192\text{ bytes} = 8,388,608\text{ bytes} = \mathbf{8.00\text{ MiB}} \approx 8.39\text{ MB}$$

### Memory Footprint by Maximum Active Slot Capacity ($M$)

| Active Slot Capacity ($M$) | Total KV Cache (Bytes) | Exact Binary Size (**MiB**) | Decimal Size (**MB**) | % of 6,141 MiB RTX 4050 VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **$M = 1$** (Single request) | $8,388,608\text{ B}$ | **$8.00\text{ MiB}$** | $8.39\text{ MB}$ | $0.13\%$ |
| **$M = 2$** | $16,777,216\text{ B}$ | **$16.00\text{ MiB}$** | $16.78\text{ MB}$ | $0.26\%$ |
| **$M = 4$** | $33,554,432\text{ B}$ | **$32.00\text{ MiB}$** | $33.55\text{ MB}$ | $0.52\%$ |
| **$M = 8$** (Default Batch) | $67,108,864\text{ B}$ | **$64.00\text{ MiB}$** | $67.11\text{ MB}$ | $1.04\%$ |
| **$M = 16$** (High Throughput) | $134,217,728\text{ B}$ | **$128.00\text{ MiB}$** | $134.22\text{ MB}$ | $2.08\%$ |
| **$M = 32$** (Server Stress) | $268,435,456\text{ B}$ | **$256.00\text{ MiB}$** | $268.44\text{ MB}$ | $4.17\%$ |

### Scope of Memory Preallocation and Transient Activations
Preallocating the KV cache buffers upfront guarantees that **KV cache tensor allocations are completely eliminated** during the decode loop. It prevents memory fragmentation caused by repeated `torch.cat` reallocations across thousands of decoding steps.

However, preallocation **does not eliminate every allocation in the inference pipeline**. We must strictly distinguish between two operational regimes:
1. **Prompt Prefill Phase ($T = P$)**:
   - The prefill pass processes the full prompt sequence $P$ ($P \le 1024$).
   - Transient activations include full sequence embeddings $[1, P, d_{\text{model}}]$, intermediate LayerNorm activations, projected queries, keys, and values $[1, n_{\text{heads}}, P, d_{\text{head}}]$, attention score matrices of size $[1, n_{\text{heads}}, P, P]$ (or FlashAttention SRAM tiles), and MLP feedforward intermediate projections $[1, P, 4 \times d_{\text{model}}]$.
   - These activations scale with prompt length: $\mathcal{O}(P \times d_{\text{model}} + P^2)$.
2. **Single-Token Batched Decode Phase ($T_q = 1$)**:
   - In each decode iteration, the query length is exactly 1.
   - Transient activations include query projections $[B_{\text{act}}, n_{\text{heads}}, 1, d_{\text{head}}]$, newly projected single-token key/value slices $[B_{\text{act}}, n_{\text{heads}}, 1, d_{\text{head}}]$, attention dot products $[B_{\text{act}}, n_{\text{heads}}, 1, T_{\text{max\_active}}]$, and MLP hidden activations $[B_{\text{act}}, 1, 4 \times d_{\text{model}}]$.
   - These transient activations scale as $\mathcal{O}(B_{\text{act}} \times d_{\text{model}} + B_{\text{act}} \times n_{\text{heads}} \times T_{\text{max\_active}})$ and are immediately freed at the conclusion of each forward step.

---

## 3. Attention, Masking, and KV Cache Correctness Specification

In continuous batching, requests in different slots have **unequal, dynamically evolving sequence lengths**. Correctness cannot rely merely on zeroing past memory. Strict mathematical masking and explicit position metadata must guarantee isolation.

### 3.1 Exact RoPE Position IDs and Cache Indexing Conventions
To resolve all off-by-one ambiguities between prefill and decode phases, we define the exact token indexing semantics:
- Let an admitted request have a prompt of $P$ tokens: $t_0, t_1, \dots, t_{P-1}$.
- **Prefill Phase**:
  1. The prompt tokens $t_0 \dots t_{P-1}$ are fed into the model with position IDs $[0, 1, \dots, P-1]$.
  2. The K and V projections for all $P$ tokens are written to physical cache positions $0 \dots P-1$.
  3. The stored cache length for this request is initialized to $L = P$.
  4. The prefill logits at column $P-1$ produce the first generated token, denoted $g_0$.
- **First Decode Iteration (Processing $g_0$)**:
  1. Prior to this decode pass, the cache contains exactly $L$ valid tokens (indices $0 \dots L-1$).
  2. The input to the model is the previously generated token $g_0$.
  3. Because $L$ tokens already precede $g_0$ in the sequence, the RoPE position ID for $g_0$ is **$L$** (not $L - 1$).
  4. The newly projected K and V vectors for $g_0$ are written to physical cache index **$L$**.
  5. The attention mechanism attends across all valid positions **$0$ through $L$, inclusive** (a total of $L + 1$ keys).
  6. The stored cache length is incremented to $L \leftarrow L + 1$.
  7. The forward pass emits logits for $g_0$, which sample the next token $g_1$.
- **General Decode Step $k$**:
  1. The cache length before the forward pass is $L_i$.
  2. The incoming token uses RoPE position ID **$L_i$**.
  3. New K/V are written to physical cache index **$L_i$**.
  4. Valid attention span covers indices **$0 \le j \le L_i$** (total length $L_i + 1$).
  5. The cache length is advanced to $L_i \leftarrow L_i + 1$.
  6. Logits sample the next token.

### 3.2 Per-Slot Decode Attention Mask Semantics
Across the active batch of size $B_{\text{act}}$, let $T_{\text{max\_active}} = \max_{i \in \mathcal{A}} (L_i + 1)$ be the maximum valid key length after writing the new token at position $L_i$.
During single-token decoding ($T_q = 1$), SDPA executes with `is_causal=False` and an explicit boolean attention mask:
$$\mathbf{M}_{\text{decode}} \in \{\text{True}, \text{False}\}^{B_{\text{act}} \times 1 \times 1 \times T_{\text{max\_active}}}$$
For batch row $b$ corresponding to active slot $i$:
$$\mathbf{M}_{\text{decode}}[b, 0, 0, j] = \begin{cases} \text{True} & \text{if } 0 \le j \le L_i \quad (\text{past cached positions plus newly written position } L_i) \\ \text{False} & \text{if } L_i < j < T_{\text{max\_active}} \quad (\text{unwritten or stale padding positions}) \end{cases}$$

### 3.3 Unwritten KV Positions and Stale Data Defense
When Slot $i$ was previously occupied by Request A (which reached sequence length 80) and is reassigned to Request B (which has prompt length 20):
- Physical cache indices $20 \dots 79$ in Slot $i$ retain stale KV vectors from Request A.
- **Primary Correctness Guarantee**:
  The attention mask $\mathbf{M}_{\text{decode}}$ sets all positions $j > L_i$ to `False`. In PyTorch SDPA, positions with `False` are masked with $-\infty$ prior to softmax, producing an exact attention weight of $0.0$. Stale vectors are mathematically eliminated from the output.
- **Secondary Defense (Slot Slice Isolation)**:
  During prefill of Request B, new vectors overwrite positions $0 \dots P_B - 1$. During decoding, new vectors overwrite positions $L_i$ step by step. Narrowed buffer views (`narrow(3, 0, T_max_active)`) ensure attention never references indices beyond $T_{\text{max\_active}}$.

---

## 4. Model-Forward Interface & Compatibility Requirements

The continuous batching implementation must integrate cleanly with the existing model architecture.

### 4.1 Interface Architecture

```
+-------------------------------------------------------------------------------+
|                        ContinuousBatchScheduler                               |
| - Maintains: waiting_queue: Deque, active_slots: Dict[int, ActiveRequestState]|
| - Slices active token IDs: input_ids [B_act, 1]                               |
| - Constructs position_ids: [B_act, 1] with value L_i for each active row      |
| - Constructs decode attn_mask: [B_act, 1, 1, T_max_active]                    |
+-------------------------------------------------------------------------------+
                                        │
                                        │ calls model.forward(...)
                                        ▼
+-------------------------------------------------------------------------------+
|                       TouchlineNanoGPT (model.py)                             |
| - Input: idx [B_act, 1], position_ids [B_act, 1], attn_mask [...]            |
| - past_key_values: SlotAllocatedKVCache or SlotLayerCacheProxy                |
| - Propagates RoPE cos/sin computed from position_ids                          |
+-------------------------------------------------------------------------------+
                                        │
                                        │ passes layer_past to each block
                                        ▼
+-------------------------------------------------------------------------------+
|                     CausalSelfAttention (attention.py)                        |
| - Projects q, k, v for incoming step tokens [B_act, 1, d_model]               |
| - Calls: k_all, v_all = layer_past.update(k, v)                               |
| - Executes: F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=...)    |
+-------------------------------------------------------------------------------+
```

### 4.2 Exact Call Signatures

```python
class SlotLayerCacheProxy:
    """Proxy providing layer-specific update interface for CausalSelfAttention."""
    def __init__(self, cache_pool: "SlotAllocatedKVCache", layer_idx: int, active_slot_indices: List[int]):
        self.cache_pool = cache_pool
        self.layer_idx = layer_idx
        self.active_slot_indices = active_slot_indices

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k, v: [B_act, n_heads, 1, head_dim] newly projected tokens for active slots.
        Writes in-place into cache_pool at physical index L_i for each active slot.
        Returns active slice [B_act, n_heads, T_max_active, head_dim] spanning 0 .. L_i.
        """
        return self.cache_pool.update_step(self.layer_idx, self.active_slot_indices, k, v)
```

### 4.3 Backward Compatibility as a Strict Implementation Requirement
Backward compatibility is an **explicit implementation requirement to be demonstrated by regression tests**, rather than an assumed property:
1. All existing single-request inference tests (`tests/test_inference.py`) must pass with zero behavioral or numerical deviations.
2. All dynamic and static KV cache tests (`tests/test_kv_cache.py`) must pass.
3. All Phase 3B batched inference tests (`tests/test_batched_inference.py`) must pass.
4. Calling `model(idx, past_key_values=None)` or `model(idx, past_key_values=StaticKVCache)` must execute without code changes to their existing forward code branches.

---

## 5. Scheduler Semantics & Lifecycle State Machine

```
                              Client submit_request()
                                         │
                                         ▼
                     +───────────────────────────────────────+
                     │     FIFO Waiting Queue Check          │
                     │  len(queue) >= max_queue_capacity?    │
                     +───────────────────────────────────────+
                                    /         \
                             YES   /           \  NO
                                  ▼             ▼
                       Raise QueueFullError   Enqueue Request
                       (Reject fast)          Status: WAITING
                                                    │
                                                    │ Scheduler step() begins
                                                    ▼
                     +───────────────────────────────────────+
                     │       Capacity Admission Check        │
                     │    free_slots = M - len(active_slots) │
                     +───────────────────────────────────────+
                                    /         \
                       free_slots == 0         free_slots > 0
                             │                      │
                             │                      ▼
                             │            Pop next request from FIFO
                             │            Assign slot index s in [0..M-1]
                             │            Status: PREFILLING
                             │                      │
                             │                      ▼
                             │            Execute Isolated Prefill Pass
                             │            Write keys/values to Slot s at 0..P-1
                             │            Sample step-0 token g_0
                             │            Cache length set to L_s = P
                             │            Status: DECODING
                             │                      │
                             └──────────────┬───────┘
                                            │
                                            ▼
                     +───────────────────────────────────────+
                     │      Active Batch Iteration Step      │
                     │  1. Collect step tokens [B_act, 1]    │
                     │  2. Model forward (pos=L_s, mask=0..L)│
                     │  3. Sample next token per request     │
                     │  4. Update cache length: L_s <- L_s+1 │
                     +───────────────────────────────────────+
                                            │
                                            ▼
                     +───────────────────────────────────────+
                     │          Completion Evaluation        │
                     │  tok == EOS or count >= max_tokens?   │
                     +───────────────────────────────────────+
                                    /         \
                             YES   /           \  NO
                                  ▼             ▼
                           Status: COMPLETED   Remain in DECODING
                           Emit Output         (Ready for next iteration)
                           Release Slot s
                           (Available for next FIFO admission)
```

### 5.1 Exact Scheduling Semantics
1. **FIFO Queue & Capacity**:
   Requests are queued in strict arrival order. When `len(waiting_queue) >= max_queue_capacity`, `submit_request()` immediately raises `QueueFullError(f"Waiting queue capacity {max_queue_capacity} exceeded")`.
2. **Immediate Prefill & Joining**:
   When a slot is free, an admitted request executes its prefill pass immediately, writes its prompt keys/values into the assigned slot, samples its first generated token $g_0$, initializes $L_i = P$, and transitions to `DECODING`. It joins the active batch on the **very next decode iteration**. Existing decoding requests do not wait.
3. **Prefill Failure Isolation**:
   If an admitted request fails during prefill (e.g. context overflow $P + \text{max\_new\_tokens} > 1024$), the exception is caught immediately:
   - The request is marked `status="failed"` with the error description.
   - The assigned slot is released back to the free pool immediately.
   - Other active decoding requests continue running without disruption.
4. **Iteration Eviction**:
   As soon as a request encounters an EOS token or generates its configured `max_new_tokens`, it is marked `COMPLETED` and evicted from the active set during that exact iteration. The slot is returned to the free pool for immediate reuse.

---

## 6. Request-Level State & Isolation

Every request is tracked as an isolated `ActiveRequestState` instance:

```python
@dataclass
class ActiveRequestState:
    # 1. Identity and Specifications
    request_id: str
    prompt: str
    prompt_tokens: List[int]
    max_new_tokens: int
    min_tokens: int
    stop_at_eos: bool
    stop_token_ids: Set[int]

    # 2. Independent Sampling Configuration
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    generator: Optional[torch.Generator]  # Deterministic per-request RNG

    # 3. Slot and Execution State
    slot_idx: int                         # Assigned slot in SlotAllocatedKVCache
    current_pos: int                      # Current total sequence length (prompt + generated)
    generated_tokens: List[int]           # Accumulator of newly generated token IDs
    latest_token_id: int                  # Input token for next decode step
    status: RequestStatus                 # WAITING, PREFILLING, DECODING, COMPLETED, FAILED
    error_message: Optional[str] = None

    # 4. Latency & Telemetry Profiling
    arrival_timestamp: float              # Enqueue time (perf_counter)
    admission_timestamp: float            # Admission / prefill start time
    first_token_timestamp: float          # Time when step-0 token was produced
    completion_timestamp: float           # Eviction time
    decode_step_times: List[float]        # Per-token decode step durations
```

---

## 7. Comprehensive Benchmark Methodology

To rigorously validate continuous batching against static lockstep batching, the benchmark must process **equivalent workloads under identical conditions**.

### 7.1 Workload Configurations
1. **Workload A: Uniform Static Workload** (Baseline Validation):
   16 requests with uniform prompt lengths (32 tokens) and uniform targets (40 tokens).
2. **Workload B: Mixed Lengths & Early Completion** (Primary Stress Test):
   32 requests with diverse prompt lengths (8 to 95 tokens) and staggered generation targets ($15, 30, 45, 60, 90$ tokens) with EOS stopping enabled.
   - In static lockstep batching, groups of 8 requests run together; shorter requests idle until the 90-token request finishes.
   - In continuous batching, slots are immediately replenished as 15-token and 30-token requests finish.
3. **Workload C: Poisson Arrival Simulation**:
   Requests arrive according to a Poisson process with arrival rate $\lambda \in \{10, 25, 50\}\text{ req/sec}$ to measure queuing delay and tail latency ($p95$).

### 7.2 Metrics Reported
- **Aggregate Output Throughput**: $\sum \text{new\_tokens} / T_{\text{total}}$ (tok/s).
- **Per-Request Completion Latency**: Mean, median, and $p95$ end-to-end latency per request.
- **Queue Waiting Time**: Duration spent in the waiting queue before admission ($T_{\text{admit}} - T_{\text{enqueue}}$).
- **Time to First Token (TTFT)**:
  - **End-to-End TTFT**: Queue time + Prefill time ($T_{\text{first\_token}} - T_{\text{enqueue}}$).
  - **Model Prefill Latency**: Isolated prompt forward pass duration ($T_{\text{first\_token}} - T_{\text{admit}}$).
- **Inter-Token Latency (ITL)**: Mean and $p95$ duration per incremental decode step.
- **Bubble Elimination Ratio**:
  $$\text{Bubble Reduction} = 1 - \frac{\sum_{\text{steps}} (M - B_{\text{act}})}{M \times \text{Total Iterations}}$$
- **Peak CUDA Allocated & Reserved Memory** (MB and MiB).
- **Concurrency Tracking**: Active slot count and waiting queue size sampled at every iteration.

---

## 8. Summary of Final Technical Conventions

| Dimension | Final Technical Convention |
| :--- | :--- |
| **RoPE Position ID for Decode** | Exact value **$L_i$** (where $L_i$ is cache length prior to forward pass). |
| **Physical Write Index** | Write new key and value vectors at index **$L_i$**. |
| **Attention Mask Range** | Keys attend over **$0 \le j \le L_i$** (total length $L_i + 1$). Positions $j > L_i$ are masked to `False`. |
| **Cache Length Advance** | Increment $L_i \leftarrow L_i + 1$ immediately following the forward pass. |
| **SDPA Causality** | Prefill uses `is_causal=True`; single-token decode ($T_q=1$) uses `is_causal=False` with explicit boolean mask. |
| **Memory Classification** | Full context KV cache is exactly **$8.00\text{ MiB}$** ($8.39\text{ MB}$) per request; preallocation eliminates KV tensor reallocations during decode, while transient activations are bounded per phase. |
| **Backward Compatibility** | Explicit implementation requirement to be verified by the 85-test regression suite. |
