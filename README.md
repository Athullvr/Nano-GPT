# TouchlineLM: Football Intelligence NanoGPT

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 103 Passed](https://img.shields.io/badge/tests-103%20passed-brightgreen.svg)]()
[![Parameters](https://img.shields.io/badge/Parameters-4.2M-orange.svg)]()
[![Inference Speed](https://img.shields.io/badge/Throughput-4%2C482%20tok%2Fs-blueviolet.svg)]()

**TouchlineLM** is an experimental, domain-specific small decoder-only Transformer (NanoGPT) and high-throughput inference serving engine engineered to model football intelligence, tactical patterns, and advanced event-level analytics from scratch.

From raw event graphs and custom Byte-Level BPE tokenization to local pre-training, dynamic/static KV caching, batched inference, and iteration-level continuous batch scheduling.

---

## 1. Project Vision

Traditional sports language applications often regurgitate superficial scorelines. **TouchlineLM** is designed to explore how an autoregressive language model can internalize the mathematical and tactical structure of modern football analytics:

- **Defensive Micro-Actions**: Interceptions, ground duels, clearances, ball recoveries, and high-pressing triggers.
- **Ball Progression**: Progressive passes, progressive carries, and zone entries.
- **Shot Creation & Finishing**: Expected Goals (xG), Expected Assists (xA), shot techniques, and freeze-frame spatial contexts.
- **Tactical Alignments**: Formations (e.g., 4-3-3, 4-2-3-1, 3-5-2), tactical substitutions, and spatial channel dynamics.
- **Inference Engineering**: Exploring modern serving optimizations (dynamic KV caching, preallocated zero-fragmentation static memory, lockstep batching with left-padding, and continuous batch scheduling).

---

## 2. Key Achievements & Milestones

### Pre-Training & Model Architecture
- **Trained From Scratch**: 4.2M parameter custom decoder-only Transformer ($n_{\text{layers}}=4, d_{\text{model}}=256, n_{\text{heads}}=4, d_{\text{head}}=64, \text{context}=1024$).
- **Modern Transformer Primitives**: Rotary Position Embeddings (RoPE), Pre-LayerNorm residual connections, PyTorch Scaled Dot-Product Attention (SDPA), and tied input/output embedding weights.
- **Domain BPE Tokenizer**: 4,096-vocabulary Byte-Level BPE tokenizer trained strictly on football analytics corpus with custom special tokens (`<|narrative|>`, `<|player_dossier|>`, `<|tactical_qa|>`, `<|endoftext|>`).
- **Convergence**: Trained for 3,000 steps on GPU (final train loss: **0.1027**, validation loss: **0.5174**, held-out test loss: **0.4808**).

### Inference & Serving Architecture
- **Single-Request Local Inference (`src/inference/engine.py`)**: CPU-safe checkpoint deserialization, deterministic greedy decoding, and top-k/top-p nucleus sampling with reproducible seed control.
- **Dynamic KV Cache**: Autoregressive decode optimization eliminating quadratic prefix recomputation, achieving **~950–1,025 tok/s** on NVIDIA RTX 4050 GPU.
- **Static Preallocated KV Cache (`src/inference/kv_cache.py`)**: Zero-allocation in-place buffer updates via `narrow().copy_()`, completely eliminating memory fragmentation during generation.
- **Batched Inference (`generate_batch`)**:
  - Supports heterogeneous prompt lengths via token `<|endoftext|>` left-padding.
  - Per-token 2D `position_ids` (`[B, T]`).
  - Causal + padding attention masks with unmasked diagonals on pad query rows to eliminate SDPA softmax NaNs.
  - Scales aggregate throughput to **4,482.3 tok/s at $B=8$** (**$4.44\times$ speedup** over sequential execution).
- **Continuous Batching Scheduler (`src/inference/continuous_scheduler.py`)**:
  - Iteration-level scheduling eliminating the early-completion execution bubble.
  - FIFO waiting queue with immediate admission into freed slots.
  - Dedicated **Slot-Allocated KV Cache (`src/inference/slot_cache.py`)** managing independent sequence lengths and slot isolation.
  - Immediate eviction on EOS or per-request token limits.
  - **100% Deterministic Parity**: Greedy tokens produced under continuous batching match single-request sequential inference token-for-token.
- **Comprehensive Test Suite**: **103 passing tests** across training, tokenization, model forward passes, RoPE, KV caching, batched inference, and continuous batching.

---

## 3. Serving Benchmarks & Performance (NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM)

### 3.1 Throughput Scaling across Workload Patterns (Static KV Cache)

| Workload Pattern | $B=1$ TPS | $B=2$ TPS (Speedup) | $B=4$ TPS (Speedup) | $B=8$ TPS (Speedup) | Peak Alloc VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Homogeneous** | $736.4\text{ tok/s}$ | $1,387.7\text{ tok/s}$ (**1.45x**) | $2,693.2\text{ tok/s}$ (**2.67x**) | **$4,482.3\text{ tok/s}$** (**4.44x**) | $94.3\text{ MB}$ |
| **Heterogeneous** (Left-Pad) | $737.1\text{ tok/s}$ | $1,259.5\text{ tok/s}$ (**1.29x**) | $2,499.9\text{ tok/s}$ (**2.55x**) | **$4,169.1\text{ tok/s}$** (**4.23x**) | $109.5\text{ MB}$ |
| **Early Completion** (Bubble) | $695.4\text{ tok/s}$ | $1,104.0\text{ tok/s}$ (**1.17x**) | $1,720.3\text{ tok/s}$ (**1.73x**) | **$2,928.8\text{ tok/s}$** (**2.98x**) | $109.5\text{ MB}$ |

*Detailed timing breakdowns, TTFT, inter-token latencies, and speedup baselines are documented in [`docs/batched_inference_benchmarks.md`](docs/batched_inference_benchmarks.md).*

---

## 4. Repository Structure

```text
TouchlineLM/
├── configs/
│   ├── model_config.json          # 4.2M parameter Transformer configuration
│   └── tokenizer/                 # Custom BPE tokenizer artifacts (vocab.json, merges.txt)
│
├── checkpoints/                   # Checkpoints (final_model.pt: 3,000 steps)
│
├── data/
│   ├── raw/                       # Raw event data (StatsBomb WC 2022 dataset)
│   ├── intermediate/              # Extracted matches, metrics, benchmark JSONs
│   └── processed/splits/          # Leak-free train, validation, and test splits
│
├── docs/
│   ├── local_inference_guide.md          # Local CLI and Python inference guide
│   ├── inference_baseline.md             # Baseline uncached profiling report
│   ├── batched_inference_design.md       # Static batching design specification
│   ├── batched_inference_benchmarks.md   # Audited batched benchmarks & throughput analysis
│   └── continuous_batching_design.md     # Continuous batch scheduler architecture specification
│
├── scripts/
│   ├── generate.py                       # CLI inference generator with sampling & REPL
│   ├── benchmark_inference.py            # Baseline timing benchmark
│   ├── profile_kv_cache_memory.py        # CUDA VRAM memory profiling script
│   ├── benchmark_batched_inference.py    # Batched throughput benchmark suite
│   ├── train.py                          # Training pipeline CLI
│   └── train_tokenizer.py                # BPE tokenizer training script
│
├── src/
│   ├── model/                     # TouchlineNanoGPT, CausalSelfAttention, RoPE, Blocks
│   ├── tokenizer/                 # TouchlineBPETokenizer wrapper
│   ├── inference/
│   │   ├── engine.py              # TouchlineInferenceEngine (single & static batch generation)
│   │   ├── kv_cache.py            # DynamicKVCache and StaticKVCache
│   │   ├── slot_cache.py          # SlotAllocatedKVCache (fixed-slot preallocated memory pool)
│   │   └── continuous_scheduler.py# ContinuousBatchScheduler (iteration-level scheduling)
│   ├── metrics/                   # Deterministic football metrics (xG, xA, progressive actions)
│   └── corpus/                    # Multi-tier text generation engine
│
└── tests/                         # 103 automated tests (unit, integration, regression)
    ├── test_model.py              # Architecture, RoPE, attention invariants
    ├── test_tokenizer.py          # Roundtrip fidelity, special tokens
    ├── test_inference.py          # Greedy invariance, nucleus sampling, EOS stopping
    ├── test_kv_cache.py           # Dynamic & Static cache numerical equivalence
    ├── test_batched_inference.py  # Left-padded batched equivalence & error handling
    ├── test_slot_cache.py         # Slot isolation, slicing, stale data masking
    └── test_continuous_batching.py# Iteration scheduling, mid-stream admission, greedy parity
```

---

## 5. Quickstart & Usage

### 5.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/Athullvr/TouchlineLM.git
cd TouchlineLM

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Text Generation (CLI & Python)

```bash
# 1. Quick CLI generation with pre-trained checkpoint:
python scripts/generate.py --prompt "<|narrative|> # Match Report: Argentina vs France" --max-new-tokens 100

# 2. Interactive Terminal REPL:
python scripts/generate.py --interactive
```

**Python Inference API**:
```python
from src.inference.engine import TouchlineInferenceEngine

# Load engine from trained checkpoint
engine = TouchlineInferenceEngine.load_from_checkpoint(
    checkpoint_path="checkpoints/final_model.pt",
    tokenizer_dir="configs/tokenizer",
    device="auto",
)

# Single request with Static KV Cache
output = engine.generate(
    prompt="<|narrative|> Match summary: In the 85th minute",
    max_new_tokens=50,
    temperature=0.7,
    cache_type="static",
)
print(output.generated_text)
```

### 5.3 Batched Inference
```python
from src.inference.engine import BatchRequest

requests = [
    BatchRequest(prompt="Hello", max_new_tokens=20),
    BatchRequest(prompt="The match was intense from the opening whistle", max_new_tokens=40),
]

# Run lockstep batched inference with left-padding:
batch_output = engine.generate_batch(requests, cache_type="static")
for item in batch_output.items:
    print(f"[{item.request_index}] {item.generated_text}")
```

### 5.4 Continuous Batching Scheduler
```python
from src.inference.continuous_scheduler import ContinuousBatchScheduler, ContinuousBatchRequest

scheduler = ContinuousBatchScheduler(
    model=engine.model,
    tokenizer=engine.tokenizer,
    max_active_slots=8,
)

# Submit asynchronous client requests
r1 = scheduler.submit_request(ContinuousBatchRequest(prompt="Short query", max_new_tokens=15))
r2 = scheduler.submit_request(ContinuousBatchRequest(prompt="Detailed tactical breakdown", max_new_tokens=60))

# Execute until complete (or call scheduler.step() in a server event loop)
results = scheduler.run_all([])
for res in results:
    print(f"Req {res.request_id} ({res.tokens_per_second:.1f} tok/s): {res.generated_text}")
```

### 5.5 Run Test Suite
Run all 103 tests across model, tokenizer, and serving infrastructure:
```bash
pytest -v
```

---

## 6. Dataset Provenance and Attribution

- **StatsBomb Open Data**: Event data provided by [StatsBomb](https://statsbomb.com/). Used under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
- **Football-Data.co.uk**: Historical match records and betting odds sourced from [Football-Data.co.uk](https://www.football-data.co.uk/).

---

## 7. License

Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for details.
