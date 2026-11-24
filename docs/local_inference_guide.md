# TouchlineLM — Local Inference Guide (Phase 2 & 3)

This guide documents the local inference workflow, CLI options, programmatic Python API, and hardware performance benchmarks for **TouchlineLM** (`TouchlineNanoGPT`), a domain-adapted decoder-only Transformer for football intelligence.

---

## 1. Checkpoint & Artifact Prerequisites

Ensure the trained checkpoint and tokenizer artifacts are present in your workspace:

| Artifact | Default Path | Verified Status |
| :--- | :--- | :--- |
| **Model Checkpoint** | `checkpoints/final_model.pt` | 4,198,912 parameters, Step 3,000 (48.1 MB) |
| **Model Configuration** | `configs/model_config.json` | 4 layers, 4 heads, $d=256$, RoPE, context 1024 |
| **Tokenizer Vocabulary** | `configs/tokenizer/vocab.json` | 4,096 tokens (Byte-Level BPE) |
| **Tokenizer Merges** | `configs/tokenizer/merges.txt` | 3,836 BPE merges |
| **Tokenizer Config** | `configs/tokenizer/tokenizer_config.json` | Special tokens: `<|endoftext|>`, `<|narrative|>`, `<|player_dossier|>`, `<|tactical_qa|>` |

---

## 2. CLI Generation Tool (`scripts/generate.py`)

The project provides a self-contained CLI tool at [`scripts/generate.py`](file:///home/athulvr/Projects/Nano Gpt/scripts/generate.py).

### 2.1 Basic Usage
Generate with the default evaluation prompt:
```bash
python scripts/generate.py
```

### 2.2 Custom Text Prompt
Provide an inline prompt:
```bash
python scripts/generate.py \
    --prompt "<|narrative|> # Match Report: Argentina vs France\n**Competition Stage**: Final | **Date**: 2022-12-18\n**Final Scoreline**: Argentina" \
    --max-new-tokens 150 \
    --temperature 0.7
```

Read prompt from a file:
```bash
python scripts/generate.py \
    --prompt-file path/to/prompt.txt \
    --max-new-tokens 200
```

### 2.3 Sampling Controls & Determinism
- **Per-Layer KV Caching (Enabled by Default)**:
  By default, generation uses per-layer KV caching for fast $\mathcal{O}(1)$ incremental decoding.
- **Uncached Baseline Mode (`--no-cache`)**:
  To disable caching and evaluate legacy full-sequence recomputation:
  ```bash
  python scripts/generate.py --no-cache
  ```
- **Greedy Decoding (Deterministic, zero temperature)**:
  ```bash
  python scripts/generate.py --prompt "..." --temperature 0.0
  ```
- **Nucleus (Top-p) & Top-k Sampling**:
  ```bash
  python scripts/generate.py --temperature 0.75 --top-k 40 --top-p 0.9
  ```
- **Reproducible Seeded Generation**:
  ```bash
  python scripts/generate.py --seed 42
  ```
- **Disable EOS Stopping**:
  By default, generation terminates immediately when encountering `<|endoftext|>`. To force generation to `max_new_tokens`:
  ```bash
  python scripts/generate.py --no-eos-stop
  ```

### 2.4 Device Selection
- **Auto (Default)**: Automatically selects CUDA if an NVIDIA GPU is available, otherwise CPU:
  ```bash
  python scripts/generate.py --device auto
  ```
- **Explicit CPU**:
  ```bash
  python scripts/generate.py --device cpu
  ```
- **Explicit CUDA**:
  ```bash
  python scripts/generate.py --device cuda
  ```

### 2.5 Standardized Benchmark Suite
Runs a 4-prompt benchmark suite measuring prompt tokens, generated tokens, latency, and throughput (tokens/second):
```bash
python scripts/generate.py --benchmark --device cpu
python scripts/generate.py --benchmark --device cuda
```

### 2.6 Interactive REPL Mode
Launch an interactive session to query match narratives, player dossiers, or tactical concepts directly:
```bash
python scripts/generate.py --interactive
```

---

## 3. Programmatic Python API

You can integrate TouchlineLM generation directly into Python applications using [`TouchlineInferenceEngine`](file:///home/athulvr/Projects/Nano Gpt/src/inference/engine.py):

```python
from src.inference.engine import TouchlineInferenceEngine

# 1. CPU-safe initial checkpoint deserialization and device placement
engine = TouchlineInferenceEngine.load_from_checkpoint(
    checkpoint_path="checkpoints/final_model.pt",
    tokenizer_dir="configs/tokenizer",
    model_config_path="configs/model_config.json",
    device="auto",  # 'auto', 'cuda', or 'cpu'
)

# 2. Run autoregressive text generation (KV caching enabled by default)
output = engine.generate(
    prompt="<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini\n**Position**: Right Wing",
    max_new_tokens=150,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    seed=42,
    stop_at_eos=True,
    min_tokens=0,
    use_cache=True,  # Set False for legacy uncached recomputation
)

# 3. Access structured results and telemetry
print(output.full_text)
print(f"Generated: {output.new_tokens} tokens in {output.latency_seconds:.3f}s")
print(f"Throughput: {output.tokens_per_second:.1f} tokens/second on {output.device.upper()}")
print(f"Cache Status: {'KV Cache ON' if output.use_cache else 'Uncached OFF'}")
print(f"Stopped on EOS: {output.stopped_on_eos}")
```

---

## 4. Hardware Performance Benchmarks (Cached vs Uncached)

Benchmarks measured on local hardware running Linux (kernel 6.18, PyTorch 2.15.0+cu132):
- **CPU**: AMD Ryzen 7 / Intel Core (x86_64)
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)

### 4.1 Comparative Benchmark Results

| Prompt / Workload | Sequence Profile | Device | Mode | Prefill (ms) | Decode ms/tok | Total Latency (ms) | Throughput (tok/s) | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short** (Match Narrative) | 32 prompt $\to$ 100 new | **CPU** | Uncached | 4.05 ms | 5.94 ms | 592.0 ms | 168.9 tok/s | Baseline |
| | | **CPU** | **KV-Cached** | **2.79 ms** | **1.18 ms** | **119.1 ms** | **839.4 tok/s** | **4.97x** |
| **Medium** (Player Dossier) | 39 prompt $\to$ 150 new | **CPU** | Uncached | 2.79 ms | 5.13 ms | 766.7 ms | 195.6 tok/s | Baseline |
| | | **CPU** | **KV-Cached** | **3.13 ms** | **1.19 ms** | **180.6 ms** | **830.5 tok/s** | **4.25x** |
| **Long** (Tactical Analysis) | 173 prompt $\to$ 200 new | **CPU** | Uncached | 5.09 ms | 12.39 ms | 2,473.7 ms | 80.9 tok/s | Baseline |
| | | **CPU** | **KV-Cached** | **3.73 ms** | **1.48 ms** | **301.3 ms** | **663.7 tok/s** | **8.21x** |
| | | | | | | | | |
| **Short** (Match Narrative) | 32 prompt $\to$ 100 new | **CUDA** | Uncached | 1.41 ms | 1.40 ms | 140.4 ms | 712.2 tok/s | Baseline |
| | | **CUDA** | **KV-Cached** | 1.74 ms | **1.32 ms** | **132.4 ms** | **755.5 tok/s** | **1.06x** |
| **Medium** (Player Dossier) | 39 prompt $\to$ 150 new | **CUDA** | Uncached | 1.36 ms | 1.42 ms | 212.6 ms | 705.4 tok/s | Baseline |
| | | **CUDA** | **KV-Cached** | 1.83 ms | **1.34 ms** | **200.9 ms** | **746.8 tok/s** | **1.06x** |
| **Long** (Tactical Analysis) | 173 prompt $\to$ 200 new | **CUDA** | Uncached | 1.43 ms | 1.55 ms | 310.5 ms | 644.2 tok/s | Baseline |
| | | **CUDA** | **KV-Cached** | 1.90 ms | **1.36 ms** | **271.9 ms** | **735.6 tok/s** | **1.14x** |

### 4.2 Key Architectural Insights:
1. **CPU Decode Scaling**: In uncached mode, per-token decode latency rises from **4.14 ms/tok** up to **15.58 ms/tok** at token 373 due to quadratic $\mathcal{O}(T)$ sequence recomputation. With KV caching, decode latency is constant at **~1.18–1.48 ms/token** regardless of position.
2. **Real-Time CPU Throughput**: On standard CPU, KV caching delivers up to **839 tokens/second** and an **8.2x speedup** on long sequence generation.
3. **GPU Memory Footprint**: Peak VRAM remains minimal (< 365 MB at full context) while maintaining > 735–755 tokens/second.
4. **Design Reference**: For detailed mathematical proofs and architecture diagrams, see [`docs/kv_cache_implementation.md`](file:///home/athulvr/Projects/Nano Gpt/docs/kv_cache_implementation.md).
