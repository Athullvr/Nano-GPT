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

# 2. Run autoregressive text generation
output = engine.generate(
    prompt="<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini\n**Position**: Right Wing",
    max_new_tokens=150,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    seed=42,
    stop_at_eos=True,
)

# 3. Access structured results and telemetry
print(output.full_text)
print(f"Generated: {output.new_tokens} tokens in {output.latency_seconds:.3f}s")
print(f"Throughput: {output.tokens_per_second:.1f} tokens/second on {output.device.upper()}")
print(f"Stopped on EOS: {output.stopped_on_eos}")
```

---

## 4. Hardware Performance Benchmarks

Benchmarks measured on local hardware running Linux (kernel 6.18, PyTorch 2.15.0+cu132):
- **CPU**: AMD Ryzen 7 / Intel Core (x86_64)
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)

### Benchmark Results (100 New Tokens per Prompt):

| Prompt ID | Domain Category | Device | Prompt Tokens | New Tokens | Latency (s) | Throughput (tok/s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | Match Narrative (Final) | **CPU** | 32 | 100 | 0.382s | **261.8 tok/s** |
| **2** | Player Dossier (Messi) | **CPU** | 34 | 100 | 0.374s | **267.0 tok/s** |
| **3** | Match Narrative (QF) | **CPU** | 23 | 100 | 0.341s | **293.4 tok/s** |
| **4** | Tactical Q&A | **CPU** | 24 | 100 | 0.343s | **291.7 tok/s** |
| **Average** | **All Categories** | **CPU** | — | — | **0.360s** | **278.5 tok/s** |
| | | | | | | |
| **1** | Match Narrative (Final) | **CUDA** | 32 | 100 | 0.119s | **839.8 tok/s** |
| **2** | Player Dossier (Messi) | **CUDA** | 34 | 100 | 0.113s | **887.2 tok/s** |
| **3** | Match Narrative (QF) | **CUDA** | 23 | 100 | 0.111s | **900.6 tok/s** |
| **4** | Tactical Q&A | **CUDA** | 24 | 100 | 0.110s | **905.4 tok/s** |
| **Average** | **All Categories** | **CUDA** | — | — | **0.113s** | **883.2 tok/s** |

### Key Takeaways:
1. **CPU Efficiency**: TouchlineLM achieves over **275 tokens/second** on standard CPU, enabling real-time local text generation with zero GPU dependency.
2. **CUDA Acceleration**: On local RTX 4050 mobile GPU, throughput reaches **~880–905 tokens/second** with ~110 ms latency for 100 tokens.
3. **Memory Footprint**: Model weights consume only **~18 MB** of VRAM / RAM in float32 evaluation mode.
