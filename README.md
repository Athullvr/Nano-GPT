# TouchlineLM: Football Intelligence NanoGPT

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/Project%20Phase-Phase%201%20Completed-green.svg)]()

**TouchlineLM** is an experimental, domain-specific small decoder-only Transformer (NanoGPT) engineered to model football intelligence, tactical patterns, and advanced event-level analytics beyond traditional match statistics.

---

## 1. Project Vision

Traditional sports language applications often regurgitate superficial scorelines. **TouchlineLM** is designed to explore how an autoregressive language model can internalize the mathematical and tactical structure of modern football analytics:

- **Defensive Micro-Actions**: Interceptions, ground duels, clearances, ball recoveries, and high-pressing triggers.
- **Ball Progression**: Progressive passes, progressive carries, and zone entries.
- **Shot Creation & Finishing**: Expected Goals (xG), Expected Assists (xA), shot techniques, and freeze-frame spatial contexts.
- **Tactical Alignments**: Formations (e.g., 4-3-3, 4-2-3-1, 3-5-2), tactical substitutions, and spatial channel dynamics.
- **Holistic Contributions**: Capturing player value that eludes traditional box scores.

The long-term objective is to deepen understanding of Transformer architectures, domain-adapted tokenization, data synthesis from structured graphs, and inference engineering through a technically rigorous, reproducible experimental pipeline.

---

## 2. Project Status & Roadmap

- **Phase 1: Dataset Research and Acquisition** *(Completed)*: Evaluated data sources, selected StatsBomb Open Data (CC BY 4.0), implemented acquisition and schema validation.
- **Phase 2: Data Extraction & Corpus Engineering** *(Completed)*: Deterministic metrics engine, multi-tier corpus generation (narratives, player dossiers, tactical Q&A), and automated fact-checking.
- **Phase 3A: Corpus Expansion & Match-Level Splits** *(Completed)*: Ingested all 64 World Cup 2022 matches (234,637 events, 1,980 documents), added syntactic permutations, and partitioned into leak-free match splits.
- **Phase 3B.1: Tokenizer Engineering** *(Completed)*: Built custom 4,096-vocabulary Byte-Level BPE tokenizer trained strictly on `train.txt`, with special tokens and benchmarked against GPT-2 baseline.
- **Phase 3B.2: NanoGPT Model Architecture** *(Completed)*: Built educational decoder-only Transformer from scratch in PyTorch (~4.20M parameters, RoPE, Pre-LayerNorm, PyTorch SDPA causal attention, tied embeddings).
- **Phase 4A: Training Pipeline Implementation** *(Completed)*: Dataset batching with binary caching, AdamW parameter grouping, cosine warmup scheduler, atomic checkpointing, and Google Colab GPU training setup.
- **Phase 4B & 4B.1: Google Colab Training Notebook & Consistency Audit** *(Completed)*: Created self-contained training notebook (`notebooks/TouchlineLM_Colab_Training.ipynb`) with hardware fail-fast guards, dataset verification, isolated 15-step GPU smoke test, audited gradient accumulation, 48-epoch token budget analysis, and Google Drive checkpointing.
- **Phase 4C: Model Pre-Training & Convergence** *(Completed)*: 3,000 optimizer steps completed on GPU (final train loss: 0.1027, val loss: 0.5174, test loss: 0.4808). Saved checkpoint at `checkpoints/final_model.pt`.
- **Phase 5: Local Inference & Validation** *(Completed)*: Built CPU-safe inference engine (`src/inference/`), CLI text generator (`scripts/generate.py`), nucleus sampling, EOS stopping, and benchmarked 278 tok/s (CPU) and 883 tok/s (CUDA). See [`docs/local_inference_guide.md`](docs/local_inference_guide.md).

---

## 3. Directory Architecture

```text
TouchlineLM/
│
├── configs/
│   ├── model_config.json    # 4.2M parameter Transformer configuration
│   └── tokenizer/           # Saved BPE tokenizer artifacts (vocab.json, merges.txt, config)
│
├── checkpoints/             # Model checkpoints (best_model.pt, final_model.pt)
│
├── data/
│   ├── raw/                 # Pristine raw datasets (StatsBomb JSON, Football-Data CSV)
│   ├── intermediate/        # Extracted matches and player metrics (JSON)
│   ├── processed/
│   │   ├── corpus/          # Generated multi-tier markdown & jsonl documents
│   │   └── splits/          # Match-level partitions (train, val, test .txt & .jsonl)
│   └── metadata/            # Manifests, diversity reports, and fact-checking audits
│
├── notebooks/
│   └── TouchlineLM_Colab_Training.ipynb # Self-contained Google Colab GPU training notebook
│
├── research/
│   ├── dataset_sources.md   # Comparative analysis of football datasets
│   ├── data_dictionary.md   # Domain schema, pitch geometry, and metric definitions
│   ├── dataset_analysis.md  # Empirical quality and completeness audit
│   ├── text_representation_strategy.md # Multi-tier corpus architecture
│   ├── corpus_quality_report.md        # Phase 2 quality and diversity audit
│   ├── phase_3a_report.md              # Phase 3A dataset expansion and splits report
│   ├── tokenizer_comparison_report.md  # Tokenizer benchmark and fragmentation report
│   ├── model_architecture_report.md    # Model architecture & parameter accounting report
│   └── colab_training_guide.md         # Google Colab GPU training guide and commands
│
├── scripts/
│   ├── download_data.py     # Deterministic data download with SHA256 verification
│   ├── inspect_data.py      # Quantitative inspection and metric distribution analyzer
│   ├── validate_data.py     # Schema integrity and coordinate bounds validator
│   ├── process_corpus.py    # Full end-to-end extraction, generation, and splitting pipeline
│   ├── train_tokenizer.py   # Custom BPE tokenizer training script
│   ├── evaluate_tokenizers.py # Comparative tokenizer evaluation benchmark
│   └── train.py             # Configurable training entrypoint (CLI & Colab)
│
├── src/
│   ├── extraction/          # Match metadata, event, and lineup extractors
│   ├── metrics/             # Deterministic calculations (progressive actions, xG, xA)
│   ├── corpus/              # Multi-tier text generation engine with permutations
│   ├── validation/          # 100% automated statistical fact-checking engine
│   ├── tokenizer/           # Custom Byte-Level BPE & GPT-2 baseline tokenizers
│   ├── model/               # TouchlineNanoGPT (RoPE, SDPA causal attention, Pre-LN, tied LM head)
│   └── training/            # Dataset batching, AdamW optimizer, scheduler, TouchlineTrainer
│
├── tests/                   # Pytest automated test suite (38 unit & integration tests)
├── pyproject.toml           # Project metadata and test configuration
├── requirements.txt         # Pinned virtual environment dependencies
├── README.md                # Project documentation
├── LICENSE                  # Apache 2.0 License
└── .gitignore               # Strict exclusion of raw data and cache directories
```

---

## 4. Quickstart & Reproducibility Guide

### 4.1 Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Standard Unix/Linux shell environment

### 4.2 Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/Athullvr/TouchlineLM.git
cd TouchlineLM

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 4.3 Data Pipeline Execution
```bash
# Download tournament data
python scripts/download_data.py --source statsbomb --all-matches

# Process corpus, extract deterministic metrics, and create match splits
python scripts/process_corpus.py
```

### 4.4 Tokenizer Training & Evaluation
```bash
# Train custom 4,096-token BPE tokenizer on train split only
python scripts/train_tokenizer.py --train-file data/processed/splits/train.txt --vocab-size 4096

# Evaluate and compare Custom BPE vs GPT-2 baseline
python scripts/evaluate_tokenizers.py
```

### 4.5 Training Pipeline & Google Colab Execution
```bash
# Verify training pipeline locally with a 3-step dry run (no full training):
python scripts/train.py --dry-run --device cpu

# Launch full pre-training on Google Colab (CUDA GPU):
# See research/colab_training_guide.md for the complete Colab setup instructions.
python scripts/train.py --device cuda --batch-size 16 --seq-len 512 --max-steps 3000
```

### 4.6 Local Inference & Generation
```bash
# Generate with default prompt (CPU or CUDA):
python scripts/generate.py

# Custom prompt with temperature and top-p sampling:
python scripts/generate.py --prompt "<|narrative|> # Match Report: Argentina vs France" --max-new-tokens 150 --temperature 0.7

# Run benchmark suite across evaluation prompts:
python scripts/generate.py --benchmark --device cuda

# Interactive terminal REPL:
python scripts/generate.py --interactive
```

### 4.7 Run Test Suite
Execute the automated test suite across all modules:
```bash
pytest -v
```

---

## 5. Dataset Provenance and Attribution

- **StatsBomb Open Data**: Event data provided by [StatsBomb](https://statsbomb.com/). Used under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
- **Football-Data.co.uk**: Historical match records and betting odds sourced from [Football-Data.co.uk](https://www.football-data.co.uk/).

---

## 6. Project Roadmap

- **Phase 1: Dataset Research and Acquisition** *(Completed)*
  - Research open sources, implement acquisition pipeline, audit data quality, establish representation strategy.
- **Phase 2: Corpus Synthesis & Tokenizer Engineering** *(Upcoming)*
  - Implement deterministic metric calculators and narrative synthesizers.
  - Train custom domain Byte-Pair Encoding (BPE) tokenizer.
- **Phase 3: NanoGPT Architecture Implementation**
  - Implement decoder-only Transformer with causal self-attention, Rotary/Learned positional embeddings, and MLP feed-forward blocks.
- **Phase 4: Pre-Training & Learning Dynamics**
  - Autoregressive pre-training, loss convergence analysis, and domain-perplexity tracking.
- **Phase 5: Evaluation & Tactical Inference**
  - Domain-specific MCQ benchmarks, scouting report generation, and tactical reasoning probing.
