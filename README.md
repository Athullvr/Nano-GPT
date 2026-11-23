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

## 2. Phase 1 Status: Dataset Research and Acquisition

Phase 1 establishes the verified data foundation and analytical framework:
- **Evaluated Data Sources**: Thorough comparison of StatsBomb Open Data, Football-Data.co.uk, Wyscout Open Soccer Dataset, and Understat.
- **Selected Primary Source**: **StatsBomb Open Data** (CC BY 4.0), providing event streams with continuous spatial coordinates, validated `statsbomb_xg`, and lineups.
- **Reproducible Pipeline**: Python scripts for deterministic acquisition, data inspection, schema validation, and test coverage.
- **Zero Hallucination Strategy**: Multi-tier corpus synthesis framework transforming event streams into structured narratives, player summaries, and tactical explanations without fabricating statistics.

---

## 3. Directory Architecture

```text
TouchlineLM/
│
├── data/
│   ├── raw/                 # Pristine raw datasets (StatsBomb JSON, Football-Data CSV)
│   ├── processed/           # Processed datasets (reproducible through scripts)
│   ├── intermediate/        # Intermediate artifacts and staged sequences
│   └── metadata/            # Checksums, download manifests, and inspection summaries
│
├── research/
│   ├── dataset_sources.md   # Comparative analysis of candidate football datasets
│   ├── data_dictionary.md   # Domain schema, pitch geometry, and metric definitions
│   ├── dataset_analysis.md  # Empirical quality, completeness, and nullness audit
│   ├── text_representation_strategy.md # Structured data to training corpus framework
│   └── unresolved_decisions.md         # Open architectural decisions for Phase 2
│
├── scripts/
│   ├── download_data.py     # Deterministic data download with SHA256 verification
│   ├── inspect_data.py      # Quantitative inspection and metric distribution analyzer
│   └── validate_data.py     # Schema integrity, bounds, and referential validator
│
├── configs/                 # Future training and model hyperparameters
├── notebooks/               # Future exploratory notebooks
├── src/                     # Core library code for future phases
├── tests/                   # Pytest test suite for data validation and integrity
│
├── pyproject.toml           # Project metadata and test configuration
├── requirements.txt         # Pinned virtual environment dependencies
├── README.md                # Project documentation
├── LICENSE                  # Apache 2.0 License
└── .gitignore               # Excludes raw data and virtual environment caches
```

---

## 4. Quickstart & Reproducibility Guide

### 4.1 Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Standard Unix/Linux shell environment

### 4.2 Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/Athullvr/Nano-GPT.git
cd Nano-GPT

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 4.3 Data Acquisition
Download the initial research sample (StatsBomb World Cup 2022 matches + Premier League macro benchmark):
```bash
python scripts/download_data.py
```
This downloads sample JSON/CSV files into `data/raw/` and records provenance metadata in `data/metadata/download_manifest.json`.

### 4.4 Data Inspection & Validation
Inspect event distributions, metrics, and spatial coordinate coverage:
```bash
python scripts/inspect_data.py
```

Run the validation suite to confirm schema compliance, coordinate bounds, and referential integrity:
```bash
python scripts/validate_data.py
```

### 4.5 Run Test Suite
Execute the automated test suite:
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
