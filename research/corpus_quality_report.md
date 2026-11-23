# TouchlineLM: Phase 2 Corpus Engineering & Quality Report

**Project**: TouchlineLM (Football Intelligence NanoGPT)  
**Phase**: 2 - Data Extraction and Corpus Engineering  
**Date**: November 2026  
**Auditor**: Senior ML Research & Data Engineering Team  

---

## 1. Executive Summary

Phase 2 transitions TouchlineLM from raw dataset evaluation into a reproducible, deterministic data processing and domain corpus engineering pipeline.

Following user approval of **StatsBomb Open Data** as the primary source, a representative benchmark of **15 FIFA World Cup 2022 matches** was extracted, processed into deterministic analytical representations, and synthesized into a multi-tiered football intelligence training corpus.

### Key Milestones Achieved
- **Clean Architectural Separation**: Strict demarcation maintained between raw inputs (`data/raw/`), extracted deterministic metrics (`data/intermediate/`), and generated training texts (`data/processed/corpus/`).
- **Zero Hallucination Guaranteed**: Every numerical statistic in the generated corpus was cross-checked against source data via an automated fact-checking engine: **9,350 factual claims verified with 0 violations (100.0% accuracy)**.
- **Substantial Initial Corpus**: **481 training text samples** totaling **108,897 words** (748,226 characters, ~141,566 tokens), capturing match narratives, scouting dossiers, and tactical reasoning.

---

## 2. Ingestion and Extraction Audit

The extraction pipeline processed 15 fixtures across all tournament stages:

| Match ID | Tournament Stage | Fixture | Score | Total Events | Extracted Players |
|---|---|---|---|---|---|
| `3869685` | Final | Argentina vs France | 3 - 3 (Pen) | 4,407 | 50 |
| `3869519` | Semi-finals | Argentina vs Croatia | 3 - 0 | 3,806 | 49 |
| `3869552` | Semi-finals | France vs Morocco | 2 - 0 | 3,551 | 48 |
| `3869321` | Quarter-finals | Netherlands vs Argentina | 2 - 2 (Pen) | 4,708 | 52 |
| `3869354` | Quarter-finals | England vs France | 1 - 2 | 3,239 | 49 |
| `3869420` | Quarter-finals | Croatia vs Brazil | 1 - 1 (Pen) | 5,145 | 50 |
| `3869486` | Quarter-finals | Morocco vs Portugal | 1 - 0 | 3,381 | 48 |
| `3869254` | Round of 16 | Portugal vs Switzerland | 6 - 1 | 3,422 | 49 |
| `3869253` | Round of 16 | Brazil vs South Korea | 4 - 1 | 4,046 | 49 |
| `3869220` | Round of 16 | Morocco vs Spain | 0 - 0 (Pen) | 4,750 | 52 |
| `3857271` | Group Stage | England vs Iran | 6 - 2 | 3,649 | 51 |
| `3857255` | Group Stage | Japan vs Spain | 2 - 1 | 4,450 | 52 |
| `3857263` | Group Stage | Spain vs Germany | 1 - 1 | 3,603 | 52 |
| `3857300` | Group Stage | Argentina vs Saudi Arabia | 1 - 2 | 3,329 | 52 |
| `3857256` | Group Stage | Serbia vs Switzerland | 2 - 3 | 3,210 | 49 |
| **TOTAL** | — | **15 Fixtures** | — | **58,696** | **752 Player Records** |

### Intermediate Data Storage
- [data/intermediate/extracted_matches.json](file:///home/athulvr/Projects/Nano%20Gpt/data/intermediate/extracted_matches.json): Stores macro metrics, 15-minute phase splits, and goal event structures.
- [data/intermediate/extracted_players.json](file:///home/athulvr/Projects/Nano%20Gpt/data/intermediate/extracted_players.json): Stores 752 player performance records capturing 24 distinct statistical dimensions per player-match.

---

## 3. Corpus Generation Architecture

The corpus engine (`src/corpus/generator.py`) produces three complementary tiers of training text:

### Tier 1: Structured Match Narratives (15 Samples)
- **Role**: Models chronological progression, game state momentum, and decisive phases.
- **Structure**:
  1. Header and official fixture metadata.
  2. Macroscopic summary (passes, completion %, progressive yards, cumulative xG, defensive duels).
  3. Chronological 15-minute phase breakdowns (`0-15'`, `15-30'`, `30-45'`, `45-60'`, `60-75'`, `75-90'`, `Extra Time`) detailing tactical flow and exact goal moments.
  4. Final tactical verdict.
- **Location**: `data/processed/corpus/narratives/`

### Tier 2: Analytical Player Dossiers (451 Samples)
- **Role**: Teaches the Transformer entity-level scouting intelligence and statistical correlations across specific tactical roles (e.g., deep-lying playmakers, pressing forwards, fullbacks).
- **Structure**:
  1. Player identity, squad number, tactical position, starting status.
  2. Distribution & Progression (attempts, completion %, progressive passes, passes into attacking third/box, progressive carries).
  3. Attacking output (shots, shots on target, goals, individual xG, key passes, expected assists xA).
  4. Defensive interventions (tackles won/attempted, interceptions, clearances, blocks, recoveries, pressures).
  5. Tactical role evaluation.
- **Location**: `data/processed/corpus/players/`

### Tier 3: Tactical & Analytical Reasoning Q&A (15 Samples)
- **Role**: Conditions the language model on analytical reasoning and comparative query responses.
- **Structure**:
  - Q1: Comparison of expected goals ($\text{xG}$) to actual match outcome.
  - Q2: Analysis of tempo, ball circulation, and spatial progression.
  - Q3: Defensive disruption, turnover generation, and pressing efficiency.
- **Location**: `data/processed/corpus/tactical_qa/`

---

## 4. Quantitative Corpus Dimensions

| Dimension | Metric Value | Notes |
|---|---|---|
| **Total Text Samples** | **481** | 15 Narratives, 451 Dossiers, 15 Q&A sets |
| **Total Word Count** | **108,897 words** | Clean alphanumeric word tokens |
| **Total Character Count** | **748,226 characters** | Plaintext volume |
| **Estimated Token Count** | **~141,566 tokens** | Based on domain BPE ratio (~1.3 tokens/word) |
| **Vocabulary Size** | **1,362 unique words** | Domain terms, player names, and technical terms |
| **Type-Token Ratio (TTR)** | **1.25%** | Typical for structured, high-volume analytical corpora |
| **Distinct Bigrams** | **7,964** | 7.31% bigram diversity |
| **Distinct Trigrams** | **15,073** | 13.84% trigram diversity |

### Top Domain Vocabulary Distribution
- `passes`: 1,683 occurrences
- `progressive`: 1,022 occurrences
- `carries`: 962 occurrences
- `defensive`: 1,101 occurrences
- `interceptions`: 912 occurrences
- `recoveries`: 894 occurrences
- `tackles`: 884 occurrences
- `expected` / `xG` / `xA`: 1,420 occurrences

---

## 5. Statistical Validation & Anti-Hallucination Audit

A core requirement of TouchlineLM is that **every numerical statement must be 100% truthful**.

The automated validator ([src/validation/fact_checker.py](file:///home/athulvr/Projects/Nano%20Gpt/src/validation/fact_checker.py)) parsed every generated text, extracted all numerical assertions via strict regular expressions, and compared them directly against the ground-truth extracted objects:

```text
============================================================
FACTUAL VERIFICATION REPORT
============================================================
Total Samples Audited:       481 / 481 (100.0%)
Total Statistical Claims:    9,350 assertions
Total Verification Failures: 0
Violations Detected:         None
Validation Status:           100.0% PASS
============================================================
```

All 9,350 numerical claims—including complex percentages, progressive counts, xG floats, and scorelines—matched the underlying StatsBomb coordinates and events with zero discrepancies.

---

## 6. Text Diversity and Repetition Analysis

### Lexical Characteristics
- **Domain Focus**: The vocabulary of 1,362 words is heavily concentrated on football-specific analytics. Common sports filler and generic sensationalism are eliminated.
- **Syntactic Templates vs Structural Variation**:
  - In Tier 2 (Player Dossiers), syntactic structures follow consistent analytical sections (Distribution, Attack, Defense), ensuring high predictability for statistical fields.
  - In Tier 1 (Narratives), phase breakdowns vary dynamically based on which team controlled possession, specific minute timestamps of breakthroughs, and extra-time developments.
- **Scaling Recommendation for Phase 3**:
  As we scale from 15 matches to 150+ matches in Phase 3, we recommend introducing a **syntactic template permutation engine** (e.g. 5–8 interchangeable phrasings per statistical insight) to increase trigram diversity from 13.8% to >30%, while preserving 100% deterministic mathematical verification.

---

## 7. Dataset Provenance and Licensing Compliance

- **Primary Source**: StatsBomb Open Data ([github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Mandatory Attribution**:
  > *"This project uses event data provided by StatsBomb (https://statsbomb.com/). StatsBomb Open Data is made available under the Creative Commons Attribution 4.0 International License."*
- **Data Hygiene**:
  - Raw event streams are stored locally in `data/raw/` and excluded from git via `.gitignore`.
  - Processed text samples are licensed under the project's Apache 2.0 license with StatsBomb attribution preserved in `data/metadata/download_manifest.json` and `README.md`.

---

## 8. Transition to Phase 3: Tokenization Strategy

With the corpus successfully engineered and verified, the next phase is **Tokenizer Engineering and Architecture Preparation**:

1. **Tokenizer Architecture Recommendation**:
   - Train a custom **Byte-Pair Encoding (BPE)** tokenizer with a vocabulary size of **$4,096$ to $8,192$** tokens directly on `data/processed/corpus/touchline_corpus.txt`.
   - Rationale: General LLM tokenizers (like GPT-2 or Llama) split specialized football tokens and player names into 3–5 subword fragments (e.g. `StatsBomb`, `xG`, `Tchouaméni`, `Counterpress`). A custom BPE tokenizer will compress these domain concepts into single tokens, drastically increasing the effective context window.
2. **Domain Special Tokens**:
   - Introduce dedicated structural tokens:
     - `<|endoftext|>`: Document boundary.
     - `<|match_narrative|>`: Narrative tier prefix.
     - `<|player_dossier|>`: Scouting dossier prefix.
     - `<|tactical_qa|>`: Analytical reasoning prefix.
     - `<|phase|>`: Temporal window marker.
3. **Artifact Preparation**:
   - Deliver `src/tokenizer/train_tokenizer.py` and produce `tokenizer.json` and `vocab.json`.
