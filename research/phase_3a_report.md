# TouchlineLM — Phase 3A: Corpus Refinement and Expansion Report

**Project**: TouchlineLM (Football Intelligence NanoGPT)  
**Phase**: 3A - Corpus Refinement, Full World Cup Ingestion & Match-Level Splitting  
**Date**: November 2026  
**Auditor**: Senior ML Research & Data Engineering Team  

---

## 1. Executive Summary

Phase 3A completes the refinement, scaling, and structured preparation of the TouchlineLM pre-training dataset. In accordance with user guidance:
1. **Metric Logic Refined**: Deep defensive clearances and clearance event types are strictly excluded from progressive pass calculations; clearances remain an independent defensive metric; Key Pass xA is retained with verified `assisted_shot_id` attribution.
2. **Syntactic Diversity Enhanced**: A deterministic sentence permutation engine with controlled modular seeds was introduced, accompanied by dynamic omission of empty zero-value sections.
3. **Full Tournament Scale Ingested**: All **64 matches of the FIFA World Cup 2022** were ingested, expanding the dataset from 15 to 64 fixtures (**234,637 raw events**, **3,244 player records**).
4. **Baseline Tokenization & Match-Level Splitting**: Using `tiktoken` (GPT-2 baseline encoding), the corpus reaches **738,352 actual tokens** (456,182 words). Strict match-level grouping was enforced across Train (47 matches), Validation (8 matches), and Test (9 matches) with **zero match leakage**.
5. **Quality Assurance Verified**: **38,448 factual claims** were verified against ground truth with **0 violations (100.0% pass rate)**. All 10 automated test suite checks pass cleanly.

---

## 2. Metric Corrections & Technical Logic

### 2.1 Progressive Pass Refinements
- **Defensive Third Exclusion**: In `src/metrics/calculator.py`, `is_progressive_pass` now enforces `start_x >= 40.0`. Completed forward deliveries originating inside the team's defensive 33% ($x < 40.0$ yards) are excluded from progressive pass metrics, preventing long defensive punts from inflating attacking progression.
- **Clearance Type Filtering**: Any pass tagged with a clearance pass type (`pass_type and "clearance" in pass_type.lower()`) is rejected.
- **Defensive Clearances Kept Separate**: Emergency danger clearances remain cleanly tracked in `t_metrics.clearances` and `p_metrics.clearances` (derived from `e_type == "Clearance"`).

### 2.2 Progressive Carry Logic & Limitations
- **Current Formulation**: A continuous ball carry advancing forward towards the opponent goal line by at least $10.0$ yards ($\Delta x \ge 10.0$ and $x_{\text{end}} > 40.0$ and $x_{\text{start}} \ge 35.0$), or entering the opponent penalty box.
- **Analytical Limitations**:
  1. *Linear Axis Bias*: Measures displacement purely along the longitudinal $x$-axis. A diagonal carry across the half-space covering 14 yards of ground with $\Delta x = 9.5\text{ yd}$ is narrowly excluded.
  2. *Opposition Pressure Blindspot*: Does not currently condition on whether the carry bypassed an opponent defensive line or was carried through unoccupied territory.

### 2.3 Key Pass xA Retention
- Retained the verified `assisted_shot_id` relationship: exactly maps each shot-creating pass to the resultant shot's `statsbomb_xg`. Audited 251 / 251 passes ($100.0\%$ match rate, $0$ orphaned IDs).

---

## 3. Improved Corpus Generation with Controlled Variations

The generator (`src/corpus/generator.py`) was upgraded to eliminate rigid template patterns while preserving 100% deterministic reproducibility:
1. **Deterministic Seeding**: Phrasings are selected using modular arithmetic hashes of `(match_id, player_id)`, ensuring identical text generation across repeated runs without random divergence.
2. **Multiple Phrasings for Analytical Blocks**:
   - 4 interchangeable syntactic frames for distribution and progression.
   - 3 interchangeable contextual frames for macroscopic match summaries and chronological phases.
   - Dual-mode framing for defensive duels and recoveries.
3. **Dynamic Zero-Omission**:
   - Zero-shot players: Rather than reciting `"registered 0 total shots (0 on target), scoring 0 goals from 0.0 xG"`, the engine cleanly generates `"In offensive situations, {player} did not register any direct shots on goal, recording 0 total shots (0 on target)..."`.
   - Zero-cross players: Replaces empty `"completed 0 crosses, 0 through balls"` with `"focused on ground progression, recording {progressive_carries} progressive carries... without delivering crosses or through balls"`.
   - Clean discipline: Players with 0 cards and 0 fouls are described as having `"maintained a clean sheet with 0 fouls and 0 bookings"` rather than reciting multiple zeroes.

---

## 4. Full Dataset Expansion Metrics (64 Matches)

```text
=================================================================
PHASE 3A CORPUS DIMENSIONS (FULL WORLD CUP 2022)
=================================================================
Total Fixtures Ingested:          64 / 64 (100.0% of Tournament)
Total Raw Events Processed:       234,637 events
Total Extracted Player Records:   3,244 records (data/intermediate/)
Total Generated Documents:        1,980 documents (data/processed/corpus/)
  - Tier 1 (Match Narratives):    64 documents
  - Tier 2 (Player Dossiers):     1,852 documents
  - Tier 3 (Tactical Q&A):        64 documents

Total Words:                      456,182 words
Total Characters:                 3,162,362 characters
Actual Baseline Tokens (GPT-2):   738,352 tokens
Tokens-to-Words Ratio:            1.619 tokens / word
Unique Domain Vocabulary:         2,571 unique words
=================================================================
```

---

## 5. Match-Level Dataset Splitting (Zero Leakage)

To ensure rigorous ML evaluation, documents were partitioned **strictly by match ID** stratified across tournament stages. No match appears in more than one split:

| Split | Match Count | Document Count | Total Words | Actual GPT-2 Tokens | Token % |
|---|---|---|---|---|---|
| **Train** | **47 matches** | **1,445 docs** | 332,752 words | **539,066 tokens** | **73.0%** |
| **Validation** | **8 matches** | **247 docs** | 57,128 words | **93,328 tokens** | **12.6%** |
| **Test** | **9 matches** | **288 docs** | 66,252 words | **105,958 tokens** | **14.4%** |
| **Total** | **64 matches** | **1,980 docs** | **456,182 words** | **738,352 tokens** | **100.0%** |

### Split Artifacts Generated
- `data/processed/splits/train.jsonl` & `train.txt`
- `data/processed/splits/val.jsonl` & `val.txt`
- `data/processed/splits/test.jsonl` & `test.txt`
- Metadata preserved in [data/metadata/splits_manifest.json](file:///home/athulvr/Projects/Nano%20Gpt/data/metadata/splits_manifest.json).

---

## 6. Quality Assurance & Diversity Audit

### 6.1 Automated Fact-Checking Verification
The factual verification suite (`src/validation/fact_checker.py`) verified every single numerical claim across all 1,980 generated documents:
- **Total Claims Audited**: **38,448 numerical assertions**.
- **Discrepancies / Violations**: **0**.
- **Verification Status**: **100.0% PASS**.

### 6.2 Lexical & Structural Diversity Comparison
Comparing the Phase 2 baseline (15 matches, single template) against Phase 3A (64 matches, syntactic permutations):

| Diversity Metric | Phase 2 Baseline (15 Matches) | Phase 3A Refined (64 Matches) | Relative Improvement |
|---|---|---|---|
| **Unique Domain Words** | 1,362 | **2,571** | **+88.8%** |
| **Unique Sentence Patterns** | 662 | **2,153** | **+225.2%** |
| **Distinct Trigrams** | 15,073 | **41,742** | **+176.9%** |
| **Distinct 4-grams** | 21,033 | **62,038** | **+195.0%** |
| **Exact Duplicate Documents** | 0 (0.0%) | **0 (0.0%)** | Zero duplicates maintained |

### 6.3 Automated Test Suite
All 10 unit and integration tests in `tests/test_corpus_pipeline.py` and `tests/test_data_integrity.py` pass:
```text
tests/test_corpus_pipeline.py::test_distance_and_box_geometry PASSED     [ 10%]
tests/test_corpus_pipeline.py::test_progressive_action_logic PASSED      [ 20%]
tests/test_corpus_pipeline.py::test_match_extraction_and_metrics PASSED  [ 30%]
tests/test_corpus_pipeline.py::test_corpus_generation_and_validation PASSED [ 40%]
tests/test_corpus_pipeline.py::test_splits_no_match_leakage PASSED       [ 50%]
tests/test_data_integrity.py::test_manifest_existence PASSED             [ 60%]
tests/test_data_integrity.py::test_statsbomb_competitions_exist PASSED   [ 70%]
tests/test_data_integrity.py::test_statsbomb_matches_exist PASSED        [ 80%]
tests/test_data_integrity.py::test_statsbomb_events_exist_and_valid PASSED [ 90%]
tests/test_data_integrity.py::test_validator_passes PASSED               [100%]
10 passed in 1.85s
```

---

## 7. Recommendation for Phase 3B (Tokenizer & Model Architecture)

With **738,352 baseline tokens** partitioned into leak-free match splits, the dataset is in a sound state to transition to **Phase 3B**:

1. **Step 1: Domain Tokenizer Implementation (`src/tokenizer/`)**:
   - Train a custom **Byte-Pair Encoding (BPE)** tokenizer with a vocabulary size of **$4,096$ tokens** directly on `data/processed/splits/train.txt`.
   - Include special tokens: `<|endoftext|>`, `<|narrative|>`, `<|player_dossier|>`, `<|tactical_qa|>`.
   - Save tokenizer configuration and vocabulary files (`tokenizer.json`, `vocab.json`).
2. **Step 2: Micro-NanoGPT Architecture (`src/model/`)**:
   - Implement the causal decoder-only Transformer in PyTorch:
     - 4 Layers, 4 Attention Heads ($d_{\text{head}} = 64$, $d_{\text{model}} = 256$, $d_{\text{ff}} = 1024$).
     - Rotary Position Embeddings (RoPE).
     - Context window: 1,024 tokens.
     - Parameter count: ~3.2M parameters.
3. **Step 3: Pre-Training & Learning Dynamics (`src/training/`)**:
   - Train on `train.txt` (539k tokens), evaluating validation loss on `val.txt` (93k tokens) every 250 steps.
