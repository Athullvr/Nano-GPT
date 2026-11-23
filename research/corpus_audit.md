# TouchlineLM: Comprehensive Corpus Audit & Phase 3 Technical Alignment

**Project**: TouchlineLM (Football Intelligence NanoGPT)  
**Artifact**: Formal Corpus Audit & Architecture Recommendation  
**Date**: November 2026  
**Auditor**: Senior ML Research & Data Engineering Team  

---

## 1. Priority 1: Verification of Expected Assists (xA) Attribution Logic

### 1.1 Empirical Audit of Raw Event Data
Across the 15 ingested FIFA World Cup 2022 matches (58,696 total events), an exhaustive audit of `pass` and `shot` linkage was conducted:
- **Total Passes Evaluated**: 17,122 passes.
- **Passes with `assisted_shot_id`**: **Exactly 251 passes**.
- **Breakdown of Key Passes**:
  - `shot_assist == True` (led to shot, non-goal): **218 passes**.
  - `goal_assist == True` (led to shot, goal scored): **33 passes**.
  - *Sum*: $218 + 33 = 251$ passes ($100.0\%$ mathematical consistency).
- **Total Shots Evaluated**: 376 shots.
- **Shots with `key_pass_id`**: **Exactly 251 shots**.
- **Referential Match Rate**: **251 / 251 (100.0%)**. Exactly 0 orphaned shot assist IDs.
- **Total Cumulative xA**: **24.754 xA** successfully retrieved and attributed across all player records.

### 1.2 xA Methodological Distinction & Limitations
In the football analytics literature, two distinct definitions of Expected Assists exist:
1. **Shot-Assisted xA (Key Pass xA — Implemented Here)**:
   - Assigns to the passer the pre-shot $\text{xG}$ of the immediate shot taken by the receiver.
   - *Advantage*: Directly verifiable from open StatsBomb event graphs without external proprietary models.
   - *Limitation*: Fails to credit the "pass before the assist" (the pre-assist or second assist), and relies on the receiver taking an immediate shot. A world-class through-ball that puts a teammate 1v1 with the goalkeeper yields $0.0\text{ xA}$ if the receiver elects to pass sideways rather than shoot.
2. **Action-Independent Continuous xA / Expected Threat (xT)**:
   - Models the probability of any pass leading to a goal over the next $N$ actions based on pitch coordinates.
   - *Recommendation for Future Phases*: Retain Shot-Assisted xA as our baseline metric, but document it explicitly in model prompts as *"Key Pass xA"* to maintain technical precision.

---

## 2. Priority 2: Progressive Action Definitions & Analytical Limitations

### 2.1 Implemented Formulations
- **Progressive Pass**: A completed pass that reduces the Euclidean distance to the center of the opponent's goal $(120, 40)$ by at least $25\%$, OR delivers the ball into the 18-yard penalty box ($x \ge 102.0, 18.0 \le y \le 62.0$).
- **Progressive Carry**: A continuous ball carry traveling forward towards the goal line by at least $10.0$ yards ($x_{\text{end}} - x_{\text{start}} \ge 10.0$ and $x_{\text{end}} > 40.0$), or entering the penalty box.

### 2.2 Comparison with Industry Standards (Opta / FBref)

| Metric / Dimension | Implemented Formulation | Opta / FBref Standard | Wyscout Standard |
|---|---|---|---|
| **Progressive Pass Volume** | **1,216 passes (8.5% of completed)** | **2,180 passes (15.3% of completed)** | ~1,600 passes (~11%) |
| **Origin Zone Exclusion** | None (Evaluated globally) | Excludes defending 40% ($x < 48$) | Excludes defending 50% |
| **Progression Criterion** | $\ge 25\%$ Euclidean distance reduction | $\ge 10$ yards forward towards goal | 30m / 15m / 10m by third |
| **Penalty Box Delivery** | Automatic qualification | Automatic qualification | Automatic qualification |
| **Progressive Carry Volume** | **1,335 carries (9.9% of carries)** | ~1,450 carries (~10.7%) | N/A |

### 2.3 Limitations of Implemented Geometry
1. **Euclidean Radial Non-Linearity**:
   Because our formula computes $\frac{d_{\text{start}} - d_{\text{end}}}{d_{\text{start}}} \ge 0.25$:
   - A pass starting at midfield $(60, 40)$ has distance $60\text{ yd}$; a $25\%$ reduction requires an advance of **$15.0\text{ yards}$**.
   - A pass starting in the final third $(90, 40)$ has distance $30\text{ yd}$; a $25\%$ reduction requires an advance of **only $7.5\text{ yards}$**.
   - Consequently, our formula is substantially more demanding in midfield than the Opta flat 10-yard rule, classifying fewer passes as progressive (8.5% vs 15.3%).
2. **Defensive Third Long Balls**:
   Without an explicit $x \ge 48$ cutoff, a long clearance from $x=10$ to $x=60$ reduces distance from $110\text{ yd}$ to $60\text{ yd}$ ($45.4\%$ reduction), counting as a progressive pass despite being an uncontrolled clearance.
3. **Correction for Phase 3**:
   Add an explicit defending third exclusion filter: `loc[0] >= 40.0` (or `48.0`) so defensive clearances are not conflated with deliberate attacking progression.

---

## 3. Priority 3: Ground-Truth Audit of Sample Generated Documents

A representative sample of generated texts was audited line-by-line against independent raw JSON queries:

### Case 1: Match 3857300 (Argentina 1 - 2 Saudi Arabia, Group Stage)
- **Raw Event Counts**:
  - Total Shots: Argentina 15 (6 on target), Saudi Arabia 3 (2 on target).
  - Raw Goals: Minute 9 (Messi, pen, $0.783\text{ xG}$), Minute 47 (Al-Shehri, $0.081\text{ xG}$), Minute 52 (Al-Dawsari, $0.033\text{ xG}$).
  - Saudi Arabia Cumulative xG: $0.081 + 0.033 + 0.035 = 0.149 \approx 0.15\text{ xG}$.
- **Narrative Text Verification**:
  - *Reported*: `"Argentina produced 15 total shots (6 on target) accumulating 2.49 expected goals (xG). Conversely, Saudi Arabia recorded... 3 shots (2 on target) for a cumulative total of 0.15 xG."`
  - *Breakdown*: `"at minute 47, Saleh Khalid Al Shehri scored for Saudi Arabia (0.081 xG...); at minute 52, Salem Mohammed Al Dawsari scored for Saudi Arabia (0.033 xG...)"`
  - *Audit Result*: **Exact 100% Match**.

### Case 2: Player Dossier (Kylian Mbappé, Match 3869685 Final)
- **Raw Event Query**:
  - Passes: 20 attempted, 16 completed ($80.0\%$). Progressive carries: 11.
  - Shots: 6 total, 4 on target, 3 goals, $2.44\text{ xG}$ (including extra-time and penalties).
  - Key passes: 2 ($0.65\text{ xA}$).
- **Generated Dossier Text**:
  - *Reported*: 16 of 20 passes ($80.0\%$), 11 progressive carries, 6 shots (4 on target), 3 goals ($2.44\text{ xG}$), 2 key passes ($0.65\text{ xA}$).
  - *Audit Result*: **Exact 100% Match**.

---

## 4. Priority 4: Duplication, Repetition, and Syntactic Entropy

A formal audit of lexical and structural redundancy across all 481 documents revealed:

```text
============================================================
REPETITION & STRUCTURAL DIVERSITY AUDIT
============================================================
Total Documents:                481
Exact Duplicate Documents:      0 (0.0%)
Total Sentences:                4,389
Unique Exact Sentences:         3,588 (81.75%)
Unique Abstracted Patterns:     662 (15.08%)

N-gram Repetition Rates (1 - unique / total):
  1-gram Repetition:            98.75% (Vocabulary: 1,362 words)
  2-gram Repetition:            92.69% (Unique bigrams: 7,964)
  3-gram Repetition:            86.16% (Unique trigrams: 15,073)
  4-gram Repetition:            80.68% (Unique 4-grams: 21,033)
============================================================
```

### Critical Finding: The Slot-Filling Degeneration Risk
- While **zero documents are duplicates** and **81.75% of exact sentences are unique** (because player names and numbers differ), the underlying grammatical skeletons are highly redundant:
  - 451 player dossiers share identical paragraph transitions (`"In terms of forward momentum, the player executed..."`, `"Without the ball, the player was active in defensive phases..."`).
- **Consequence for NanoGPT**:
  An autoregressive Transformer trained repeatedly on this corpus will memorize the boilerplate phrasing within 2 epochs and act merely as a deterministic slot filler.
- **Required Action for Phase 3**:
  Implement a **syntactic permutation engine** featuring:
  - 5 to 8 alternative grammatical phrasings per analytical block.
  - Variable sentence ordering (e.g. leading with defensive impact for defensive midfielders; leading with progressive carries for wingers).
  - Dynamic clause omission (omitting sections with 0 output rather than reciting `"completed 0 crosses, 0 through balls"`).

---

## 5. Priority 5: Token Volume Sufficiency & Scaling Analysis

### 5.1 The Chinchilla & Overfitting Dilemma
The current Phase 2 corpus contains:
- **Total Words**: 108,897 words.
- **Estimated Tokens**: **~141,566 tokens**.

According to empirical neural scaling laws (Hoffmann et al., Chinchilla):
$$\text{Tokens}_{\text{optimal}} \approx 20 \times N_{\text{parameters}}$$

- For a **10M parameter model**: Chinchilla optimal token volume is $\approx 200\text{M tokens}$.
- If we train a 10M–25M parameter model on 140k tokens:
  $$\frac{\text{Tokens}}{\text{Parameters}} = \frac{141,566}{10,000,000} \approx 0.014$$
  The model is starved of data by a factor of $>1,000$. Within 3 to 5 training epochs, training loss will plummet to near-zero while validation loss explodes. The model will overfit completely.

### 5.2 Evaluative Verdict
1. **Can 140k tokens be used for an experiment?**
   - **Yes, as a pipeline verification smoke-test**: To confirm that the tokenizer vocabulary, embedding matrix, causal attention mask, cross-entropy loss, optimizer, and generation loops function properly on hardware.
2. **Is 140k tokens sufficient for a meaningful domain model?**
   - **No**. For the model to genuinely learn semantic associations between tactical roles and player contributions, we must expand the corpus.
3. **Expansion Roadmap**:
   - Extracting all **64 World Cup 2022 matches** + **51 Euro 2024 matches** will expand the corpus to:
     - ~1,200 match-narrative / tactical samples.
     - ~4,500 player dossiers.
     - **~1.5M to 2.0M tokens**, providing a viable foundation for a small 2M–3M parameter Transformer.

---

## 6. Priority 6: Recommended Initial Model Architecture & Context Length

### 6.1 Context Length Determination
Document length analysis of the 481 corpus samples:
- **Median Document Length**: 220 words (~285 tokens).
- **95th Percentile**: 243 words (~315 tokens).
- **Maximum Document Length**: 539 words (~700 tokens, representing the 120-minute World Cup Final with extra time and shootout).

**Recommendation**:
- **Context Length ($T$)**: **1,024 tokens**.
  - Fully accommodates 100% of all generated documents without any truncation.
  - Leaves sufficient headroom for multi-turn prompt conditioning (e.g., prompt instruction + full match narrative + generated tactical evaluation).

### 6.2 Initial Model Architecture Specification (Micro-NanoGPT)

To prevent severe overfitting on our initial dataset while keeping training fast and fully reproducible on CPU or consumer GPU:

```text
======================================================================
MICRO-NANOGPT ARCHITECTURAL SPECIFICATION
======================================================================
Architecture Type:              Decoder-Only Causal Transformer
Target Parameter Count:         ~3.2 Million Parameters
Context Window Length (T):      1,024 Tokens
Vocabulary Size (V):            4,096 Tokens (Custom BPE Tokenizer)
Embedding Dimension (d_model):  256
Number of Transformer Layers:   4 Layers
Number of Attention Heads:      4 Heads (Head Dimension: 64)
Feed-Forward Dimension (d_ff):  1,024 (4 * d_model)
Activation Function:            GELU (or SwiGLU)
Positional Encoding:            Rotary Position Embeddings (RoPE)
Normalization:                  RMSNorm / LayerNorm (Pre-Norm)
Dropout Rate:                   0.1 (High regularization for small corpus)
Weight Decay:                   0.1
======================================================================
```

### Parameter Calculation Breakdown
1. **Token Embeddings**: $4,096 \times 256 = 1,048,576$ parameters.
2. **Attention Blocks (4 layers)**: $4 \times (4 \times 256^2) = 1,048,576$ parameters.
3. **Feed-Forward Blocks (4 layers)**: $4 \times (2 \times 256 \times 1,024) = 2,097,152$ parameters.
4. **Norms and Projections**: $\approx 20,000$ parameters.
5. **Total Parameters**: $\approx 4.2\text{M}$ (with untied lm_head) or $\approx 3.2\text{M}$ (with weight-tied embeddings).

This 3.2M parameter scale allows the model to:
- Learn the syntax of football intelligence without immediate memorization.
- Train in under 15 minutes on a modern laptop or free-tier GPU.
- Serve as the ideal stepping stone before scaling to the 25M model once the corpus is expanded to 5M+ tokens.

---

## 7. Required Corrections and Next Steps Summary

Before launching Phase 3 training:
1. **Metric Engine Refinement**: Add `loc[0] >= 40.0` exclusion to the progressive pass calculator to eliminate defensive clearances from attacking metrics.
2. **Permutation Engine**: Introduce 5–8 interchangeable phrasings in `src/corpus/generator.py` to break rigid syntactic n-gram repetition.
3. **Corpus Expansion**: Execute batch extraction across the full 64 matches of World Cup 2022 to scale the token budget from 140k to ~600k tokens before model training.
4. **Tokenizer Implementation**: Train the custom 4,096-vocabulary BPE tokenizer in `src/tokenizer/`.
