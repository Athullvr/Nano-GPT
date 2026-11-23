# Unresolved Technical Decisions & Decision Matrix

**Project**: TouchlineLM (Football Intelligence NanoGPT)  
**Phase**: 1 - Pre-Phase 2 Alignment Document  
**Date**: November 2026  

---

This document outlines open technical decisions that require architectural alignment and user review prior to transitioning to Phase 2 (Data Processing & Tokenization Pipeline).

---

## 1. Open Architectural Decisions

### Decision 1: Model Sizing and Context Window Architecture
- **Options**:
  - **Option A (Micro-Nano, ~10M params)**: 6 layers, 6 heads, 384 embedding dim, 512 context length. (Extremely fast iteration on consumer GPU or CPU; ideal for rapid prototyping).
  - **Option B (Standard Nano, ~25M-30M params)**: 8 layers, 8 heads, 512 embedding dim, 1024 context length. (Sufficient capacity to memorize player styles and tactical structures while fitting complex narratives in context).
  - **Option C (Extended Nano, ~50M-85M params)**: 12 layers, 12 heads, 768 embedding dim, 1024-2048 context length. (Higher natural language fluidity, requires dedicated GPU training).
- **Engineering Recommendation**: **Option B (~25M-30M parameters, 1024 context)**. Striking the optimal balance between educational transparency and generative fluency.

---

### Decision 2: Tokenizer Strategy (Domain Custom vs General Transfer)
- **Options**:
  - **Option A (Custom Domain BPE Tokenizer)**: Train a custom Byte-Pair Encoding tokenizer (vocab size 4,096 to 8,192) directly on our football intelligence corpus.
    - *Pros*: Extreme compression of specialized football tokens (`[xG:`, `[ARG]`, `Interception`, `Half-Space`, player surnames).
    - *Cons*: Weak general English syntax outside the sports domain.
  - **Option B (GPT-2 Tokenizer with Added Football Special Tokens)**: Use standard 50k GPT-2 BPE tokenizer augmented with domain special tokens (`<|event|>`, `<|phase|>`, `<|shot|>`).
    - *Pros*: Immediate transfer of general English grammar and morphology.
    - *Cons*: Larger embedding table footprint relative to model size.
- **Engineering Recommendation**: **Option A with a targeted 8,192 vocabulary**, preserving high token density per match sequence.

---

### Decision 3: Scale of Training Ingestion for Phase 2
- **Options**:
  - **Tier 1 (Curated Benchmark, ~150 matches)**: FIFA World Cup 2022 (64 matches), UEFA Euro 2024 (51 matches), select UEFA Champions League finals (~35 matches). (~1.5M tokens).
  - **Tier 2 (Full International + Modern Tournaments, ~500 matches)**: World Cups 2018 & 2022, Euros 2020 & 2024, FA WSL seasons, Champions League matches. (~5M tokens).
  - **Tier 3 (Complete Open Archive, ~1,500+ matches)**: Ingesting the entire StatsBomb public repository including Messi's complete 17-season La Liga career. (~15M+ tokens).
- **Engineering Recommendation**: Begin Phase 2 with **Tier 1**, establishing a validated, leak-free pipeline before scaling to Tier 2 and Tier 3.

---

### Decision 4: Narrative Synthesis Pipeline (Deterministic vs Hybrid Distillation)
- **Options**:
  - **Option A (Pure Deterministic Python Templating)**: Construct sentences strictly using algorithmic grammars, rule trees, and deterministic slot filling.
    - *Pros*: 0% hallucination risk, fully verifiable, zero external model dependencies.
    - *Cons*: Risk of repetitive syntactic cadence.
  - **Option B (Hybrid Teacher Distillation with Strict Assertion Guardrails)**: Use a larger teacher model offline to generate diverse linguistic variations, passed through an automated fact-checking validator that rejects any text where metrics deviate from raw JSON.
    - *Pros*: Rich, natural journalistic prose.
    - *Cons*: Requires external inference execution and rigorous validation filters.
- **Engineering Recommendation**: **Option A as the foundational baseline**, with systematic linguistic variation rules, evaluating Option B as an optional augmentation.

---

### Decision 5: Quantitative Evaluation Framework
- **Options**:
  - **Perplexity on Held-Out Tournaments**: Measure next-token prediction loss on unseen matches (e.g., train on Euro 2020, test on Euro 2024).
  - **Domain Probing Benchmarks**: Construct a 200-question multiple-choice benchmark evaluating:
    1. Player positioning and roles.
    2. Tactical outcome attribution (which team controlled the xG).
    3. Metric reasoning (identifying progressive actions and defensive contributors).
- **Engineering Recommendation**: Implement **both** held-out tournament perplexity and domain probing benchmarks for rigorous evaluation.
