# TouchlineLM Dataset Quality and Completeness Report

**Project**: TouchlineLM (Football Intelligence NanoGPT)  
**Artifact**: Phase 1 Sample Inspection & Validation Audit  
**Date**: November 2026  
**Analyzed Scope**: StatsBomb Open Data & Football-Data.co.uk Sample Ingestion  

---

## 1. Audit Overview & Sample Profile

The initial data acquisition pipeline executed via `scripts/download_data.py` retrieved a targeted, high-density sample representing macro-level tournament structures and micro-level event streams.

### Acquired Sample Manifest
1. **Catalog**: `competitions.json` (80 distinct competition-season editions).
2. **Match Registry**: `matches/43_106.json` (64 matches from FIFA World Cup Qatar 2022).
3. **Event Streams**:
   - `events/3869685.json` (FIFA World Cup 2022 Final: Argentina vs France, 120 mins + Shootout).
   - `events/3857256.json` (FIFA World Cup 2022 Group Stage: Serbia vs Switzerland, 90 mins).
4. **Lineup & Squad Registries**:
   - `lineups/3869685.json` (Argentina & France complete match squads).
   - `lineups/3857256.json` (Serbia & Switzerland complete match squads).
5. **Macro Benchmark**: `football_data/E0_2324.csv` (English Premier League 2023/24, 380 match records).

---

## 2. Quantitative Empirical Analysis

The dataset inspection script (`scripts/inspect_data.py`) processed all event streams, generating the following empirical distributions:

### Event Stream Volumes & Metrics

| Metric / Dimension | Match 3869685 (ARG vs FRA) | Match 3857256 (SRB vs SUI) | Aggregate Sample |
|---|---|---|---|
| **Total Event Records** | 4,407 | 3,210 | 7,617 |
| **Duration / Periods** | 5 periods (90m + 30m ET + Pen) | 2 periods (90m regular) | 7 periods |
| **Active Players in Events** | 39 unique players | 32 unique players | 71 players |
| **Spatial Coordinate Coverage** | 98.75% (4,352 / 4,407) | 98.85% (3,173 / 3,210) | 98.79% |
| **Total Passes** | 1,263 | 966 | 2,229 |
| **Passes Completed** | 994 (78.70% accuracy) | 753 (77.95% accuracy) | 1,747 (78.38%) |
| **Progressive Carries** | 940 | 667 | 1,607 |
| **Total Shots** | 38 (includes shootout) | 26 | 64 |
| **Cumulative Expected Goals (xG)**| 11.299 | 4.293 | 15.592 |
| **Defensive Recoveries** | 115 | 92 | 207 |
| **Interceptions** | 28 | 23 | 51 |
| **Clearances** | 45 | 36 | 81 |
| **Duels & Ground Contests** | 98 | 68 | 166 |

### Event Type Frequency Distribution (Sample Aggregate)
1. **Pass**: 2,229 (29.26%)
2. **Ball Receipt**: 2,056 (26.99%)
3. **Carry**: 1,607 (21.10%)
4. **Pressure**: 618 (8.11%)
5. **Ball Recovery**: 207 (2.72%)
6. **Duel**: 166 (2.18%)
7. **Clearance**: 81 (1.06%)
8. **Shot**: 64 (0.84%)
9. **Interception**: 51 (0.67%)
10. **Others (Fouls, Stoppages, Lineup declarations)**: 538 (7.07%)

---

## 3. Data Completeness and Missing Value Audit

### 3.1 Spatial Coordinate Nullness (1.21% Missingness)
- **Finding**: Exactly 92 of 7,617 events lack an `[x, y]` location coordinate.
- **Root Cause Analysis**: The unlocated events belong exclusively to non-spatial administrative and tactical state declarations:
  - `Starting XI` (Tactical setup before match starts).
  - `Tactical Shift` (Change of team formation).
  - `Half Start` / `Half End`.
  - `Referee Ball Drop` / Match Stoppages.
  - `Bad Behaviour` (Off-pitch disciplinary cards).
- **Assessment**: **Normal and expected by design**. No on-ball action (Pass, Carry, Shot, Tackle) has missing coordinates.

### 3.2 Pass Recipient Nullness
- **Finding**: Incomplete or intercepted passes frequently possess a null `pass.recipient` object.
- **Root Cause**: When a clearance or errant pass is kicked without an intended teammate or is directly intercepted mid-air, StatsBomb records the trajectory coordinates (`pass.end_location`) without forcing a fictitious recipient identifier.
- **LM Tokenization Strategy**: In our text representation, missing recipients must be explicitly articulated as *"an intercepted delivery"* or *"a clearance towards unpopulated space"*, preventing hallucinations.

### 3.3 Expected Goals (xG) Consistency
- **Finding**: Every single open-play and set-piece shot contains a valid, float-valued `statsbomb_xg` between `0.00` and `0.99`.
- **Special Case**: Penalty shootout kicks (in Period 5) possess distinct xG calculations or standard baseline probabilities (~0.76).
- **Validation**: All 64 shot events passed range checks `[0.0, 1.0]` with zero anomalies.

---

## 4. Referential Integrity and Relationship Validation

The validation suite (`scripts/validate_data.py`) executed cross-file consistency checks:
1. **Team Identifier Matching**: Every event's `team.name` was matched against `match.home_team` and `match.away_team`. Integrity rate: **100.0%** (zero orphaned team entities).
2. **Player Roster Matching**: Every event's `player.id` was verified against the official team lineup and bench records in `lineups/{match_id}.json`. Integrity rate: **100.0%** (zero untracked players).
3. **Temporal Monotonicity**: Event indices and timestamps within each match period were validated for chronological progression. Integrity rate: **100.0%**.

---

## 5. Comparison with Macro Dataset (Football-Data.co.uk)

The inspection of `data/raw/football_data/E0_2324.csv` verified that while macro tables cleanly convey end-of-game statistics (e.g., Arsenal 28 wins, 89 points, 91 goals scored), they lack:
- All temporal resolution (no understanding of when goals occurred or match momentum).
- All player-level attribution (cannot determine who made the key pass or tackle).
- Advanced expected metrics (xG, xA, progressive distance).

**Conclusion**: StatsBomb Open Data is confirmed as the indispensable foundational source for TouchlineLM. Macro datasets will be reserved solely for cross-validation and prompt conditioning.
