# Football Intelligence Dataset Comparison Report

**Project**: TouchlineLM (Football Intelligence NanoGPT)  
**Author**: ML Research & Data Engineering Team  
**Phase**: 1 - Dataset Research and Acquisition  
**Date**: November 2026  

---

## 1. Executive Summary

To train a domain-specific decoder-only language model (**TouchlineLM**) capable of reasoning over football tactics, advanced metrics (xG, xA, progressive actions), and match narratives, the underlying corpus must capture granular, event-level dynamics rather than aggregated scorelines.

This report evaluates four major public football data ecosystems against the core requirements of TouchlineLM:
1. **StatsBomb Open Data** (Recommended Primary Source)
2. **Football-Data.co.uk** (Evaluated Macro Benchmark)
3. **Wyscout Open Soccer Dataset** (Pappalardo et al., Scientific Data 2019)
4. **Understat / Kaggle Scraped Repositories**

---

## 2. Detailed Dataset Profiles

### 2.1 StatsBomb Open Data (GitHub)

- **Source & Repository**: [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) with explicit commercial attribution guidelines.
- **Data Format**: Standardized JSON documents organized into hierarchical trees:
  - `competitions.json` (Competition catalog)
  - `matches/{competition_id}/{season_id}.json` (Match metadata)
  - `events/{match_id}.json` (Fine-grained event stream)
  - `lineups/{match_id}.json` (Squads, jersey numbers, positions)
  - `three-sixty/{match_id}.json` (Computer vision player tracking freeze frames for select competitions)
- **Available Competitions & Seasons**:
  - 80 competition-season combinations available.
  - Major Men's Tournaments: FIFA World Cup (2018, 2022), UEFA Euro (2020, 2024), UEFA Champions League (select historical finals & seasons).
  - Domestic Leagues: Premier League (Arsenal 2003/04 Invincibles), La Liga (Complete Lionel Messi league career: 2004/05 through 2020/21, >500 matches).
  - Women's Tournaments: FIFA Women's World Cup (2019, 2023), FA Women's Super League, NWSL.
- **Volume & Granularity**:
  - Thousands of matches; each match contains between 3,000 and 4,500 distinct events.
  - Total events in open repository exceed 4 million.
- **Available Event Types**:
  - **Passing**: Progressive distance, length, angle, pass recipient, body part, pass type (cross, through ball, corner, throw-in), switch, under-pressure indicator, height (ground, low, high).
  - **Goal Scoring**: Shots with calibrated `statsbomb_xg` (Expected Goals), shot technique (volley, half-volley, lob), body part, outcome (Goal, Saved, Blocked, Off Target, Wayward, Post), goalkeeper positioning.
  - **Ball Progression**: Carries (progressive distance, start and end coordinates).
  - **Defensive Actions**: Duels (tackles, aerial contests), Interceptions, Clearances, Blocks, Ball Recoveries, Pressures, Counterpress flags.
  - **Discipline & Stoppages**: Fouls committed, fouls won, yellow/red cards, injury stoppages, VAR delays.
  - **Tactical**: Starting formations (e.g., 4-3-3, 4-2-3-1, 3-5-2), tactical substitutions, real-time tactical shifts, player tactical roles.
- **Player-Level Information**:
  - Full player name, nickname, player ID, jersey number, country of birth.
  - Explicit starting position and minute-by-minute positional adjustments.
- **Missing or Incomplete Fields**:
  - Continuous 25Hz tracking coordinates are not open (only event locations and 360 freeze frames are provided).
  - Physical biometric data (distance covered in km, top sprint speed) is not present.
- **Usefulness for TouchlineLM**: **Highest**. Provides the foundational mathematical and tactical richness required to synthesize natural language tactical breakdowns, progressive action analyses, and player performance narratives.

---

### 2.2 Football-Data.co.uk

- **Source**: [football-data.co.uk](https://www.football-data.co.uk/)
- **License**: Freely accessible for research/personal use; commercial use requires explicit permission.
- **Data Format**: Flat CSV files segmented by league code and season year (e.g., `E0_2324.csv` for English Premier League 2023–2024).
- **Available Competitions & Seasons**:
  - Major European leagues (England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Greece, Turkey).
  - Multi-decade historical coverage (from 1993/94 to present).
- **Volume & Granularity**:
  - 380 rows per 20-team league season (~100,000 match records across all historical tiers).
  - 106 columns per row in recent seasons.
- **Available Metrics**:
  - Final full-time score (FTHG, FTAG) and half-time score (HTHG, HTAG).
  - Match-level macro metrics: Home/Away total shots (HS, AS), shots on target (HST, AST), corners (HC, AC), fouls (HF, AF), yellow cards (HY, AY), red cards (HR, AR).
  - Extensive betting market odds from 10+ bookmakers (Bet365, Pinnacle, Betway, William Hill) including opening, closing, Asian handicaps, and over/under 2.5 goals.
- **Player-Level Information**: **None**. No rosters, player lineups, goalscorer names, or individual performance figures.
- **Missing Fields for LM Training**:
  - Zero event timestamps.
  - Zero spatial pitch coordinates.
  - Zero individual player identification.
  - Zero expected metrics (no xG, xA, progressive actions, or defensive duels).
- **Usefulness for TouchlineLM**: **Low for core generative modeling**. Valuable solely as a macro reference table for validation of final scores, league tables, and macroscopic outcome conditioning.

---

### 2.3 Wyscout Open Soccer Dataset (Pappalardo et al.)

- **Source**: Nature Scientific Data (2019) / Figshare / Kaggle
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Data Format**: JSON files covering matches, events, players, coaches, and competitions.
- **Available Competitions & Seasons**:
  - 2017/2018 season across 5 top European leagues (England, Spain, Italy, Germany, France) plus 2018 World Cup and Euro 2016.
  - 1,941 matches and 3,251,294 events.
- **Available Event Types**:
  - Pass, Duel, Shot, Free Kick, Others on the ball, Foul, Offside.
  - Normalized pitch coordinates `[0, 100]` for length and width.
  - Sub-event tags (e.g., accurate/inaccurate, opportunity, key pass).
- **Player-Level Information**:
  - Comprehensive player registry with names, birth dates, roles, and passport country.
- **Missing or Incomplete Fields**:
  - Proprietary xG values are not included in the open release (must be estimated from raw shot features).
  - Event hierarchy and sub-attributes are less granular than StatsBomb (e.g., fewer pass trajectory types, no freeze frames).
  - Confined predominantly to a single historical season (2017/2018).
- **Usefulness for TouchlineLM**: **High as a supplementary validation corpus**, but second to StatsBomb in analytical depth.

---

### 2.4 Understat / Kaggle Scraped Datasets

- **Source**: Various GitHub scrapers and Kaggle community uploads derived from Understat.com.
- **License**: Unclear / Gray-area scraping. Understat terms do not officially grant bulk redistribution rights.
- **Data Format**: CSV and JSON.
- **Available Metrics**:
  - Shot-level data with xG, coordinates, shooter identity, assist identity, shot type, situation (open play, corner, set piece).
  - Match-level aggregated xG and expected points (xPTS).
- **Player-Level Information**: Season totals for goals, assists, xG, xA, key passes, shots per 90.
- **Missing Fields**:
  - No defensive events (no tackles, interceptions, clearances, pressures).
  - No passing sequences outside of key passes and assists.
  - Gray-area intellectual property rights make it risky for open-source AI reproducibility.
- **Usefulness for TouchlineLM**: **Not recommended** due to provenance ambiguity and lack of defensive event data.

---

## 3. Comparative Capability Matrix

| Feature / Dimension | StatsBomb Open Data | Football-Data.co.uk | Wyscout Open Data | Understat Scraped |
|---|---|---|---|---|
| **License Transparency** | Clear CC BY 4.0 | Free non-commercial | CC BY 4.0 | Unclear / Scraped |
| **Data Format** | JSON (Nested Schema) | CSV (Flat Table) | JSON (Nested) | CSV / JSON |
| **Total Events per Match** | 3,000 - 4,500 | None (0) | 1,500 - 2,000 | 20 - 35 (Shots only) |
| **Spatial Coordinates** | Continuous (0-120 x 0-80) | None | Normalized (0-100) | Normalized (0-1) |
| **Timestamps & Periods** | Minute, Second, Period | Match Date only | Minute, Second, Period | Minute only |
| **Expected Goals (xG)** | Built-in calibrated `statsbomb_xg` | None | None (Requires calculation) | Scraped Model xG |
| **Progressive Passes / Carries**| Explicitly derivable from coords | Impossible | Derivable | Impossible |
| **Defensive Micro-Actions** | Duels, Interceptions, Clearances, Pressures | Team totals only (Fouls/Cards) | Duels, Fouls | None |
| **Tactical Lineups & Formations**| Explicit formations & positions | None | Starting XI only | Starting XI only |
| **Suitability for TouchlineLM**| **Primary (Tier 1)** | **Reference (Tier 3)** | **Secondary (Tier 2)** | **Excluded** |

---

## 4. Phase 1 Acquisition Recommendation

### Primary Choice: StatsBomb Open Data
1. **Unrivaled Analytical Granularity**: StatsBomb provides native `statsbomb_xg`, pass freeze-frames, pressure metrics, and exact coordinate paths for both passes and progressive carries.
2. **Legal and Ethical Provenance**: Openly published under CC BY 4.0 with explicit guidelines.
3. **Domain Coverage**: Covers world-class football benchmarks (FIFA World Cup 2022, UEFA Euro 2024, Messi La Liga career, Arsenal Invincibles) allowing both high-stakes tournament modeling and individual superstar multi-season longitudinal studies.
4. **Actionable Tokenization**: The nesting of action types, recipient names, pitch zones, and timestamps allows systematic serialization into domain-specific narrative sentences without hallucinating stats.

### Secondary Reference: Football-Data.co.uk
- Maintained in `data/raw/football_data/` strictly to validate macroscopic match results and provide cross-league scoreline distributions.
