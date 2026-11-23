"""
TouchlineLM - Data Inspection Script (Phase 1)
Deeply inspects raw datasets (StatsBomb JSON & Football-Data CSV),
evaluates entity relationships, record counts, event distributions,
spatial coordinate coverage, and missingness.
Outputs an inspection summary to stdout and data/metadata/inspection_summary.json.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inspect_data")


def inspect_statsbomb(raw_dir: Path) -> Dict[str, Any]:
    """Inspect StatsBomb files: competitions, matches, events, and lineups."""
    sb_dir = raw_dir / "statsbomb"
    if not sb_dir.exists():
        logger.warning(f"StatsBomb directory not found: {sb_dir}")
        return {}

    summary: Dict[str, Any] = {"competitions": {}, "matches": {}, "events_analysis": []}

    # 1. Competitions
    comp_file = sb_dir / "competitions.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comps = json.load(f)
        summary["competitions"]["total_available"] = len(comps)
        summary["competitions"]["sample_entries"] = [
            {
                "competition_id": c.get("competition_id"),
                "competition_name": c.get("competition_name"),
                "season_id": c.get("season_id"),
                "season_name": c.get("season_name"),
                "match_available": c.get("match_available")
            }
            for c in comps[:5]
        ]
        logger.info(f"Loaded {len(comps)} competitions from {comp_file.name}")

    # 2. Matches
    matches_dir = sb_dir / "matches"
    match_lookup = {}
    if matches_dir.exists():
        for mf in matches_dir.glob("*.json"):
            with open(mf, "r", encoding="utf-8") as f:
                matches_data = json.load(f)
            summary["matches"][mf.name] = {
                "match_count": len(matches_data),
                "matches": [
                    {
                        "match_id": m.get("match_id"),
                        "match_date": m.get("match_date"),
                        "home_team": m.get("home_team", {}).get("home_team_name"),
                        "away_team": m.get("away_team", {}).get("away_team_name"),
                        "home_score": m.get("home_score"),
                        "away_score": m.get("away_score"),
                        "competition_stage": m.get("competition_stage", {}).get("name")
                    }
                    for m in matches_data
                ]
            }
            for m in matches_data:
                match_lookup[m.get("match_id")] = m

    # 3. Events & Lineups
    events_dir = sb_dir / "events"
    lineups_dir = sb_dir / "lineups"

    if events_dir.exists():
        for ef in sorted(events_dir.glob("*.json")):
            m_id = int(ef.stem)
            with open(ef, "r", encoding="utf-8") as f:
                events = json.load(f)

            match_meta = match_lookup.get(m_id, {})
            home_team = match_meta.get("home_team", {}).get("home_team_name", "Unknown Home")
            away_team = match_meta.get("away_team", {}).get("away_team_name", "Unknown Away")

            # Analyze events
            type_counter = Counter(e.get("type", {}).get("name") for e in events)
            players = {e.get("player", {}).get("name") for e in events if e.get("player")}
            teams = {e.get("team", {}).get("name") for e in events if e.get("team")}

            # Specific football intelligence metrics
            shot_events = [e for e in events if e.get("type", {}).get("name") == "Shot"]
            pass_events = [e for e in events if e.get("type", {}).get("name") == "Pass"]
            carry_events = [e for e in events if e.get("type", {}).get("name") == "Carry"]
            interceptions = [e for e in events if e.get("type", {}).get("name") == "Interception"]
            ball_recoveries = [e for e in events if e.get("type", {}).get("name") == "Ball Recovery"]
            clearances = [e for e in events if e.get("type", {}).get("name") == "Clearance"]
            duels = [e for e in events if e.get("type", {}).get("name") == "Duel"]

            # Shot analysis (xG, outcomes)
            xg_values = [
                s.get("shot", {}).get("statsbomb_xg")
                for s in shot_events
                if s.get("shot", {}).get("statsbomb_xg") is not None
            ]
            shot_outcomes = Counter(s.get("shot", {}).get("outcome", {}).get("name") for s in shot_events)

            # Pass outcome & types
            pass_types = Counter(p.get("pass", {}).get("type", {}).get("name", "Regular") for p in pass_events)
            pass_outcomes = Counter(p.get("pass", {}).get("outcome", {}).get("name", "Complete") for p in pass_events)

            # Spatial coordinates coverage
            events_with_location = sum(1 for e in events if "location" in e and e["location"])
            location_coverage_pct = round((events_with_location / len(events)) * 100, 2) if events else 0

            # Lineup verification
            lineup_file = lineups_dir / f"{m_id}.json"
            lineup_info = {}
            if lineup_file.exists():
                with open(lineup_file, "r", encoding="utf-8") as lf:
                    lineups_data = json.load(lf)
                lineup_info = {
                    team.get("team_name"): len(team.get("lineup", []))
                    for team in lineups_data
                }

            match_summary = {
                "match_id": m_id,
                "fixture": f"{home_team} vs {away_team}",
                "total_events": len(events),
                "unique_players_in_events": len(players),
                "teams": list(teams),
                "lineup_roster_sizes": lineup_info,
                "spatial_location_coverage_pct": location_coverage_pct,
                "tactical_breakdown": {
                    "passes": len(pass_events),
                    "passes_completed": pass_outcomes.get("Complete", 0),
                    "pass_types_top": dict(pass_types.most_common(5)),
                    "carries": len(carry_events),
                    "shots": len(shot_events),
                    "shot_xg_recorded_count": len(xg_values),
                    "total_xg_sum": round(sum(xg_values), 3),
                    "shot_outcomes": dict(shot_outcomes),
                    "defensive_actions": {
                        "interceptions": len(interceptions),
                        "ball_recoveries": len(ball_recoveries),
                        "clearances": len(clearances),
                        "duels_tackles": len(duels)
                    }
                },
                "event_type_distribution": dict(type_counter.most_common(12))
            }
            summary["events_analysis"].append(match_summary)
            logger.info(f"Analyzed match {m_id} ({home_team} vs {away_team}): {len(events)} events, {len(shot_events)} shots, total xG: {round(sum(xg_values), 2)}")

    return summary


def inspect_football_data_uk(raw_dir: Path) -> Dict[str, Any]:
    """Inspect Football-Data.co.uk CSV files."""
    fd_dir = raw_dir / "football_data"
    if not fd_dir.exists():
        logger.warning(f"Football-Data directory not found: {fd_dir}")
        return {}

    summary: Dict[str, Any] = {"files": {}}
    for csv_file in fd_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            summary["files"][csv_file.name] = {
                "total_rows_matches": len(df),
                "total_columns": len(df.columns),
                "columns_list": list(df.columns),
                "teams": list(set(df["HomeTeam"].dropna().tolist() + df["AwayTeam"].dropna().tolist())),
                "sample_record": df.head(1).to_dict(orient="records")[0] if not df.empty else {},
                "missing_values_per_col": df.isnull().sum().to_dict(),
                "available_metrics": {
                    "has_event_timestamps": False,
                    "has_spatial_coordinates": False,
                    "has_player_identities": False,
                    "has_xg": False,
                    "has_match_score": "FTHG" in df.columns and "FTAG" in df.columns,
                    "has_shots": "HS" in df.columns and "AS" in df.columns,
                    "has_fouls_and_cards": "HF" in df.columns and "HY" in df.columns,
                    "has_betting_odds": any("Avg" in c or "B365" in c for c in df.columns)
                }
            }
            logger.info(f"Loaded Football-Data CSV {csv_file.name}: {len(df)} matches, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            summary["files"][csv_file.name] = {"error": str(e)}

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect downloaded football datasets.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to data/raw directory")
    parser.add_argument("--output-json", type=str, default="data/metadata/inspection_summary.json",
                        help="Path to output summary JSON")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dataset inspection...")
    sb_summary = inspect_statsbomb(raw_dir)
    fd_summary = inspect_football_data_uk(raw_dir)

    full_inspection = {
        "statsbomb": sb_summary,
        "football_data_uk": fd_summary
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_inspection, f, indent=2)

    logger.info(f"Inspection summary saved to {output_json}")

    # Print high-level comparison
    print("\n" + "=" * 60)
    print("DATASET INSPECTION & COMPARISON SUMMARY")
    print("=" * 60)

    if "events_analysis" in sb_summary and sb_summary["events_analysis"]:
        print("\n--- STATSBOMB OPEN DATA (Event-Level) ---")
        for match in sb_summary["events_analysis"]:
            print(f"Match: {match['fixture']} (ID: {match['match_id']})")
            print(f"  Total Events: {match['total_events']}")
            print(f"  Location Coverage: {match['spatial_location_coverage_pct']}%")
            tb = match["tactical_breakdown"]
            print(f"  Passes: {tb['passes']} (Completed: {tb['passes_completed']})")
            print(f"  Carries: {tb['carries']}")
            print(f"  Shots: {tb['shots']} (Total xG: {tb['total_xg_sum']})")
            da = tb["defensive_actions"]
            print(f"  Defensive Actions: Recoveries={da['ball_recoveries']}, Interceptions={da['interceptions']}, Clearances={da['clearances']}, Duels/Tackles={da['duels_tackles']}")

    if "files" in fd_summary and fd_summary["files"]:
        print("\n--- FOOTBALL-DATA.CO.UK (Macro Match-Level) ---")
        for fname, info in fd_summary["files"].items():
            if "total_rows_matches" in info:
                print(f"File: {fname}")
                print(f"  Matches: {info['total_rows_matches']}, Columns: {info['total_columns']}")
                am = info["available_metrics"]
                print(f"  Player-level stats: {am['has_player_identities']}")
                print(f"  Event timestamps:   {am['has_event_timestamps']}")
                print(f"  Spatial coordinates:{am['has_spatial_coordinates']}")
                print(f"  Expected Goals(xG): {am['has_xg']}")
                print(f"  Betting Odds:       {am['has_betting_odds']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
