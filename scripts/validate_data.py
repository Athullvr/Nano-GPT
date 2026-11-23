"""
TouchlineLM - Data Validation Script (Phase 1)
Validates schema consistency, coordinate bounds, timestamp monotonicity,
and referential integrity across match metadata, events, and lineups.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("validate_data")


class DataValidator:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.sb_dir = raw_dir / "statsbomb"
        self.fd_dir = raw_dir / "football_data"
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_statsbomb_competitions(self) -> int:
        comp_file = self.sb_dir / "competitions.json"
        if not comp_file.exists():
            self.errors.append("Missing competitions.json")
            return 0

        with open(comp_file, "r", encoding="utf-8") as f:
            comps = json.load(f)

        if not isinstance(comps, list) or len(comps) == 0:
            self.errors.append("competitions.json is empty or not a list")
            return 0

        required_keys = {"competition_id", "season_id", "country_name", "competition_name", "season_name"}
        for i, c in enumerate(comps):
            missing = required_keys - set(c.keys())
            if missing:
                self.errors.append(f"Competition entry #{i} missing keys: {missing}")

        logger.info(f"Validated {len(comps)} competition records in competitions.json")
        return len(comps)

    def validate_statsbomb_matches(self) -> Dict[int, Dict[str, Any]]:
        matches_dir = self.sb_dir / "matches"
        if not matches_dir.exists():
            self.errors.append("Missing matches directory")
            return {}

        match_map = {}
        required_keys = {"match_id", "match_date", "home_team", "away_team", "home_score", "away_score"}

        for mf in matches_dir.glob("*.json"):
            with open(mf, "r", encoding="utf-8") as f:
                matches_data = json.load(f)

            for m in matches_data:
                m_id = m.get("match_id")
                if not m_id:
                    self.errors.append(f"Match record in {mf.name} missing match_id")
                    continue
                missing = required_keys - set(m.keys())
                if missing:
                    self.errors.append(f"Match {m_id} missing keys: {missing}")

                home_name = m.get("home_team", {}).get("home_team_name")
                away_name = m.get("away_team", {}).get("away_team_name")
                if not home_name or not away_name:
                    self.errors.append(f"Match {m_id} missing home or away team name")

                match_map[m_id] = m

        logger.info(f"Validated {len(match_map)} match records across {len(list(matches_dir.glob('*.json')))} file(s)")
        return match_map

    def validate_statsbomb_events_and_lineups(self, match_map: Dict[int, Dict[str, Any]]) -> Tuple[int, int]:
        events_dir = self.sb_dir / "events"
        lineups_dir = self.sb_dir / "lineups"

        total_events_checked = 0
        total_matches_checked = 0

        for ef in sorted(events_dir.glob("*.json")):
            m_id = int(ef.stem)
            total_matches_checked += 1

            with open(ef, "r", encoding="utf-8") as f:
                events = json.load(f)

            if not isinstance(events, list):
                self.errors.append(f"Event file {ef.name} is not a valid list")
                continue

            # Load lineups for referential integrity
            lineup_file = lineups_dir / f"{m_id}.json"
            lineup_players = set()
            lineup_teams = set()
            if lineup_file.exists():
                with open(lineup_file, "r", encoding="utf-8") as lf:
                    lineups_data = json.load(lf)
                for team_entry in lineups_data:
                    lineup_teams.add(team_entry.get("team_name"))
                    for p in team_entry.get("lineup", []):
                        lineup_players.add(p.get("player_id"))
            else:
                self.warnings.append(f"No lineup file found for match {m_id}")

            # Match meta teams
            match_meta = match_map.get(m_id, {})
            expected_teams = {
                match_meta.get("home_team", {}).get("home_team_name"),
                match_meta.get("away_team", {}).get("away_team_name")
            } - {None}

            prev_index = 0
            for e_idx, event in enumerate(events):
                total_events_checked += 1
                curr_index = event.get("index", 0)
                if curr_index <= prev_index and e_idx > 0:
                    self.warnings.append(f"Match {m_id}: non-monotonic event index at event {e_idx} (prev {prev_index}, curr {curr_index})")
                prev_index = curr_index

                # Period validation (1: 1H, 2: 2H, 3: ET1, 4: ET2, 5: Penalties)
                period = event.get("period")
                if period not in [1, 2, 3, 4, 5]:
                    self.errors.append(f"Match {m_id}, Event {curr_index}: invalid period {period}")

                # Minute/Second validation
                minute = event.get("minute", -1)
                second = event.get("second", -1)
                if minute < 0 or second < 0 or second > 59:
                    self.errors.append(f"Match {m_id}, Event {curr_index}: invalid time {minute}:{second}")

                # Spatial coordinates validation: 0 <= x <= 120, 0 <= y <= 80
                location = event.get("location")
                if location:
                    if len(location) >= 2:
                        x, y = location[0], location[1]
                        if not (0.0 <= x <= 120.0 and 0.0 <= y <= 80.0):
                            self.warnings.append(f"Match {m_id}, Event {curr_index}: coordinate out of standard bounds: ({x}, {y})")
                    else:
                        self.errors.append(f"Match {m_id}, Event {curr_index}: malformed location array: {location}")

                # Expected Goals range validation: 0.0 <= xG <= 1.0
                shot_data = event.get("shot")
                if shot_data and "statsbomb_xg" in shot_data:
                    xg = shot_data["statsbomb_xg"]
                    if not (0.0 <= xg <= 1.0):
                        self.errors.append(f"Match {m_id}, Event {curr_index}: invalid xG value {xg}")

                # Team referential integrity
                event_team = event.get("team", {}).get("name")
                if expected_teams and event_team and event_team not in expected_teams:
                    self.errors.append(f"Match {m_id}, Event {curr_index}: event team '{event_team}' not in match teams {expected_teams}")

                # Player referential integrity
                player_id = event.get("player", {}).get("id")
                if player_id and lineup_players and player_id not in lineup_players:
                    self.warnings.append(f"Match {m_id}, Event {curr_index}: player ID {player_id} ({event.get('player', {}).get('name')}) not found in starting/bench lineup")

        logger.info(f"Validated {total_events_checked} events across {total_matches_checked} match(es)")
        return total_matches_checked, total_events_checked

    def run_all_validations(self) -> bool:
        logger.info("Executing comprehensive dataset validation suite...")
        self.validate_statsbomb_competitions()
        match_map = self.validate_statsbomb_matches()
        self.validate_statsbomb_events_and_lineups(match_map)

        print("\n" + "=" * 60)
        print("DATASET VALIDATION REPORT")
        print("=" * 60)
        print(f"Total Errors:   {len(self.errors)}")
        print(f"Total Warnings: {len(self.warnings)}")

        if self.errors:
            print("\nCritical Errors:")
            for err in self.errors[:10]:
                print(f"  [ERROR] {err}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            print(f"\nWarnings (Sample of {min(5, len(self.warnings))}):")
            for warn in self.warnings[:5]:
                print(f"  [WARN] {warn}")

        print("=" * 60 + "\n")
        return len(self.errors) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate integrity of raw datasets.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    validator = DataValidator(Path(args.raw_dir))
    success = validator.run_all_validations()

    if not success:
        logger.error("Validation failed with errors.")
        sys.exit(1)
    else:
        logger.info("All dataset validations passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
