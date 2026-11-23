"""
TouchlineLM - Raw Event and Match Extractor
Loads raw JSON files from StatsBomb Open Data and structures them into
typed, normalized entities for downstream deterministic metric extraction.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PlayerLineupInfo:
    player_id: int
    player_name: str
    jersey_number: int
    country_name: str
    primary_position: Optional[str] = None
    starting_xi: bool = False


@dataclass
class TeamInfo:
    team_id: int
    team_name: str
    lineup: Dict[int, PlayerLineupInfo] = field(default_factory=dict)


@dataclass
class MatchMetadata:
    match_id: int
    competition_id: int
    season_id: int
    match_date: str
    competition_stage: str
    home_team: TeamInfo
    away_team: TeamInfo
    home_score: int
    away_score: int
    duration_periods: int = 2


class StatsBombExtractor:
    """Extracts and normalizes raw StatsBomb data files."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.sb_dir = raw_dir / "statsbomb"

    def load_match_metadata(self, comp_id: int, season_id: int, match_id: int) -> MatchMetadata:
        matches_file = self.sb_dir / "matches" / f"{comp_id}_{season_id}.json"
        if not matches_file.exists():
            raise FileNotFoundError(f"Matches file not found: {matches_file}")

        with open(matches_file, "r", encoding="utf-8") as f:
            matches_data = json.load(f)

        target_match = next((m for m in matches_data if m.get("match_id") == match_id), None)
        if not target_match:
            raise ValueError(f"Match ID {match_id} not found in {matches_file.name}")

        home_t = TeamInfo(
            team_id=target_match["home_team"]["home_team_id"],
            team_name=target_match["home_team"]["home_team_name"]
        )
        away_t = TeamInfo(
            team_id=target_match["away_team"]["away_team_id"],
            team_name=target_match["away_team"]["away_team_name"]
        )

        # Load lineups
        lineups_file = self.sb_dir / "lineups" / f"{match_id}.json"
        if lineups_file.exists():
            with open(lineups_file, "r", encoding="utf-8") as lf:
                lineups_data = json.load(lf)
            for t_entry in lineups_data:
                team_obj = home_t if t_entry.get("team_id") == home_t.team_id else away_t
                for p in t_entry.get("lineup", []):
                    positions = p.get("positions", [])
                    first_pos = positions[0].get("position") if positions else None
                    starting = any(pos.get("start_reason") == "Starting XI" for pos in positions)
                    p_info = PlayerLineupInfo(
                        player_id=p["player_id"],
                        player_name=p["player_name"],
                        jersey_number=p.get("jersey_number", 0),
                        country_name=p.get("country", {}).get("name", "Unknown"),
                        primary_position=first_pos,
                        starting_xi=starting
                    )
                    team_obj.lineup[p["player_id"]] = p_info

        return MatchMetadata(
            match_id=match_id,
            competition_id=comp_id,
            season_id=season_id,
            match_date=target_match.get("match_date", "Unknown"),
            competition_stage=target_match.get("competition_stage", {}).get("name", "Match"),
            home_team=home_t,
            away_team=away_t,
            home_score=target_match.get("home_score", 0),
            away_score=target_match.get("away_score", 0)
        )

    def load_events(self, match_id: int) -> List[Dict[str, Any]]:
        events_file = self.sb_dir / "events" / f"{match_id}.json"
        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")

        with open(events_file, "r", encoding="utf-8") as f:
            events = json.load(f)
        return events
