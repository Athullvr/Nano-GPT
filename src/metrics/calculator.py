"""
TouchlineLM - Deterministic Metrics Engine
Calculates match-level, team-level, player-level, and phase-level metrics
directly from raw StatsBomb event coordinates and action dictionaries.
Every calculation is 100% deterministic and mathematically verifiable.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


def distance_to_goal(x: float, y: float, goal_x: float = 120.0, goal_y: float = 40.0) -> float:
    """Calculate Euclidean distance to the center of the opponent's goal."""
    return math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)


def is_inside_box(x: float, y: float) -> bool:
    """Check if coordinate is inside the opponent's 18-yard penalty box."""
    return x >= 102.0 and 18.0 <= y <= 62.0


def is_progressive_pass(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    pass_type: Optional[str] = None
) -> bool:
    """
    Standard Football Analytics definition of a progressive pass:
    A completed forward pass that moves the ball at least 25% closer to the
    opponent's goal center, OR delivers the ball into the opponent's penalty box.
    
    Refinements & Exclusions:
    - Excludes defensive clearance events or passes marked as clearances.
    - Excludes passes originating in own defensive third (start_x < 40.0)
      to prevent deep defensive clearances from inflating progression metrics.
    """
    # Explicitly exclude clearance pass types
    if pass_type and "clearance" in pass_type.lower():
        return False

    # Exclude passes originating in own defensive third (x < 40.0)
    if start_x < 40.0:
        return False

    d_start = distance_to_goal(start_x, start_y)
    d_end = distance_to_goal(end_x, end_y)

    # Deliveries into the box from outside
    if not is_inside_box(start_x, start_y) and is_inside_box(end_x, end_y):
        return True

    # 25% reduction in goal distance
    if d_start > 0 and (d_start - d_end) / d_start >= 0.25:
        return True

    return False


def is_progressive_carry(start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
    """
    Standard definition of a progressive carry:
    A continuous ball carry advancing forward towards the opponent goal line
    by at least 10 yards, or entering the penalty box.
    
    Excludes carries confined entirely to the defensive third (end_x <= 40.0).
    """
    if not is_inside_box(start_x, start_y) and is_inside_box(end_x, end_y):
        return True

    dx = end_x - start_x
    if dx >= 10.0 and end_x > 40.0 and start_x >= 35.0:
        return True

    return False


@dataclass
class PlayerMatchMetrics:
    player_id: int
    player_name: str
    team_name: str
    position: str
    jersey_number: int
    minutes_played: int = 90
    starting_xi: bool = False

    # Passing
    passes_attempted: int = 0
    passes_completed: int = 0
    progressive_passes: int = 0
    passes_into_final_third: int = 0
    passes_into_box: int = 0
    crosses_attempted: int = 0
    through_balls: int = 0
    key_passes: int = 0  # Shot assists
    expected_assists: float = 0.0

    # Progression / Carries
    carries_count: int = 0
    progressive_carries: int = 0

    # Shooting / Threat
    shots_total: int = 0
    shots_on_target: int = 0
    goals_scored: int = 0
    expected_goals: float = 0.0

    # Defensive
    tackles_attempted: int = 0
    tackles_won: int = 0
    interceptions: int = 0
    clearances: int = 0
    blocks: int = 0
    ball_recoveries: int = 0
    pressures_applied: int = 0

    # Discipline
    fouls_committed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0

    @property
    def pass_completion_pct(self) -> float:
        if self.passes_attempted == 0:
            return 0.0
        return round((self.passes_completed / self.passes_attempted) * 100, 1)

    @property
    def tackle_success_pct(self) -> float:
        if self.tackles_attempted == 0:
            return 0.0
        return round((self.tackles_won / self.tackles_attempted) * 100, 1)


@dataclass
class MatchPhaseMetrics:
    phase_label: str  # e.g., "0-15'", "15-30'", "30-45'", "45-60'", "60-75'", "75-90'", "Extra Time"
    start_minute: int
    end_minute: int
    home_passes: int = 0
    away_passes: int = 0
    home_shots: int = 0
    away_shots: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
    goals_events: List[Dict[str, Any]] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)


@dataclass
class TeamMatchMetrics:
    team_name: str
    score: int = 0
    total_passes: int = 0
    completed_passes: int = 0
    progressive_passes: int = 0
    progressive_carries: int = 0
    total_shots: int = 0
    shots_on_target: int = 0
    total_xg: float = 0.0
    interceptions: int = 0
    ball_recoveries: int = 0
    clearances: int = 0
    tackles_won: int = 0
    tackles_attempted: int = 0
    pressures: int = 0

    @property
    def pass_accuracy_pct(self) -> float:
        if self.total_passes == 0:
            return 0.0
        return round((self.completed_passes / self.total_passes) * 100, 1)


@dataclass
class ExtractedMatchData:
    match_id: int
    fixture: str
    competition_stage: str
    match_date: str
    home_team_name: str
    away_team_name: str
    home_score: int
    away_score: int
    home_metrics: TeamMatchMetrics
    away_metrics: TeamMatchMetrics
    player_metrics: Dict[int, PlayerMatchMetrics]
    phases: List[MatchPhaseMetrics]
    goals_summary: List[Dict[str, Any]]


class MetricsCalculator:
    """Performs deterministic event processing across a full match."""

    @staticmethod
    def process_match_events(
        match_id: int,
        meta_info: Dict[str, Any],
        events: List[Dict[str, Any]],
        lineups_info: Dict[str, Any]
    ) -> ExtractedMatchData:
        home_name = meta_info["home_team"]["home_team_name"]
        away_name = meta_info["away_team"]["away_team_name"]
        home_score = meta_info.get("home_score", 0)
        away_score = meta_info.get("away_score", 0)

        home_team = TeamMatchMetrics(team_name=home_name, score=home_score)
        away_team = TeamMatchMetrics(team_name=away_name, score=away_score)

        # Build player dictionary from lineups
        player_dict: Dict[int, PlayerMatchMetrics] = {}
        for t_name, squad in lineups_info.items():
            for p_id, p_obj in squad.items():
                player_dict[p_id] = PlayerMatchMetrics(
                    player_id=p_id,
                    player_name=p_obj.player_name,
                    team_name=t_name,
                    position=p_obj.primary_position or "Substitute",
                    jersey_number=p_obj.jersey_number,
                    starting_xi=p_obj.starting_xi
                )

        # Build map of shot events for xA calculation
        shot_map: Dict[str, Dict[str, Any]] = {}
        for e in events:
            if e.get("type", {}).get("name") == "Shot":
                shot_map[e.get("id")] = e

        # Standard 15-minute phase bins
        phases = [
            MatchPhaseMetrics(phase_label="0-15'", start_minute=0, end_minute=15),
            MatchPhaseMetrics(phase_label="15-30'", start_minute=15, end_minute=30),
            MatchPhaseMetrics(phase_label="30-45'", start_minute=30, end_minute=45),
            MatchPhaseMetrics(phase_label="45-60'", start_minute=45, end_minute=60),
            MatchPhaseMetrics(phase_label="60-75'", start_minute=60, end_minute=75),
            MatchPhaseMetrics(phase_label="75-90'", start_minute=75, end_minute=90),
            MatchPhaseMetrics(phase_label="Extra Time", start_minute=90, end_minute=130),
        ]

        goals_summary = []

        # Iterate through events chronologically
        for e in events:
            e_type = e.get("type", {}).get("name")
            team_n = e.get("team", {}).get("name")
            t_metrics = home_team if team_n == home_name else away_team
            p_data = e.get("player")
            minute = e.get("minute", 0)
            period = e.get("period", 1)

            # Determine match phase
            if period in [3, 4, 5] or minute >= 90:
                cur_phase = phases[6]
            else:
                idx = min(minute // 15, 5)
                cur_phase = phases[idx]

            # Player object reference
            p_metrics: Optional[PlayerMatchMetrics] = None
            if p_data:
                p_id = p_data.get("id")
                if p_id in player_dict:
                    p_metrics = player_dict[p_id]
                else:
                    # Player not in starting/bench lineup (edge case)
                    p_metrics = PlayerMatchMetrics(
                        player_id=p_id,
                        player_name=p_data.get("name", "Unknown"),
                        team_name=team_n or "Unknown",
                        position="Reserve",
                        jersey_number=0
                    )
                    player_dict[p_id] = p_metrics

            # 1. PASSES
            if e_type == "Pass":
                t_metrics.total_passes += 1
                if team_n == home_name:
                    cur_phase.home_passes += 1
                else:
                    cur_phase.away_passes += 1

                pass_sub = e.get("pass", {})
                outcome = pass_sub.get("outcome", {}).get("name")
                loc = e.get("location", [0, 0])
                end_loc = pass_sub.get("end_location", [0, 0])
                is_completed = (outcome is None)

                if p_metrics:
                    p_metrics.passes_attempted += 1

                if is_completed:
                    t_metrics.completed_passes += 1
                    if p_metrics:
                        p_metrics.passes_completed += 1

                    # Progressive pass calculation
                    if len(loc) >= 2 and len(end_loc) >= 2:
                        p_type = pass_sub.get("type", {}).get("name")
                        if is_progressive_pass(loc[0], loc[1], end_loc[0], end_loc[1], pass_type=p_type):
                            t_metrics.progressive_passes += 1
                            if p_metrics:
                                p_metrics.progressive_passes += 1

                        # Passes into final third & box
                        if loc[0] < 80.0 and end_loc[0] >= 80.0 and p_metrics:
                            p_metrics.passes_into_final_third += 1
                        if not is_inside_box(loc[0], loc[1]) and is_inside_box(end_loc[0], end_loc[1]) and p_metrics:
                            p_metrics.passes_into_box += 1

                # Key passes / Shot assists (xA)
                assisted_shot_id = pass_sub.get("assisted_shot_id")
                if assisted_shot_id and assisted_shot_id in shot_map:
                    if p_metrics:
                        p_metrics.key_passes += 1
                        shot_xg = shot_map[assisted_shot_id].get("shot", {}).get("statsbomb_xg", 0.0)
                        p_metrics.expected_assists += shot_xg

                if pass_sub.get("cross") and p_metrics:
                    p_metrics.crosses_attempted += 1
                if pass_sub.get("through_ball") and p_metrics:
                    p_metrics.through_balls += 1

            # 2. CARRIES
            elif e_type == "Carry":
                carry_sub = e.get("carry", {})
                loc = e.get("location", [0, 0])
                end_loc = carry_sub.get("end_location", [0, 0])
                if len(loc) >= 2 and len(end_loc) >= 2:
                    is_prog = is_progressive_carry(loc[0], loc[1], end_loc[0], end_loc[1])
                    if is_prog:
                        t_metrics.progressive_carries += 1
                        if p_metrics:
                            p_metrics.progressive_carries += 1
                if p_metrics:
                    p_metrics.carries_count += 1

            # 3. SHOTS
            elif e_type == "Shot":
                shot_sub = e.get("shot", {})
                xg = float(shot_sub.get("statsbomb_xg", 0.0))
                outcome = shot_sub.get("outcome", {}).get("name", "Unknown")

                t_metrics.total_shots += 1
                t_metrics.total_xg += xg

                if team_n == home_name:
                    cur_phase.home_shots += 1
                    cur_phase.home_xg += xg
                else:
                    cur_phase.away_shots += 1
                    cur_phase.away_xg += xg

                if outcome in ["Goal", "Saved", "Saved Off Target"]:
                    t_metrics.shots_on_target += 1
                    if p_metrics:
                        p_metrics.shots_on_target += 1

                if outcome == "Goal":
                    goal_record = {
                        "minute": minute,
                        "second": e.get("second", 0),
                        "period": period,
                        "team": team_n,
                        "scorer": p_metrics.player_name if p_metrics else "Unknown",
                        "xg": round(xg, 3),
                        "body_part": shot_sub.get("body_part", {}).get("name", "Open Play"),
                        "technique": shot_sub.get("technique", {}).get("name", "Normal")
                    }
                    goals_summary.append(goal_record)
                    cur_phase.goals_events.append(goal_record)
                    if p_metrics:
                        p_metrics.goals_scored += 1

                if p_metrics:
                    p_metrics.shots_total += 1
                    p_metrics.expected_goals += xg

            # 4. DEFENSIVE ACTIONS
            elif e_type == "Duel":
                duel_sub = e.get("duel", {})
                d_type = duel_sub.get("type", {}).get("name", "")
                outcome = duel_sub.get("outcome", {}).get("name", "")
                if "Tackle" in d_type or d_type == "Tackle":
                    t_metrics.tackles_attempted += 1
                    if p_metrics:
                        p_metrics.tackles_attempted += 1
                    if outcome in ["Won", "Success"]:
                        t_metrics.tackles_won += 1
                        if p_metrics:
                            p_metrics.tackles_won += 1

            elif e_type == "Interception":
                t_metrics.interceptions += 1
                if p_metrics:
                    p_metrics.interceptions += 1

            elif e_type == "Clearance":
                t_metrics.clearances += 1
                if p_metrics:
                    p_metrics.clearances += 1

            elif e_type == "Block":
                if p_metrics:
                    p_metrics.blocks += 1

            elif e_type == "Ball Recovery":
                t_metrics.ball_recoveries += 1
                if p_metrics:
                    p_metrics.ball_recoveries += 1

            elif e_type == "Pressure":
                t_metrics.pressures += 1
                if p_metrics:
                    p_metrics.pressures_applied += 1

            # 5. DISCIPLINE
            elif e_type == "Foul Committed":
                if p_metrics:
                    p_metrics.fouls_committed += 1

            elif e_type == "Bad Behaviour":
                card = e.get("bad_behaviour", {}).get("card", {}).get("name")
                if p_metrics and card:
                    if "Yellow" in card:
                        p_metrics.yellow_cards += 1
                    elif "Red" in card:
                        p_metrics.red_cards += 1

        # Round all floating point totals
        home_team.total_xg = round(home_team.total_xg, 2)
        away_team.total_xg = round(away_team.total_xg, 2)
        for ph in phases:
            ph.home_xg = round(ph.home_xg, 2)
            ph.away_xg = round(ph.away_xg, 2)
        for pm in player_dict.values():
            pm.expected_goals = round(pm.expected_goals, 2)
            pm.expected_assists = round(pm.expected_assists, 2)

        stage_val = meta_info.get("competition_stage", "Match")
        stage_name = stage_val.get("name", "Match") if isinstance(stage_val, dict) else str(stage_val)

        fixture_str = f"{home_name} vs {away_name}"
        return ExtractedMatchData(
            match_id=match_id,
            fixture=fixture_str,
            competition_stage=stage_name,
            match_date=meta_info.get("match_date", "Unknown"),
            home_team_name=home_name,
            away_team_name=away_name,
            home_score=home_score,
            away_score=away_score,
            home_metrics=home_team,
            away_metrics=away_team,
            player_metrics=player_dict,
            phases=phases,
            goals_summary=goals_summary
        )
