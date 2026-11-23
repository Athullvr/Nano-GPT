"""
TouchlineLM - Football Intelligence Corpus Generator
Transforms extracted match and player statistics into rich, diverse,
multi-tiered training text samples:
- Tier 1: Structured Match Narratives (chronological phase-by-phase flow, xG shifts, goals)
- Tier 2: Player Performance Dossiers (scouting reports, progressive metrics, defensive duels)
- Tier 3: Tactical & Statistical Q&A (domain reasoning, metric explanations, game state analysis)
Every single numerical statistic is strictly derived from ground-truth extractions.
"""

from typing import Any, Dict, List, Optional
from src.metrics.calculator import ExtractedMatchData, PlayerMatchMetrics, MatchPhaseMetrics


class CorpusGenerator:
    """Generates multi-tiered training text samples from extracted match metrics."""

    @staticmethod
    def generate_match_narrative(match: ExtractedMatchData) -> Dict[str, Any]:
        """Generate a structured, chronological match narrative."""
        h = match.home_metrics
        a = match.away_metrics

        lines = [
            f"# Match Report: {match.fixture}",
            f"**Competition Stage**: {match.competition_stage} | **Date**: {match.match_date}",
            f"**Final Scoreline**: {match.home_team_name} {match.home_score} - {match.away_score} {match.away_team_name}",
            "",
            "## 1. Macroscopic Match Summary",
            f"In this {match.competition_stage} encounter, {match.home_team_name} and {match.away_team_name} contested an intense match. "
            f"{match.home_team_name} completed {h.completed_passes} of {h.total_passes} attempted passes ({h.pass_accuracy_pct}% accuracy), "
            f"generating {h.progressive_passes} progressive passes and {h.progressive_carries} progressive carries. "
            f"In attack, {match.home_team_name} produced {h.total_shots} total shots ({h.shots_on_target} on target) accumulating {h.total_xg} expected goals (xG). "
            f"Conversely, {match.away_team_name} recorded {a.completed_passes} completed passes from {a.total_passes} attempts ({a.pass_accuracy_pct}% accuracy), "
            f"driving {a.progressive_passes} progressive passes and {a.progressive_carries} progressive carries. "
            f"{match.away_team_name} fashioned {a.total_shots} shots ({a.shots_on_target} on target) for a cumulative total of {a.total_xg} xG. "
            f"Defensively, {match.home_team_name} registered {h.interceptions} interceptions, {h.ball_recoveries} recoveries, and won {h.tackles_won} of {h.tackles_attempted} tackles, "
            f"while {match.away_team_name} countered with {a.interceptions} interceptions, {a.ball_recoveries} recoveries, and won {a.tackles_won} of {a.tackles_attempted} tackles.",
            "",
            "## 2. Chronological Phase-by-Phase Progression"
        ]

        # Phase breakdowns
        for ph in match.phases:
            # Only include phases that had events or shots
            if ph.home_passes == 0 and ph.away_passes == 0 and ph.home_shots == 0 and ph.away_shots == 0:
                continue

            phase_header = f"### Phase {ph.phase_label}"
            lines.append(phase_header)

            phase_text = (
                f"During the {ph.phase_label} window, {match.home_team_name} attempted {ph.home_passes} passes and took {ph.home_shots} shots ({ph.home_xg} xG), "
                f"while {match.away_team_name} mounted {ph.away_passes} passes and took {ph.away_shots} shots ({ph.away_xg} xG)."
            )

            if ph.goals_events:
                goals_text = " Decisive breakthrough: " + "; ".join([
                    f"at minute {g['minute']}, {g['scorer']} scored for {g['team']} ({g['xg']} xG, {g['body_part']}, {g['technique']})"
                    for g in ph.goals_events
                ]) + "."
                phase_text += goals_text
            else:
                phase_text += " Neither side managed to breach the opposing defensive lines during this interval."

            lines.append(phase_text)
            lines.append("")

        # Conclusion
        if match.home_score > match.away_score:
            verdict = f"{match.home_team_name} secured victory with a {match.home_score}-{match.away_score} scoreline."
        elif match.away_score > match.home_score:
            verdict = f"{match.away_team_name} claimed victory with a {match.away_score}-{match.home_score} scoreline."
        else:
            verdict = f"The contest concluded as a {match.home_score}-{match.away_score} draw after regular duration."

        lines.append("## 3. Analytical Conclusion")
        lines.append(verdict)

        full_text = "\n".join(lines)
        return {
            "sample_id": f"narrative_{match.match_id}",
            "sample_type": "match_narrative",
            "match_id": match.match_id,
            "fixture": match.fixture,
            "content": full_text
        }

    @staticmethod
    def generate_player_dossier(player: PlayerMatchMetrics, match: ExtractedMatchData) -> Optional[Dict[str, Any]]:
        """Generate a player scouting report / analytical performance dossier."""
        # Focus on players with meaningful on-ball contributions
        if player.passes_attempted < 10 and player.tackles_attempted == 0 and player.shots_total == 0 and player.interceptions == 0:
            return None

        lines = [
            f"# Analytical Player Dossier: {player.player_name}",
            f"**Fixture**: {match.fixture} ({match.competition_stage}) | **Date**: {match.match_date}",
            f"**Team**: {player.team_name} | **Position**: {player.position} | **Squad Number**: {player.jersey_number}",
            f"**Starting Status**: {'Starting XI' if player.starting_xi else 'Substitute'}",
            "",
            "## 1. Distribution and Ball Progression",
            f"{player.player_name} completed {player.passes_completed} of {player.passes_attempted} attempted passes, achieving a pass completion rate of {player.pass_completion_pct}%. "
            f"In terms of forward momentum, the player executed {player.progressive_passes} progressive passes, delivering {player.passes_into_final_third} into the attacking third "
            f"and {player.passes_into_box} into the opponent penalty box. "
            f"Additionally, {player.player_name} completed {player.crosses_attempted} crosses, {player.through_balls} through balls, and recorded {player.progressive_carries} progressive carries "
            f"across {player.carries_count} total on-ball carries.",
            "",
            "## 2. Attacking Output and Chance Creation",
            f"In offensive situations, {player.player_name} registered {player.shots_total} total shots ({player.shots_on_target} on target), "
            f"scoring {player.goals_scored} goals from an accumulated expected goals value of {player.expected_goals} xG. "
            f"As a creator, {player.player_name} generated {player.key_passes} key passes (shot assists), yielding an expected assists tally of {player.expected_assists} xA.",
            "",
            "## 3. Defensive Interventions and Duels",
            f"Without the ball, {player.player_name} was active in defensive phases, attempting {player.tackles_attempted} tackles and winning {player.tackles_won} ({player.tackle_success_pct}% success rate). "
            f"The player registered {player.interceptions} interceptions, {player.clearances} clearances, {player.blocks} blocks, {player.ball_recoveries} ball recoveries, "
            f"and applied pressure on {player.pressures_applied} occasions. "
            f"In discipline, {player.player_name} committed {player.fouls_committed} fouls and was cautioned with {player.yellow_cards} yellow cards and {player.red_cards} red cards.",
            "",
            "## 4. Tactical Evaluation",
            f"Operating as a {player.position} for {player.team_name}, {player.player_name}'s contributions provided vital balance in possession and transition phases."
        ]

        full_text = "\n".join(lines)
        return {
            "sample_id": f"player_{match.match_id}_{player.player_id}",
            "sample_type": "player_dossier",
            "match_id": match.match_id,
            "player_id": player.player_id,
            "player_name": player.player_name,
            "team_name": player.team_name,
            "content": full_text
        }

    @staticmethod
    def generate_tactical_qa(match: ExtractedMatchData) -> Dict[str, Any]:
        """Generate tactical and analytical reasoning Q&A samples."""
        h = match.home_metrics
        a = match.away_metrics

        lines = [
            f"# Tactical and Statistical Analysis: {match.fixture}",
            f"**Fixture Context**: {match.competition_stage} match ending {match.home_score}-{match.away_score}.",
            "",
            "### Question 1: How did the expected goals (xG) metrics compare to the actual match outcome?",
            f"Answer: {match.home_team_name} recorded {h.total_xg} xG from {h.total_shots} shots, converting {match.home_score} goals. "
            f"Conversely, {match.away_team_name} generated {a.total_xg} xG from {a.total_shots} shots, resulting in {match.away_score} goals. "
            f"The cumulative xG battle finished {h.total_xg} to {a.total_xg}.",
            "",
            "### Question 2: Which team controlled the tempo and spatial progression?",
            f"Answer: {match.home_team_name} attempted {h.total_passes} passes ({h.completed_passes} completed, {h.pass_accuracy_pct}%), "
            f"driving {h.progressive_passes} progressive passes and {h.progressive_carries} progressive carries. "
            f"In contrast, {match.away_team_name} registered {a.total_passes} passes ({a.completed_passes} completed, {a.pass_accuracy_pct}%), "
            f"producing {a.progressive_passes} progressive passes and {a.progressive_carries} progressive carries.",
            "",
            "### Question 3: How effective was the defensive disruption and pressing from both sides?",
            f"Answer: Defensively, {match.home_team_name} tallied {h.interceptions} interceptions, {h.clearances} clearances, and {h.ball_recoveries} recoveries, "
            f"winning {h.tackles_won} of {h.tackles_attempted} tackles. "
            f"{match.away_team_name} recorded {a.interceptions} interceptions, {a.clearances} clearances, and {a.ball_recoveries} recoveries, "
            f"winning {a.tackles_won} of {a.tackles_attempted} tackles."
        ]

        full_text = "\n".join(lines)
        return {
            "sample_id": f"qa_{match.match_id}",
            "sample_type": "tactical_qa",
            "match_id": match.match_id,
            "fixture": match.fixture,
            "content": full_text
        }
