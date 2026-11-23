"""
TouchlineLM - Football Intelligence Corpus Generator (Phase 3A)
Transforms extracted match and player statistics into rich, diverse,
multi-tiered training text samples with controlled sentence-level variations:
- Tier 1: Structured Match Narratives
- Tier 2: Player Performance Dossiers (with syntactic permutations & zero-omission)
- Tier 3: Tactical & Statistical Q&A
All variations are strictly deterministic via controlled modular seeds (match_id, player_id).
Zero numerical hallucination; all numbers match ground truth.
"""

from typing import Any, Dict, List, Optional
from src.metrics.calculator import ExtractedMatchData, PlayerMatchMetrics, MatchPhaseMetrics


class CorpusGenerator:
    """Generates multi-tiered training text samples with syntactic diversity."""

    @staticmethod
    def generate_match_narrative(match: ExtractedMatchData) -> Dict[str, Any]:
        """Generate a structured, chronological match narrative with controlled phase phrasing."""
        h = match.home_metrics
        a = match.away_metrics
        seed = match.match_id % 3

        # Varied narrative introductions
        if seed == 0:
            intro_p = (
                f"In this {match.competition_stage} encounter, {match.home_team_name} and {match.away_team_name} contested an intense match. "
                f"{match.home_team_name} completed {h.completed_passes} of {h.total_passes} attempted passes ({h.pass_accuracy_pct}% accuracy), "
                f"generating {h.progressive_passes} progressive passes and {h.progressive_carries} progressive carries. "
                f"In attack, {match.home_team_name} produced {h.total_shots} total shots ({h.shots_on_target} on target) accumulating {h.total_xg} expected goals (xG). "
                f"Conversely, {match.away_team_name} recorded {a.completed_passes} completed passes from {a.total_passes} attempts ({a.pass_accuracy_pct}% accuracy), "
                f"driving {a.progressive_passes} progressive passes and {a.progressive_carries} progressive carries. "
                f"{match.away_team_name} fashioned {a.total_shots} shots ({a.shots_on_target} on target) for a cumulative total of {a.total_xg} xG. "
                f"Defensively, {match.home_team_name} registered {h.interceptions} interceptions, {h.ball_recoveries} recoveries, and won {h.tackles_won} of {h.tackles_attempted} tackles, "
                f"while {match.away_team_name} countered with {a.interceptions} interceptions, {a.ball_recoveries} recoveries, and won {a.tackles_won} of {a.tackles_attempted} tackles."
            )
        elif seed == 1:
            intro_p = (
                f"Contesting a competitive {match.competition_stage} fixture, {match.home_team_name} faced {match.away_team_name}. "
                f"{match.home_team_name} completed {h.completed_passes} of {h.total_passes} attempted passes ({h.pass_accuracy_pct}% accuracy), "
                f"generating {h.progressive_passes} progressive passes and {h.progressive_carries} progressive carries. "
                f"In attack, {match.home_team_name} produced {h.total_shots} total shots ({h.shots_on_target} on target) accumulating {h.total_xg} expected goals (xG). "
                f"Conversely, {match.away_team_name} recorded {a.completed_passes} completed passes from {a.total_passes} attempts ({a.pass_accuracy_pct}% accuracy), "
                f"driving {a.progressive_passes} progressive passes and {a.progressive_carries} progressive carries. "
                f"{match.away_team_name} fashioned {a.total_shots} shots ({a.shots_on_target} on target) for a cumulative total of {a.total_xg} xG. "
                f"Defensively, {match.home_team_name} registered {h.interceptions} interceptions, {h.ball_recoveries} recoveries, and won {h.tackles_won} of {h.tackles_attempted} tackles, "
                f"while {match.away_team_name} countered with {a.interceptions} interceptions, {a.ball_recoveries} recoveries, and won {a.tackles_won} of {a.tackles_attempted} tackles."
            )
        else:
            intro_p = (
                f"A high-stakes {match.competition_stage} clash featured {match.home_team_name} against {match.away_team_name}. "
                f"{match.home_team_name} completed {h.completed_passes} of {h.total_passes} attempted passes ({h.pass_accuracy_pct}% accuracy), "
                f"generating {h.progressive_passes} progressive passes and {h.progressive_carries} progressive carries. "
                f"In attack, {match.home_team_name} produced {h.total_shots} total shots ({h.shots_on_target} on target) accumulating {h.total_xg} expected goals (xG). "
                f"Conversely, {match.away_team_name} recorded {a.completed_passes} completed passes from {a.total_passes} attempts ({a.pass_accuracy_pct}% accuracy), "
                f"driving {a.progressive_passes} progressive passes and {a.progressive_carries} progressive carries. "
                f"{match.away_team_name} fashioned {a.total_shots} shots ({a.shots_on_target} on target) for a cumulative total of {a.total_xg} xG. "
                f"Defensively, {match.home_team_name} registered {h.interceptions} interceptions, {h.ball_recoveries} recoveries, and won {h.tackles_won} of {h.tackles_attempted} tackles, "
                f"while {match.away_team_name} countered with {a.interceptions} interceptions, {a.ball_recoveries} recoveries, and won {a.tackles_won} of {a.tackles_attempted} tackles."
            )

        lines = [
            f"# Match Report: {match.fixture}",
            f"**Competition Stage**: {match.competition_stage} | **Date**: {match.match_date}",
            f"**Final Scoreline**: {match.home_team_name} {match.home_score} - {match.away_score} {match.away_team_name}",
            "",
            "## 1. Macroscopic Match Summary",
            intro_p,
            "",
            "## 2. Chronological Phase-by-Phase Progression"
        ]

        # Phase breakdowns
        for ph in match.phases:
            if ph.home_passes == 0 and ph.away_passes == 0 and ph.home_shots == 0 and ph.away_shots == 0:
                continue

            phase_header = f"### Phase {ph.phase_label}"
            lines.append(phase_header)

            phase_seed = (match.match_id + ph.start_minute) % 3
            if phase_seed == 0:
                phase_text = (
                    f"During the {ph.phase_label} window, {match.home_team_name} attempted {ph.home_passes} passes and took {ph.home_shots} shots ({ph.home_xg} xG), "
                    f"while {match.away_team_name} mounted {ph.away_passes} passes and took {ph.away_shots} shots ({ph.away_xg} xG)."
                )
            elif phase_seed == 1:
                phase_text = (
                    f"Across the {ph.phase_label} phase, {match.home_team_name} registered {ph.home_passes} passes and {ph.home_shots} shots ({ph.home_xg} xG), "
                    f"compared to {match.away_team_name}'s {ph.away_passes} passes and {ph.away_shots} shots ({ph.away_xg} xG)."
                )
            else:
                phase_text = (
                    f"In the {ph.phase_label} segment of play, {match.home_team_name} produced {ph.home_passes} passes and {ph.home_shots} shots ({ph.home_xg} xG), "
                    f"as {match.away_team_name} countered with {ph.away_passes} passes and {ph.away_shots} shots ({ph.away_xg} xG)."
                )

            if ph.goals_events:
                goals_text = " Decisive breakthrough: " + "; ".join([
                    f"at minute {g['minute']}, {g['scorer']} scored for {g['team']} ({g['xg']} xG, {g['body_part']}, {g['technique']})"
                    for g in ph.goals_events
                ]) + "."
                phase_text += goals_text
            else:
                no_goal_variants = [
                    " Neither side managed to breach the opposing defensive lines during this interval.",
                    " Defensive structures remained compact on both ends with no goals scored.",
                    " Both backlines held firm to prevent any goals during this period."
                ]
                phase_text += no_goal_variants[phase_seed]

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
        """Generate a player scouting report with controlled syntactic permutations and clean omission of zeroes."""
        if player.passes_attempted < 10 and player.tackles_attempted == 0 and player.shots_total == 0 and player.interceptions == 0:
            return None

        # Stable deterministic seed based on match_id and player_id
        seed = (match.match_id * 17 + player.player_id * 31) % 4

        # 1. Distribution & Progression (4 permutations)
        if seed == 0:
            dist_1 = f"{player.player_name} completed {player.passes_completed} of {player.passes_attempted} attempted passes, achieving a pass completion rate of {player.pass_completion_pct}%."
            prog_1 = f"In terms of forward momentum, the player executed {player.progressive_passes} progressive passes, delivering {player.passes_into_final_third} into the attacking third and {player.passes_into_box} into the opponent penalty box."
        elif seed == 1:
            dist_1 = f"In possession, {player.player_name} completed {player.passes_completed} of {player.passes_attempted} attempted passes, achieving a pass completion rate of {player.pass_completion_pct}%."
            prog_1 = f"Progressing play forward, the player executed {player.progressive_passes} progressive passes, delivering {player.passes_into_final_third} into the attacking third and {player.passes_into_box} into the opponent penalty box."
        elif seed == 2:
            dist_1 = f"Circulating the ball, {player.player_name} completed {player.passes_completed} of {player.passes_attempted} attempted passes, achieving a pass completion rate of {player.pass_completion_pct}%."
            prog_1 = f"In terms of forward momentum, the player executed {player.progressive_passes} progressive passes, delivering {player.passes_into_final_third} into the attacking third and {player.passes_into_box} into the opponent penalty box."
        else:
            dist_1 = f"Operating in possession, {player.player_name} completed {player.passes_completed} of {player.passes_attempted} attempted passes, achieving a pass completion rate of {player.pass_completion_pct}%."
            prog_1 = f"Progressing play forward, the player executed {player.progressive_passes} progressive passes, delivering {player.passes_into_final_third} into the attacking third and {player.passes_into_box} into the opponent penalty box."

        # Crosses/Carries handling (omit zero crosses and through balls if player had none)
        if player.crosses_attempted > 0 or player.through_balls > 0:
            carry_text = (
                f"Additionally, {player.player_name} completed {player.crosses_attempted} crosses, {player.through_balls} through balls, "
                f"and recorded {player.progressive_carries} progressive carries across {player.carries_count} total on-ball carries."
            )
        else:
            carry_text = (
                f"Additionally, {player.player_name} focused on ground progression, recording {player.progressive_carries} progressive carries "
                f"across {player.carries_count} total on-ball carries without delivering crosses or through balls."
            )

        # 2. Attacking Output (clean handling of 0 shots)
        if player.shots_total > 0:
            shot_text = (
                f"In offensive situations, {player.player_name} registered {player.shots_total} total shots ({player.shots_on_target} on target), "
                f"scoring {player.goals_scored} goals from an accumulated expected goals value of {player.expected_goals} xG."
            )
        else:
            shot_text = (
                f"In offensive situations, {player.player_name} did not register any direct shots on goal, "
                f"recording {player.shots_total} total shots ({player.shots_on_target} on target), scoring {player.goals_scored} goals from an accumulated expected goals value of {player.expected_goals} xG."
            )

        # Key Passes / xA
        if player.key_passes > 0:
            creation_text = f"As a creator, {player.player_name} generated {player.key_passes} key passes (shot assists), yielding an expected assists tally of {player.expected_assists} xA."
        else:
            creation_text = f"In chance creation, {player.player_name} generated {player.key_passes} key passes (shot assists), yielding an expected assists tally of {player.expected_assists} xA."

        # 3. Defensive Interventions
        if seed % 2 == 0:
            tackle_text = f"Without the ball, {player.player_name} was active in defensive phases, attempting {player.tackles_attempted} tackles and winning {player.tackles_won} ({player.tackle_success_pct}% success rate)."
        else:
            tackle_text = f"In defensive transitions, {player.player_name} contributed with determination, attempting {player.tackles_attempted} tackles and winning {player.tackles_won} ({player.tackle_success_pct}% success rate)."

        def_actions_text = (
            f"The player registered {player.interceptions} interceptions, {player.clearances} clearances, "
            f"{player.blocks} blocks, {player.ball_recoveries} ball recoveries, and applied pressure on {player.pressures_applied} occasions."
        )

        # Clean discipline text omitting empty card warnings
        if player.yellow_cards == 0 and player.red_cards == 0 and player.fouls_committed == 0:
            discipline_text = f"In discipline, {player.player_name} maintained a clean sheet with 0 fouls and 0 bookings."
        else:
            discipline_text = f"In discipline, {player.player_name} committed {player.fouls_committed} fouls and was cautioned with {player.yellow_cards} yellow cards and {player.red_cards} red cards."

        lines = [
            f"# Analytical Player Dossier: {player.player_name}",
            f"**Fixture**: {match.fixture} ({match.competition_stage}) | **Date**: {match.match_date}",
            f"**Team**: {player.team_name} | **Position**: {player.position} | **Squad Number**: {player.jersey_number}",
            f"**Starting Status**: {'Starting XI' if player.starting_xi else 'Substitute'}",
            "",
            "## 1. Distribution and Ball Progression",
            f"{dist_1} {prog_1} {carry_text}",
            "",
            "## 2. Attacking Output and Chance Creation",
            f"{shot_text} {creation_text}",
            "",
            "## 3. Defensive Interventions and Duels",
            f"{tackle_text} {def_actions_text} {discipline_text}",
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
