"""
TouchlineLM - Unit Tests for Phase 2 Corpus Engineering Pipeline
Validates extraction, deterministic metric computation, text synthesis,
and 100% statistical fact-checking accuracy.
"""

from pathlib import Path
import pytest
from src.extraction.match_extractor import StatsBombExtractor
from src.metrics.calculator import (
    MetricsCalculator,
    is_progressive_pass,
    is_progressive_carry,
    distance_to_goal,
    is_inside_box
)
from src.corpus.generator import CorpusGenerator
from src.validation.fact_checker import FactChecker


@pytest.fixture
def raw_data_dir():
    return Path("data/raw")


def test_distance_and_box_geometry():
    # Opponent goal is at (120, 40)
    assert distance_to_goal(120, 40) == 0.0
    assert distance_to_goal(120, 0) == 40.0
    assert distance_to_goal(0, 40) == 120.0

    # Penalty box: x >= 102 and 18 <= y <= 62
    assert is_inside_box(105, 30) is True
    assert is_inside_box(90, 40) is False
    assert is_inside_box(105, 10) is False


def test_progressive_action_logic():
    # Pass into penalty box from outside
    assert is_progressive_pass(start_x=80, start_y=40, end_x=105, end_y=40) is True

    # Pass moving >= 25% closer to goal originating outside defensive third
    # Start at (60, 40) -> dist = 60
    # End at (85, 40) -> dist = 35 -> (60 - 35)/60 = 41.7% reduction >= 25%
    assert is_progressive_pass(start_x=60, start_y=40, end_x=85, end_y=40) is True

    # Backward / lateral pass
    assert is_progressive_pass(start_x=80, start_y=40, end_x=70, end_y=40) is False

    # Clearance / Deep defensive third exclusion (start_x < 40)
    # Even if distance is reduced by >25%, deep defensive clearances must be excluded
    assert is_progressive_pass(start_x=25, start_y=40, end_x=65, end_y=40) is False
    assert is_progressive_pass(start_x=60, start_y=40, end_x=90, end_y=40, pass_type="Clearance") is False

    # Progressive carry advancing >= 10 yards into attacking territory
    assert is_progressive_carry(start_x=50, start_y=30, end_x=65, end_y=30) is True
    assert is_progressive_carry(start_x=50, start_y=30, end_x=52, end_y=30) is False

    # Carry confined to defensive third
    assert is_progressive_carry(start_x=20, start_y=30, end_x=32, end_y=30) is False


def test_match_extraction_and_metrics(raw_data_dir):
    extractor = StatsBombExtractor(raw_data_dir)
    # Match 3869685: Argentina vs France
    match_meta = extractor.load_match_metadata(43, 106, 3869685)
    events = extractor.load_events(3869685)

    assert match_meta.home_team.team_name == "Argentina"
    assert match_meta.away_team.team_name == "France"
    assert len(events) > 4000

    lineups_dict = {
        match_meta.home_team.team_name: match_meta.home_team.lineup,
        match_meta.away_team.team_name: match_meta.away_team.lineup
    }

    match_data = MetricsCalculator.process_match_events(
        match_id=3869685,
        meta_info={
            "home_team": {"home_team_name": match_meta.home_team.team_name},
            "away_team": {"away_team_name": match_meta.away_team.team_name},
            "home_score": match_meta.home_score,
            "away_score": match_meta.away_score,
            "competition_stage": match_meta.competition_stage,
            "match_date": match_meta.match_date
        },
        events=events,
        lineups_info=lineups_dict
    )

    assert match_data.home_metrics.total_passes > 500
    assert match_data.home_metrics.total_xg > 3.0
    assert match_data.away_metrics.total_shots > 10
    assert len(match_data.goals_summary) > 0


def test_corpus_generation_and_validation(raw_data_dir):
    extractor = StatsBombExtractor(raw_data_dir)
    match_meta = extractor.load_match_metadata(43, 106, 3857256)  # Serbia vs Switzerland
    events = extractor.load_events(3857256)
    lineups_dict = {
        match_meta.home_team.team_name: match_meta.home_team.lineup,
        match_meta.away_team.team_name: match_meta.away_team.lineup
    }

    match_data = MetricsCalculator.process_match_events(
        match_id=3857256,
        meta_info={
            "home_team": {"home_team_name": match_meta.home_team.team_name},
            "away_team": {"away_team_name": match_meta.away_team.team_name},
            "home_score": match_meta.home_score,
            "away_score": match_meta.away_score,
            "competition_stage": match_meta.competition_stage,
            "match_date": match_meta.match_date
        },
        events=events,
        lineups_info=lineups_dict
    )

    # 1. Narratives
    narrative = CorpusGenerator.generate_match_narrative(match_data)
    assert "Serbia" in narrative["content"]
    assert "Switzerland" in narrative["content"]

    # 2. QA
    qa = CorpusGenerator.generate_tactical_qa(match_data)
    assert "Question 1:" in qa["content"]

    # 3. Fact checking
    fact_checker = FactChecker({3857256: match_data})
    claims_narrative, violations = fact_checker.verify_match_narrative(narrative)
    assert claims_narrative > 0
    assert len(violations) == 0

    claims_qa, violations_qa = fact_checker.verify_tactical_qa(qa)
    assert claims_qa > 0
    assert len(violations_qa) == 0
