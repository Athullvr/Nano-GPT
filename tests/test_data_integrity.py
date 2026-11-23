"""
TouchlineLM - Unit & Integration Tests (Phase 1)
Tests data acquisition manifest, structural integrity, and schema compliance.
"""

import json
from pathlib import Path
import pytest
from scripts.validate_data import DataValidator


@pytest.fixture
def raw_data_dir():
    return Path("data/raw")


def test_manifest_existence():
    manifest_path = Path("data/metadata/download_manifest.json")
    assert manifest_path.exists(), "Download manifest file must exist in data/metadata/"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "sources" in manifest
    assert len(manifest["sources"]) > 0
    for src in manifest["sources"]:
        assert src.get("status") == "success"


def test_statsbomb_competitions_exist(raw_data_dir):
    comp_file = raw_data_dir / "statsbomb" / "competitions.json"
    assert comp_file.exists()
    with open(comp_file, "r", encoding="utf-8") as f:
        comps = json.load(f)
    assert isinstance(comps, list)
    assert len(comps) > 0


def test_statsbomb_matches_exist(raw_data_dir):
    matches_dir = raw_data_dir / "statsbomb" / "matches"
    assert matches_dir.exists()
    match_files = list(matches_dir.glob("*.json"))
    assert len(match_files) > 0


def test_statsbomb_events_exist_and_valid(raw_data_dir):
    events_dir = raw_data_dir / "statsbomb" / "events"
    assert events_dir.exists()
    event_files = list(events_dir.glob("*.json"))
    assert len(event_files) >= 2

    # Check first event file
    with open(event_files[0], "r", encoding="utf-8") as f:
        events = json.load(f)
    assert isinstance(events, list)
    assert len(events) > 1000

    # Ensure shot events contain xG
    shot_events = [e for e in events if e.get("type", {}).get("name") == "Shot"]
    assert len(shot_events) > 0
    xg_found = any("statsbomb_xg" in s.get("shot", {}) for s in shot_events)
    assert xg_found, "StatsBomb shot events should contain statsbomb_xg"


def test_validator_passes(raw_data_dir):
    validator = DataValidator(raw_data_dir)
    result = validator.run_all_validations()
    assert result is True, f"Validator failed with errors: {validator.errors}"
