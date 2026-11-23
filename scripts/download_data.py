"""
TouchlineLM - Data Acquisition Script (Phase 1)
Reproducible script for downloading open-source football data samples.

Supported sources:
1. StatsBomb Open Data (GitHub raw CDN):
   - competitions.json
   - matches/<competition_id>/<season_id>.json
   - events/<match_id>.json
   - lineups/<match_id>.json
2. Football-Data.co.uk (CSV sample for macro-level comparison)

Maintains raw data integrity and records a provenance manifest in data/metadata/.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_data")

STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Target sample matches (e.g., FIFA World Cup 2022: competition_id=43, season_id=106)
DEFAULT_STATSBOMB_COMPETITION = 43
DEFAULT_STATSBOMB_SEASON = 106
# Match 3869685: 2022 World Cup Final (Argentina vs France)
# Match 3857256: 2022 World Cup Semi-Final (Argentina vs Croatia)
DEFAULT_SAMPLE_MATCH_IDS = [3869685, 3857256]


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, destination: Path, max_retries: int = 3, delay: float = 1.0) -> Dict[str, Any]:
    """Download a remote file and return provenance metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "TouchlineLM-Research-Agent/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading: {url} -> {destination}")
            with urllib.request.urlopen(req, timeout=30) as response, open(destination, "wb") as out_file:
                content = response.read()
                out_file.write(content)
            
            file_size = destination.stat().st_size
            sha256 = calculate_sha256(destination)
            
            logger.info(f"Successfully downloaded {destination.name} ({file_size} bytes)")
            return {
                "url": url,
                "local_path": str(destination.relative_to(destination.parents[2])),
                "size_bytes": file_size,
                "sha256": sha256,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "success"
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(delay * attempt)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return {
                    "url": url,
                    "local_path": str(destination),
                    "error": str(e),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "failed"
                }


def download_statsbomb_sample(
    output_dir: Path,
    comp_id: int = DEFAULT_STATSBOMB_COMPETITION,
    season_id: int = DEFAULT_STATSBOMB_SEASON,
    match_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Download sample StatsBomb Open Data files."""
    manifest_records = []
    sb_dir = output_dir / "statsbomb"

    # 1. Competitions
    comp_url = f"{STATSBOMB_BASE_URL}/competitions.json"
    comp_dest = sb_dir / "competitions.json"
    manifest_records.append(download_file(comp_url, comp_dest))

    # 2. Matches for specified competition & season
    matches_url = f"{STATSBOMB_BASE_URL}/matches/{comp_id}/{season_id}.json"
    matches_dest = sb_dir / "matches" / f"{comp_id}_{season_id}.json"
    manifest_records.append(download_file(matches_url, matches_dest))

    # 3. Events and lineups for target sample matches
    matches_to_fetch = match_ids or DEFAULT_SAMPLE_MATCH_IDS
    for m_id in matches_to_fetch:
        event_url = f"{STATSBOMB_BASE_URL}/events/{m_id}.json"
        event_dest = sb_dir / "events" / f"{m_id}.json"
        manifest_records.append(download_file(event_url, event_dest))

        lineup_url = f"{STATSBOMB_BASE_URL}/lineups/{m_id}.json"
        lineup_dest = sb_dir / "lineups" / f"{m_id}.json"
        manifest_records.append(download_file(lineup_url, lineup_dest))

    return manifest_records


def download_football_data_sample(output_dir: Path, season_code: str = "2324", league_code: str = "E0") -> List[Dict[str, Any]]:
    """Download sample Football-Data.co.uk CSV."""
    manifest_records = []
    fd_dir = output_dir / "football_data"
    
    # E0 is English Premier League
    csv_url = f"{FOOTBALL_DATA_BASE_URL}/{season_code}/{league_code}.csv"
    csv_dest = fd_dir / f"{league_code}_{season_code}.csv"
    manifest_records.append(download_file(csv_url, csv_dest))
    
    return manifest_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample datasets for TouchlineLM research.")
    parser.add_argument("--source", choices=["statsbomb", "football_data", "all"], default="all",
                        help="Data source to download (default: all)")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory for raw data (default: data/raw)")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata",
                        help="Output directory for provenance metadata (default: data/metadata)")
    parser.add_argument("--comp-id", type=int, default=DEFAULT_STATSBOMB_COMPETITION,
                        help="StatsBomb competition ID (default: 43 - FIFA World Cup)")
    parser.add_argument("--season-id", type=int, default=DEFAULT_STATSBOMB_SEASON,
                        help="StatsBomb season ID (default: 106 - 2022)")
    parser.add_argument("--match-ids", nargs="*", type=int, default=DEFAULT_SAMPLE_MATCH_IDS,
                        help="StatsBomb match IDs to download events and lineups for")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.output_dir)
    meta_dir = Path(args.metadata_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing TouchlineLM sample data download...")
    all_manifest = []

    if args.source in ["statsbomb", "all"]:
        logger.info("Downloading StatsBomb Open Data sample...")
        sb_manifest = download_statsbomb_sample(raw_dir, args.comp_id, args.season_id, args.match_ids)
        all_manifest.extend(sb_manifest)

    if args.source in ["football_data", "all"]:
        logger.info("Downloading Football-Data.co.uk sample...")
        fd_manifest = download_football_data_sample(raw_dir)
        all_manifest.extend(fd_manifest)

    # Save manifest
    manifest_path = meta_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": all_manifest
        }, f, indent=2)

    logger.info(f"Manifest written to {manifest_path}")
    success_count = sum(1 for m in all_manifest if m.get("status") == "success")
    logger.info(f"Download complete: {success_count}/{len(all_manifest)} successful.")


if __name__ == "__main__":
    main()
