"""
TouchlineLM - Phase 2 End-to-End Corpus Engineering Pipeline
Orchestrates:
1. Batch extraction across all sample matches
2. Deterministic metrics calculation
3. Intermediate data persistence (data/intermediate/)
4. Multi-tier text generation (data/processed/corpus/)
5. 100% automated statistical fact-checking and validation
6. Text diversity and vocabulary analysis
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extraction.match_extractor import StatsBombExtractor
from src.metrics.calculator import ExtractedMatchData, MetricsCalculator, TeamMatchMetrics, PlayerMatchMetrics
from src.corpus.generator import CorpusGenerator
from src.validation.fact_checker import FactChecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("process_corpus")


def analyze_corpus_diversity(texts: List[str]) -> Dict[str, Any]:
    """Calculate lexical diversity, word counts, and n-gram statistics."""
    total_chars = sum(len(t) for t in texts)
    
    # Tokenize words cleanly (lowercase alphanumeric)
    words = []
    for t in texts:
        tokens = re.findall(r"\b[A-Za-z0-9_\-\.%]+\b", t.lower())
        words.extend(tokens)

    total_words = len(words)
    unique_words = set(words)
    vocab_size = len(unique_words)
    ttr = round((vocab_size / total_words) * 100, 2) if total_words > 0 else 0.0

    # 2-grams and 3-grams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]

    distinct_bigrams = len(set(bigrams))
    distinct_trigrams = len(set(trigrams))
    bigram_diversity = round((distinct_bigrams / len(bigrams)) * 100, 2) if bigrams else 0.0
    trigram_diversity = round((distinct_trigrams / len(trigrams)) * 100, 2) if trigrams else 0.0

    # Top domain words
    word_freq = Counter(words)
    stop_words = {
        "and", "the", "in", "of", "a", "to", "with", "for", "from", "at", "as", "by", "was",
        "on", "this", "an", "that", "during", "while", "across", "neither", "during", "or"
    }
    top_domain_words = [
        {"word": w, "count": c}
        for w, c in word_freq.most_common(50)
        if w not in stop_words and len(w) > 2
    ][:20]

    return {
        "total_characters": total_chars,
        "total_words": total_words,
        "estimated_tokens_ratio_0_75": int(total_words * 1.3),
        "vocabulary_size": vocab_size,
        "type_token_ratio_ttr_pct": ttr,
        "distinct_bigrams": distinct_bigrams,
        "bigram_diversity_pct": bigram_diversity,
        "distinct_trigrams": distinct_trigrams,
        "trigram_diversity_pct": trigram_diversity,
        "top_domain_vocabulary": top_domain_words
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Phase 2 Corpus Engineering Pipeline.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--intermediate-dir", type=str, default="data/intermediate", help="Intermediate output dir")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed corpus output dir")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata", help="Metadata output dir")
    parser.add_argument("--comp-id", type=int, default=43, help="Competition ID")
    parser.add_argument("--season-id", type=int, default=106, help="Season ID")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    inter_dir = Path(args.intermediate_dir)
    proc_dir = Path(args.processed_dir)
    meta_dir = Path(args.metadata_dir)

    inter_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = proc_dir / "corpus"
    (corpus_dir / "narratives").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "players").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "tactical_qa").mkdir(parents=True, exist_ok=True)

    extractor = StatsBombExtractor(raw_dir)

    # Discover available event files
    events_dir = raw_dir / "statsbomb" / "events"
    if not events_dir.exists():
        logger.error(f"No events directory found at {events_dir}")
        sys.exit(1)

    available_match_ids = [int(f.stem) for f in sorted(events_dir.glob("*.json"))]
    logger.info(f"Discovered {len(available_match_ids)} match event files in {events_dir}")

    extracted_matches_list = []
    extracted_players_list = []
    match_data_map: Dict[int, ExtractedMatchData] = {}
    total_events_processed = 0

    # 1. Extraction and deterministic metrics calculation
    logger.info("Executing deterministic extraction and metrics calculations...")
    for m_id in available_match_ids:
        try:
            match_meta = extractor.load_match_metadata(args.comp_id, args.season_id, m_id)
            events = extractor.load_events(m_id)
            total_events_processed += len(events)

            lineups_dict = {
                match_meta.home_team.team_name: match_meta.home_team.lineup,
                match_meta.away_team.team_name: match_meta.away_team.lineup
            }

            match_data = MetricsCalculator.process_match_events(
                match_id=m_id,
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

            match_data_map[m_id] = match_data

            # Prepare intermediate serializable objects
            match_dict = {
                "match_id": m_id,
                "fixture": match_data.fixture,
                "stage": match_data.competition_stage,
                "date": match_data.match_date,
                "home_team": asdict(match_data.home_metrics),
                "away_team": asdict(match_data.away_metrics),
                "phases": [asdict(ph) for ph in match_data.phases],
                "goals": match_data.goals_summary
            }
            extracted_matches_list.append(match_dict)

            for p_metric in match_data.player_metrics.values():
                p_dict = asdict(p_metric)
                p_dict["match_id"] = m_id
                p_dict["fixture"] = match_data.fixture
                extracted_players_list.append(p_dict)

            logger.info(f"Processed match {m_id}: {match_data.fixture} ({len(events)} events, {len(match_data.player_metrics)} players)")
        except Exception as e:
            logger.error(f"Error processing match {m_id}: {e}")

    # Save intermediate records
    with open(inter_dir / "extracted_matches.json", "w", encoding="utf-8") as f:
        json.dump(extracted_matches_list, f, indent=2)

    with open(inter_dir / "extracted_players.json", "w", encoding="utf-8") as f:
        json.dump(extracted_players_list, f, indent=2)

    logger.info(f"Saved {len(extracted_matches_list)} match records and {len(extracted_players_list)} player records to {inter_dir}")

    # 2. Generate corpus samples
    logger.info("Generating multi-tier football intelligence training corpus...")
    corpus_samples = []
    all_sample_texts = []

    for m_id, match_data in match_data_map.items():
        # Tier 1: Match narrative
        narrative = CorpusGenerator.generate_match_narrative(match_data)
        corpus_samples.append(narrative)
        all_sample_texts.append(narrative["content"])
        with open(corpus_dir / "narratives" / f"{m_id}_narrative.md", "w", encoding="utf-8") as f:
            f.write(narrative["content"])

        # Tier 2: Player dossiers
        for p_metric in match_data.player_metrics.values():
            dossier = CorpusGenerator.generate_player_dossier(p_metric, match_data)
            if dossier:
                corpus_samples.append(dossier)
                all_sample_texts.append(dossier["content"])
                with open(corpus_dir / "players" / f"{m_id}_{p_metric.player_id}.md", "w", encoding="utf-8") as f:
                    f.write(dossier["content"])

        # Tier 3: Tactical Q&A
        qa = CorpusGenerator.generate_tactical_qa(match_data)
        corpus_samples.append(qa)
        all_sample_texts.append(qa["content"])
        with open(corpus_dir / "tactical_qa" / f"{m_id}_qa.md", "w", encoding="utf-8") as f:
            f.write(qa["content"])

    # Save consolidated JSONL and consolidated text
    jsonl_path = corpus_dir / "touchline_corpus.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in corpus_samples:
            f.write(json.dumps(s) + "\n")

    txt_path = corpus_dir / "touchline_corpus.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n<|endoftext|>\n\n".join(all_sample_texts))

    logger.info(f"Generated {len(corpus_samples)} total corpus samples saved to {corpus_dir}")

    # 3. Fact-checking validation
    logger.info("Executing 100% automated statistical validation against ground truth...")
    fact_checker = FactChecker(match_data_map)
    total_claims_verified = 0

    for sample in corpus_samples:
        stype = sample["sample_type"]
        if stype == "match_narrative":
            claims, _ = fact_checker.verify_match_narrative(sample)
            total_claims_verified += claims
        elif stype == "player_dossier":
            claims, _ = fact_checker.verify_player_dossier(sample)
            total_claims_verified += claims
        elif stype == "tactical_qa":
            claims, _ = fact_checker.verify_tactical_qa(sample)
            total_claims_verified += claims

    val_report = {
        "total_samples_validated": len(corpus_samples),
        "total_claims_verified": total_claims_verified,
        "total_violations": len(fact_checker.violations),
        "violations_list": fact_checker.violations,
        "status": "PASS" if len(fact_checker.violations) == 0 else "FAIL"
    }
    with open(meta_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    logger.info(f"Statistical validation complete: {total_claims_verified} claims verified with {len(fact_checker.violations)} violations.")

    # 4. Text diversity and vocabulary analysis
    logger.info("Computing text diversity, vocabulary, and n-gram statistics...")
    diversity_report = analyze_corpus_diversity(all_sample_texts)
    diversity_report["sample_counts"] = {
        "total_samples": len(corpus_samples),
        "narratives": sum(1 for s in corpus_samples if s["sample_type"] == "match_narrative"),
        "player_dossiers": sum(1 for s in corpus_samples if s["sample_type"] == "player_dossier"),
        "tactical_qa": sum(1 for s in corpus_samples if s["sample_type"] == "tactical_qa")
    }
    diversity_report["extracted_counts"] = {
        "total_matches": len(match_data_map),
        "total_events_processed": total_events_processed,
        "total_player_records": len(extracted_players_list)
    }

    with open(meta_dir / "corpus_diversity_report.json", "w", encoding="utf-8") as f:
        json.dump(diversity_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 2: CORPUS ENGINEERING SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Matches Processed:     {len(match_data_map)}")
    print(f"Total Events Processed:      {total_events_processed}")
    print(f"Total Player Records:        {len(extracted_players_list)}")
    print(f"Total Generated Samples:     {len(corpus_samples)}")
    print(f"  - Match Narratives:        {diversity_report['sample_counts']['narratives']}")
    print(f"  - Player Dossiers:         {diversity_report['sample_counts']['player_dossiers']}")
    print(f"  - Tactical Q&A:            {diversity_report['sample_counts']['tactical_qa']}")
    print(f"Corpus Total Words:          {diversity_report['total_words']:,}")
    print(f"Corpus Total Characters:     {diversity_report['total_characters']:,}")
    print(f"Vocabulary Size:             {diversity_report['vocabulary_size']:,} unique words")
    print(f"Type-Token Ratio (TTR):      {diversity_report['type_token_ratio_ttr_pct']}%")
    print(f"Trigram Diversity:           {diversity_report['trigram_diversity_pct']}%")
    print("-" * 60)
    print(f"Statistical Claims Verified: {total_claims_verified}")
    print(f"Statistical Violations:      {len(fact_checker.violations)}")
    print(f"Validation Status:           {val_report['status']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
