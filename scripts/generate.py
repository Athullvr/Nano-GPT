#!/usr/bin/env python3
"""
TouchlineLM CLI Text Generation Tool
Provides interactive and batch text generation from trained TouchlineNanoGPT checkpoints.
"""

import argparse
import sys
from pathlib import Path
import logging

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.engine import TouchlineInferenceEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


EVAL_PROMPTS = [
    (
        "Match Narrative (Final)",
        "<|narrative|> # Match Report: Argentina vs France\n"
        "**Competition Stage**: Final | **Date**: 2022-12-18\n"
        "**Final Scoreline**: Argentina"
    ),
    (
        "Player Dossier (Messi)",
        "<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini\n"
        "**Position**: Right Wing\n\n"
        "## 1. Distribution and Ball Progression\n"
        "Operating in possession, the player completed"
    ),
    (
        "Match Narrative (Quarter-finals)",
        "<|narrative|> # Match Report: Croatia vs Brazil\n"
        "**Competition Stage**: Quarter-finals\n"
        "**Final Scoreline**:"
    ),
    (
        "Tactical Q&A",
        "<|tactical_qa|> In modern football analytics, expected goals (xG) measures"
    ),
]


def run_benchmark(
    engine: TouchlineInferenceEngine,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool = True,
):
    """Executes benchmark across standard evaluation prompts and prints performance metrics."""
    print("\n" + "=" * 75)
    print(f"TOUCHLINELM BENCHMARK SUITE (Device: {engine.device.upper()} | Cache: {'ENABLED' if use_cache else 'DISABLED (Uncached)'})")
    print("=" * 75)

    total_tokens = 0
    total_time = 0.0

    for idx, (label, prompt) in enumerate(EVAL_PROMPTS, 1):
        print(f"\n[{idx}/4] Prompt: {label}")
        print("-" * 50)
        output = engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=42 + idx,
            use_cache=use_cache,
        )

        print(output.full_text)
        print("-" * 50)
        print(
            f"Telemetry: {output.new_tokens} tokens in {output.latency_seconds:.3f}s "
            f"({output.tokens_per_second:.1f} tok/s) | Cache: {'ON' if output.use_cache else 'OFF'} | "
            f"Stopped on EOS: {output.stopped_on_eos}"
        )

        total_tokens += output.new_tokens
        total_time += output.latency_seconds

    avg_speed = total_tokens / total_time if total_time > 0 else 0.0
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY:")
    print(f"  • Total Generated Tokens: {total_tokens}")
    print(f"  • Total Inference Time:   {total_time:.3f}s")
    print(f"  • Average Generation Speed: {avg_speed:.2f} tokens/second")
    print(f"  • Mode:                   {'KV Cached (O(1) decode)' if use_cache else 'Uncached (O(T) recomputation)'}")
    print("=" * 75 + "\n")


def interactive_mode(
    engine: TouchlineInferenceEngine,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool = True,
):
    """Runs interactive terminal REPL for ad-hoc prompt testing."""
    print("\n" + "=" * 70)
    print(f"TouchlineLM Interactive Inference REPL (Cache: {'ENABLED' if use_cache else 'DISABLED'})")
    print("Type your tactical prompt and press Enter. Enter 'exit' or Ctrl+D to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            prompt = input("TouchlineLM > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", ":q"]:
                print("Exiting interactive REPL.")
                break

            output = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache,
            )

            print("\n--- MODEL GENERATION ---")
            print(output.generated_text)
            print("-" * 70)
            print(
                f"Generated {output.new_tokens} tokens in {output.latency_seconds:.3f}s "
                f"({output.tokens_per_second:.1f} tok/s | Cache: {'ON' if output.use_cache else 'OFF'})\n"
            )

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive REPL.")
            break


def main():
    parser = argparse.ArgumentParser(description="TouchlineLM Local Text Generation CLI")

    # Artifact paths
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final_model.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="configs/tokenizer",
        help="Directory containing tokenizer artifacts (vocab.json, merges.txt)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.json",
        help="Path to model configuration JSON file",
    )

    # Prompt options
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for autoregressive generation",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to text file containing prompt",
    )

    # Sampling hyperparameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Maximum new tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7; <=0.0 for greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k filtering cutoff (default: 40)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling cumulative probability (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic generation",
    )
    parser.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="Disable automatic termination upon generating EOS (<|endoftext|>)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable per-layer KV caching and recompute entire sequence at each step (legacy baseline)",
    )

    # Execution controls
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device ('auto', 'cpu', or 'cuda')",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Execute standardized benchmark across evaluation prompts",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal REPL",
    )

    args = parser.parse_args()

    use_cache = not args.no_cache

    # Load inference engine
    engine = TouchlineInferenceEngine.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        model_config_path=args.model_config,
        device=args.device,
    )

    # Mode 1: Benchmark suite
    if args.benchmark:
        run_benchmark(
            engine=engine,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=use_cache,
        )
        return

    # Mode 2: Interactive REPL
    if args.interactive:
        interactive_mode(
            engine=engine,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=use_cache,
        )
        return

    # Mode 3: Single prompt generation
    prompt = args.prompt
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
            sys.exit(1)
        prompt = prompt_path.read_text(encoding="utf-8")

    if not prompt:
        # Default prompt if none provided
        prompt = (
            "<|narrative|> # Match Report: Argentina vs France\n"
            "**Competition Stage**: Final | **Date**: 2022-12-18\n"
            "**Final Scoreline**: Argentina"
        )
        print("Note: No prompt provided; using standard World Cup Final evaluation prompt.\n")

    output = engine.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        stop_at_eos=not args.no_eos_stop,
        use_cache=use_cache,
    )

    print("=" * 70)
    print("PROMPT:")
    print(output.prompt)
    print("-" * 70)
    print("GENERATED TEXT:")
    print(output.generated_text)
    print("=" * 70)
    print(
        f"Telemetry: {output.new_tokens} tokens generated in {output.latency_seconds:.3f}s "
        f"({output.tokens_per_second:.1f} tokens/s) on {output.device.upper()} | "
        f"Cache: {'ON (KV Cache)' if output.use_cache else 'OFF (Uncached)'} | "
        f"Stopped on EOS: {output.stopped_on_eos}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
