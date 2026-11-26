"""Run the evaluation harness.

uv run python -m chess_trainer.eval --agent book --split test
uv run python -m chess_trainer.eval --agent policy --checkpoint resnet_main --compare book
uv run python -m chess_trainer.eval --agent hybrid --checkpoint resnet_main \
    --compare book resnet-main
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from chess_trainer import config
from chess_trainer.agents.base import Agent
from chess_trainer.agents.book import BookAgent
from chess_trainer.agents.hybrid import HybridAgent
from chess_trainer.data.repertoire import load_tree
from chess_trainer.engine import EvalCache, evaluate_positions
from chess_trainer.eval.compare import comparison_markdown
from chess_trainer.eval.coverage import evaluate_coverage, metrics_markdown
from chess_trainer.eval.quality import (
    SAMPLE_PER_STRATUM,
    agreement,
    engine_quality,
    quality_markdown,
    sample_positions,
)


def build_agent(args: argparse.Namespace, positions: pd.DataFrame, tree: dict) -> tuple[Agent, str]:
    """The agent and the name of its run directory."""
    if args.agent in ("policy", "hybrid"):
        from chess_trainer.agents.policy import PolicyAgent  # imports torch

        if not args.checkpoint:
            raise SystemExit(f"--agent {args.agent} needs --checkpoint (a name or a path)")
        path = Path(args.checkpoint)
        if not path.exists():
            path = config.CHECKPOINT_DIR / f"{args.checkpoint}.pt"
        network = PolicyAgent.from_checkpoint(path, seed=args.seed)
        if args.agent == "policy":
            return network, path.stem.replace("_", "-")
        book = BookAgent.from_positions(positions, seed=args.seed)
        return HybridAgent(book, network, tree, seed=args.seed), "hybrid"
    weighted = args.agent == "book-weighted"
    return BookAgent.from_positions(positions, result_weighted=weighted, seed=args.seed), args.agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent on held-out positions.")
    parser.add_argument(
        "--agent", choices=["book", "book-weighted", "policy", "hybrid"], required=True
    )
    parser.add_argument("--checkpoint", help="policy checkpoint: a path or a name")
    parser.add_argument(
        "--compare", nargs="+", default=[], help="runs to put side by side with this one"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--depth", type=int, default=config.ENGINE_DEPTH)
    parser.add_argument(
        "--sample", type=int, default=SAMPLE_PER_STRATUM, help="positions per opening and stratum"
    )
    parser.add_argument("--workers", type=int, default=config.ENGINE_WORKERS)
    args = parser.parse_args()

    positions = pd.read_parquet(args.data / "positions.parquet")
    tree = load_tree(args.data / "repertoire.json")
    agent, run_name = build_agent(args, positions, tree)
    agent.name = run_name
    sha = config.git_sha()

    coverage = evaluate_coverage(
        agent, positions, tree, split=args.split, seed=args.seed, git_sha=sha
    )
    sample = sample_positions(
        positions, split=args.split, n_per_stratum=args.sample, seed=args.seed
    )
    evals = evaluate_positions(
        sample["fen_norm"], depth=args.depth, cache=EvalCache(), workers=args.workers
    )
    quality = {
        "depth": args.depth,
        "sample_seed": args.seed,
        "sample_per_stratum": args.sample,
        "agreement": agreement(agent, positions, split=args.split),
        "engine": engine_quality(agent, sample, evals, seed=args.seed),
    }
    metrics = {
        "agent": run_name,
        "split": args.split,
        "seed": args.seed,
        "git_sha": sha,
        "coverage": coverage,
        "quality": quality,
    }
    report = metrics_markdown(coverage) + "\n" + quality_markdown(quality)
    if args.compare:
        paths = [args.results / f"{name}-{args.split}" / "metrics.json" for name in args.compare]
        baselines = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        report = comparison_markdown(metrics, baselines) + "\n" + report

    run_dir = args.results / f"{run_name}-{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "metrics.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
