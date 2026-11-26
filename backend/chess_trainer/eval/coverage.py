"""Opening coverage on held-out positions.

Position coverage: share of held-out positions where the agent produces a non-fallback
move. Reported on all positions and on the strict subset (position never seen in train).

Line coverage: each held-out game is replayed along the moves the masters played. At each
of the agent's own turns the move counts as known when the agent is not falling back and
its most probable move is theory (played from that position in train, or in repertoire).
A line's depth is the number of consecutive known own moves from move one; the line is
covered when the depth reaches `min_own_moves`. Lines too short to have that many own
moves are left out of the denominator. Reported with the agent on each side.

Prep coverage: share of held-out positions with the repertoire side to move for which the
prep tree has an in-repertoire move. It describes the prep files, not the agent.
"""

import chess
import pandas as pd

from chess_trainer.agents.base import Agent
from chess_trainer.config import SEED
from chess_trainer.data.openings import OPENINGS, board_from
from chess_trainer.data.repertoire import repertoire_moves

MIN_OWN_MOVES = 10
# HybridAgent.source() -> label in the report
TIER_LABELS = {"book": "book", "repertoire": "prep", "policy": "resnet"}


def _share(n: int, covered: int) -> dict:
    return {"n": n, "covered": covered, "pct": round(100 * covered / n, 2) if n else None}


def _tiers(rows: pd.DataFrame) -> dict:
    """Share of positions answered by each tier of a tiered agent."""
    counts = rows["tier"].value_counts()
    return {
        "n": len(rows),
        **{
            label: round(100 * int(counts.get(label, 0)) / len(rows), 2) if len(rows) else None
            for label in TIER_LABELS.values()
        },
    }


def _depth(known: list[bool]) -> int:
    return next((i for i, ok in enumerate(known) if not ok), len(known))


def _lines(rows: pd.DataFrame, own_parity: int, min_own_moves: int) -> dict:
    depths = []
    for _, game in rows[rows["ply"] % 2 == own_parity].groupby("game_id"):
        known = list(game.sort_values("ply")["known"])
        if len(known) >= min_own_moves:
            depths.append(_depth(known))
    share = _share(len(depths), sum(d >= min_own_moves for d in depths))
    share["mean_depth"] = round(sum(depths) / len(depths), 2) if depths else None
    return share


def evaluate_coverage(
    agent: Agent,
    positions: pd.DataFrame,
    tree: dict,
    split: str = "test",
    min_own_moves: int = MIN_OWN_MOVES,
    seed: int = SEED,
    git_sha: str = "unknown",
) -> dict:
    train = positions[positions["split"] == "train"]
    theory = train.groupby("fen_norm")["move_uci"].agg(set).to_dict()
    rows = positions[positions["split"] == split].copy()

    # One agent call per distinct position, at the earliest ply it occurs.
    first_ply = rows.groupby("fen_norm")["ply"].min().to_dict()
    in_book, known, in_prep, tier = {}, {}, {}, {}
    tiered = hasattr(agent, "source")
    for fen, ply in first_ply.items():
        board = board_from(fen, ply)
        in_repertoire = repertoire_moves(tree, fen)
        in_book[fen] = not agent.is_fallback(board)
        known[fen] = in_book[fen] and agent.best(board) in theory.get(fen, set()) | in_repertoire
        in_prep[fen] = bool(in_repertoire)
        if tiered:
            tier[fen] = TIER_LABELS[agent.source(board)]
    rows["in_book"] = rows["fen_norm"].map(in_book)
    rows["known"] = rows["fen_norm"].map(known)
    rows["in_prep"] = rows["fen_norm"].map(in_prep)
    if tiered:
        rows["tier"] = rows["fen_norm"].map(tier)

    openings = {}
    for oid, opening in OPENINGS.items():
        sub = rows[rows["opening"] == oid]
        strict = sub[sub["strict"]]
        own_parity = 0 if opening.side == chess.WHITE else 1
        own_turns = sub[sub["ply"] % 2 == own_parity]
        openings[oid] = {
            "positions": {
                "all": _share(len(sub), int(sub["in_book"].sum())),
                "strict": _share(len(strict), int(strict["in_book"].sum())),
            },
            "lines": {
                "repertoire_side": _lines(sub, own_parity, min_own_moves),
                "opponent_side": _lines(sub, 1 - own_parity, min_own_moves),
            },
            "prep": _share(len(own_turns), int(own_turns["in_prep"].sum())),
        }
        if tiered:
            openings[oid]["tiers"] = {"all": _tiers(sub), "strict": _tiers(strict)}
    return {
        "metric": "opening_coverage",
        "agent": agent.name,
        "split": split,
        "seed": seed,
        "git_sha": git_sha,
        "min_own_moves": min_own_moves,
        "openings": openings,
    }


def _pct(share: dict) -> str:
    return "n/a" if share["pct"] is None else f"{share['pct']:.1f}%"


def metrics_markdown(metrics: dict) -> str:
    ops = [(OPENINGS[oid].name, m) for oid, m in metrics["openings"].items()]
    k = metrics["min_own_moves"]
    head = "| | " + " | ".join(name for name, _ in ops) + " |\n|---|" + "---:|" * len(ops) + "\n"

    def row(label: str, get) -> str:
        return f"| {label} | " + " | ".join(get(m) for _, m in ops) + " |\n"

    def cell(share: dict) -> str:
        return f"{_pct(share)} ({share['covered']:,} / {share['n']:,})"

    def tier(subset: str, label: str):
        return lambda m: (
            "n/a" if m["tiers"][subset][label] is None else f"{m['tiers'][subset][label]:.1f}%"
        )

    tier_section = []
    if all("tiers" in m for _, m in ops):
        tier_section = [
            "\n## Which tier answers (book, then prep repertoire, then ResNet)\n\n",
            head,
            *[
                row(f"% {label}, {subset} positions", tier(subset, label))
                for subset in ("all", "strict")
                for label in TIER_LABELS.values()
            ],
        ]

    def depth(share: dict) -> str:
        return "n/a" if share["mean_depth"] is None else f"{share['mean_depth']:.2f}"

    return "".join(
        [
            f"# Opening coverage: {metrics['agent']} agent, {metrics['split']} split\n\n",
            f"Git `{metrics['git_sha']}`, seed {metrics['seed']}. ",
            "Definitions are in `chess_trainer/eval/coverage.py`.\n\n",
            "## Position coverage (non-fallback move)\n\n",
            head,
            row("all positions", lambda m: cell(m["positions"]["all"])),
            row("strict positions (never seen in train)", lambda m: cell(m["positions"]["strict"])),
            f"\n## Line coverage (at least {k} consecutive known own moves)\n\n",
            head,
            row("agent on the repertoire side", lambda m: cell(m["lines"]["repertoire_side"])),
            row("mean depth, repertoire side", lambda m: depth(m["lines"]["repertoire_side"])),
            row("agent on the opponent side", lambda m: cell(m["lines"]["opponent_side"])),
            row("mean depth, opponent side", lambda m: depth(m["lines"]["opponent_side"])),
            "\n## Prep coverage (repertoire side to move, prep tree has a repertoire move)\n\n",
            head,
            row("positions", lambda m: cell(m["prep"])),
            *tier_section,
        ]
    )
