"""Move quality on held-out positions.

Agreement (every held-out position): top-1 and top-3 agreement of the agent's policy with
the move the master played. Ties in the policy are scored as the expected hit under a
random tie-break, so a uniform fallback over n legal moves scores k/n for top-k.

Engine quality (a fixed, seeded sample of distinct positions, stratified by opening and
by seen / strict): Stockfish scores every legal move, giving for any move
- centipawn loss (CPL) = best score - move score, with scores clipped to +/-1000 cp and
  mates counted as +/-1000;
- WDL expectation loss = best expected score - move's expected score, using
  python-chess's Stockfish WDL model at the position's ply.
Reported for the agent's top move (ties broken by a seeded draw, which is how a uniform
fallback is scored), for the whole policy (expectation over sampling at temperature 1),
and for the master's move as a reference.
"""

import math
import random
import statistics

import chess
import chess.engine
import pandas as pd

from chess_trainer.agents.base import Agent
from chess_trainer.config import SEED
from chess_trainer.data.openings import OPENINGS, board_from
from chess_trainer.engine import clipped_cp

BLUNDER_CPL = 100
SAMPLE_PER_STRATUM = 500


def _pct(hits: float, n: int) -> float | None:
    return round(100 * hits / n, 2) if n else None


# ---------------------------------------------------------------- agreement


def top_k_hit(policy: dict[str, float], move: str, k: int) -> float:
    """Probability that `move` is among the k most probable, ties broken at random."""
    p = policy.get(move)
    if p is None:
        return 0.0
    higher = sum(q > p for q in policy.values())
    ties = sum(q == p for q in policy.values())
    if higher >= k:
        return 0.0
    return min(1.0, (k - higher) / ties)


def top_move(policy: dict[str, float], fen: str, seed: int = SEED) -> str:
    """Most probable move; ties broken by a draw seeded on (seed, position)."""
    best = max(policy.values())
    ties = sorted(move for move, p in policy.items() if p == best)
    return ties[0] if len(ties) == 1 else random.Random(f"{seed}:{fen}").choice(ties)


def agreement(agent: Agent, positions: pd.DataFrame, split: str = "test") -> dict:
    rows = positions[positions["split"] == split]
    first_ply = rows.groupby("fen_norm")["ply"].min().to_dict()
    policies: dict[str, dict[str, float]] = {}
    legal: dict[str, set[str]] = {}
    sanity = {"illegal_moves": 0, "bad_distributions": 0, "illegal_master_moves": 0}
    hit1, hit3 = [], []
    for fen, master in zip(rows["fen_norm"], rows["move_uci"], strict=True):
        if fen not in policies:
            board = board_from(fen, first_ply[fen])  # earliest ply the position occurs
            policies[fen] = agent.policy(board)
            legal[fen] = {move.uci() for move in board.legal_moves}
        policy = policies[fen]
        # Sanity checks are counted per held-out position, not per distinct position.
        sanity["illegal_moves"] += bool(set(policy) - legal[fen])
        total = sum(policy.values())
        sanity["bad_distributions"] += not (
            policy and math.isclose(total, 1.0, abs_tol=1e-6) and min(policy.values()) >= 0
        )
        sanity["illegal_master_moves"] += master not in legal[fen]
        hit1.append(top_k_hit(policy, master, 1))
        hit3.append(top_k_hit(policy, master, 3))
    scored = rows[["opening", "strict"]].assign(hit1=hit1, hit3=hit3)

    def block(frame: pd.DataFrame) -> dict:
        n = len(frame)
        return {"n": n, "top1": _pct(frame["hit1"].sum(), n), "top3": _pct(frame["hit3"].sum(), n)}

    openings = {}
    for oid in OPENINGS:
        sub = scored[scored["opening"] == oid]
        openings[oid] = {"all": block(sub), "strict": block(sub[sub["strict"]])}
    return {"openings": openings, "sanity": {"positions_checked": len(rows), **sanity}}


# ---------------------------------------------------------------- engine quality


def sample_positions(
    positions: pd.DataFrame,
    split: str = "test",
    n_per_stratum: int = SAMPLE_PER_STRATUM,
    seed: int = SEED,
) -> pd.DataFrame:
    """Up to n distinct positions per (opening, seen/strict). Each keeps the master move
    of one randomly chosen game that reached it."""
    rows = positions[positions["split"] == split]
    rows = rows.sort_values(["game_id", "ply"], kind="stable")  # order independent of input
    rows = rows.sample(frac=1, random_state=seed).drop_duplicates(["opening", "fen_norm"])
    rows = rows.assign(stratum=rows["strict"].map({True: "strict", False: "seen"}))
    picked = rows.groupby(["opening", "stratum"], sort=True).head(n_per_stratum)
    columns = ["opening", "stratum", "fen_norm", "ply", "move_uci"]
    columns += [extra for extra in ("eco", "game_id") if extra in picked.columns]
    return picked.sort_values(["opening", "stratum", "fen_norm"])[columns].reset_index(drop=True)


def _expectation(cp: int | None, mate: int | None, ply: int) -> float:
    score = chess.engine.Mate(mate) if mate is not None else chess.engine.Cp(cp)
    return score.wdl(model="sf", ply=ply).expectation()


def _losses(evals: dict, ply: int) -> dict[str, tuple[int, float]]:
    """(centipawn loss, WDL expectation loss) of every legal move."""
    cps = {move: clipped_cp(*score) for move, score in evals.items()}
    exps = {move: _expectation(*score, ply) for move, score in evals.items()}
    best_cp, best_exp = max(cps.values()), max(exps.values())
    return {move: (best_cp - cps[move], best_exp - exps[move]) for move in evals}


def _summary(cpl: list[float], wdl: list[float], blunder: list[float]) -> dict:
    if not cpl:
        return {"mean_cpl": None, "median_cpl": None, "blunder_rate": None, "mean_wdl_loss": None}
    return {
        "mean_cpl": round(statistics.fmean(cpl), 2),
        "median_cpl": round(statistics.median(cpl), 2),
        "blunder_rate": round(100 * statistics.fmean(blunder), 2),
        "mean_wdl_loss": round(statistics.fmean(wdl), 4),
    }


def engine_quality(agent: Agent, sample: pd.DataFrame, evals: dict, seed: int = SEED) -> dict:
    records = []
    for row in sample.itertuples():
        board = board_from(row.fen_norm, row.ply)
        policy = agent.policy(board)
        losses = _losses(evals[row.fen_norm], row.ply)
        top = losses[top_move(policy, row.fen_norm, seed)]
        master = losses[row.move_uci]
        records.append(
            {
                "opening": row.opening,
                "stratum": row.stratum,
                "fallback": agent.is_fallback(board),
                "top_cpl": top[0],
                "top_wdl": top[1],
                "top_blunder": top[0] >= BLUNDER_CPL,
                "policy_cpl": sum(p * losses[m][0] for m, p in policy.items()),
                "policy_wdl": sum(p * losses[m][1] for m, p in policy.items()),
                "policy_blunder": sum(p for m, p in policy.items() if losses[m][0] >= BLUNDER_CPL),
                "master_cpl": master[0],
                "master_wdl": master[1],
                "master_blunder": master[0] >= BLUNDER_CPL,
            }
        )
    columns = list(records[0]) if records else ["opening", "stratum"]
    frame = pd.DataFrame(records, columns=columns)

    def block(sub: pd.DataFrame) -> dict:
        def summary(kind: str) -> dict:
            if sub.empty:
                return _summary([], [], [])
            return _summary(
                list(sub[f"{kind}_cpl"]), list(sub[f"{kind}_wdl"]), list(sub[f"{kind}_blunder"])
            )

        return {
            "n": len(sub),
            "fallback_rate": _pct(sub["fallback"].sum(), len(sub)) if len(sub) else None,
            "top_move": summary("top"),
            "policy": summary("policy"),
            "master": summary("master"),
        }

    openings = {}
    for oid in OPENINGS:
        sub = frame[frame["opening"] == oid]
        openings[oid] = {s: block(sub[sub["stratum"] == s]) for s in ("seen", "strict")}
    return {"openings": openings}


# ---------------------------------------------------------------- report


def _fmt(value: float | None, spec: str = ".1f", suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def quality_markdown(quality: dict) -> str:
    agree, engine = quality["agreement"], quality["engine"]
    names = {oid: OPENINGS[oid].name for oid in agree["openings"]}

    def table(subsets: tuple[str, str], rows: list[tuple[str, callable]], source: dict) -> str:
        cols = [(oid, s) for oid in names for s in subsets]
        head = "| | " + " | ".join(f"{names[o]}, {s}" for o, s in cols) + " |\n"
        head += "|---|" + "---:|" * len(cols) + "\n"
        body = "".join(
            f"| {label} | " + " | ".join(get(source["openings"][o][s]) for o, s in cols) + " |\n"
            for label, get in rows
        )
        return head + body

    sanity = agree["sanity"]
    out = [
        "## Move quality: agreement with the master move (every held-out position)\n\n",
        table(
            ("all", "strict"),
            [
                ("positions", lambda b: f"{b['n']:,}"),
                ("Top-1 agreement", lambda b: _fmt(b["top1"], ".1f", "%")),
                ("Top-3 agreement", lambda b: _fmt(b["top3"], ".1f", "%")),
            ],
            agree,
        ),
        f"\nSanity checks over {sanity['positions_checked']:,} positions: "
        f"{sanity['illegal_moves']:,} policies containing an illegal move, "
        f"{sanity['bad_distributions']:,} policies that are empty or do not sum to 1, "
        f"{sanity['illegal_master_moves']:,} illegal master moves.\n\n",
        f"## Move quality: Stockfish depth {quality['depth']} "
        f"(seeded sample of distinct positions, seed {quality['sample_seed']})\n\n",
    ]
    rows = [
        ("positions", lambda b: f"{b['n']:,}"),
        ("out of book (fallback)", lambda b: _fmt(b["fallback_rate"], ".1f", "%")),
    ]
    for key, label in (("top_move", "top move"), ("policy", "sampled"), ("master", "master")):
        rows += [
            (f"mean CPL, {label}", lambda b, k=key: _fmt(b[k]["mean_cpl"])),
            (f"median CPL, {label}", lambda b, k=key: _fmt(b[k]["median_cpl"])),
            (
                f"blunder rate (CPL >= {BLUNDER_CPL}), {label}",
                lambda b, k=key: _fmt(b[k]["blunder_rate"], ".1f", "%"),
            ),
            (
                f"mean WDL expectation loss, {label}",
                lambda b, k=key: _fmt(b[k]["mean_wdl_loss"], ".3f"),
            ),
        ]
    out.append(table(("seen", "strict"), rows, engine))
    out.append(
        "\n`top move`: the agent's most probable move (a uniform fallback is one seeded random "
        "legal move). `sampled`: expectation over the agent's whole policy. `master`: the move "
        "actually played, for reference. For `sampled`, the median is over per-position "
        "expected CPL.\n"
    )
    return "".join(out)
