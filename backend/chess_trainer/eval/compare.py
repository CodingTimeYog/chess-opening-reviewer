"""Side-by-side tables of several eval runs: strict positions, then seen positions."""

from collections.abc import Callable

from chess_trainer.data.openings import OPENINGS

Getter = Callable[[dict, str], str]


def _fmt(value: float | None, spec: str, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def _agree(subset: str, key: str) -> Getter:
    return lambda run, oid: _fmt(
        run["quality"]["agreement"]["openings"][oid][subset][key], ".1f", "%"
    )


def _engine(stratum: str, who: str, key: str, spec: str, suffix: str = "") -> Getter:
    return lambda run, oid: _fmt(
        run["quality"]["engine"]["openings"][oid][stratum][who][key], spec, suffix
    )


def _coverage(subset: str) -> Getter:
    return lambda run, oid: _fmt(
        run["coverage"]["openings"][oid]["positions"][subset]["pct"], ".1f", "%"
    )


def _engine_rows(stratum: str) -> list[tuple[str, Getter]]:
    return [
        ("mean CPL, top move", _engine(stratum, "top_move", "mean_cpl", ".1f")),
        ("median CPL, top move", _engine(stratum, "top_move", "median_cpl", ".1f")),
        (
            "blunder rate (CPL >= 100), top move",
            _engine(stratum, "top_move", "blunder_rate", ".1f", "%"),
        ),
        (
            "mean WDL expectation loss, top move",
            _engine(stratum, "top_move", "mean_wdl_loss", ".3f"),
        ),
        ("mean CPL, sampling from the policy", _engine(stratum, "policy", "mean_cpl", ".1f")),
        (
            "mean CPL of the master's move (reference)",
            _engine(stratum, "master", "mean_cpl", ".1f"),
        ),
    ]


def comparison_markdown(ours: dict, baselines: dict | list[dict]) -> str:
    """Arguments are full metrics dicts as written to metrics.json. Baselines come first."""
    if isinstance(baselines, dict):
        baselines = [baselines]
    runs = [*baselines, ours]
    openings = list(ours["quality"]["agreement"]["openings"])
    columns = [(oid, run) for oid in openings for run in runs]

    def table(rows: list[tuple[str, Getter]]) -> str:
        head = "| | " + " | ".join(f"{OPENINGS[o].name}: {r['agent']}" for o, r in columns)
        head += " |\n|---|" + "---:|" * len(columns) + "\n"
        return head + "".join(
            f"| {label} | " + " | ".join(get(run, oid) for oid, run in columns) + " |\n"
            for label, get in rows
        )

    counts = ours["quality"]["agreement"]["openings"]
    sizes = ", ".join(f"{OPENINGS[o].name} {counts[o]['strict']['n']:,}" for o in openings)
    others = ", ".join(run["agent"] for run in baselines)
    strict_rows = [
        ("position coverage (non-fallback)", _coverage("strict")),
        ("Top-1 agreement with master", _agree("strict", "top1")),
        ("Top-3 agreement with master", _agree("strict", "top3")),
        *_engine_rows("strict"),
    ]
    seen_rows = [
        ("Top-1 agreement with master, all positions", _agree("all", "top1")),
        ("Top-3 agreement with master, all positions", _agree("all", "top3")),
        *_engine_rows("seen"),
    ]
    return (
        f"## {ours['agent']} vs {others} on strict positions "
        f"({ours['split']} split, never seen in train)\n\n"
        f"Agreement and coverage use every strict position ({sizes}); the Stockfish rows use "
        f"the same seeded sample for every agent.\n\n"
        + table(strict_rows)
        + f"\n## {ours['agent']} vs {others}: all positions and the seen sample\n\n"
        "Agreement is over every held-out position; the Stockfish rows are the sample of "
        "positions that also occur in train.\n\n" + table(seen_rows)
    )
