"""Side-by-side table of two eval runs on the strict (never seen in train) subset."""

from chess_trainer.data.openings import OPENINGS


def _fmt(value: float | None, spec: str, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def comparison_markdown(ours: dict, baseline: dict) -> str:
    """Both arguments are full metrics dicts as written to metrics.json."""
    runs = [(baseline["agent"], baseline), (ours["agent"], ours)]
    openings = list(ours["quality"]["agreement"]["openings"])
    columns = [(oid, name, run) for oid in openings for name, run in runs]

    def agree(key: str):
        return lambda run, oid: _fmt(
            run["quality"]["agreement"]["openings"][oid]["strict"][key], ".1f", "%"
        )

    def engine(who: str, key: str, spec: str, suffix: str = ""):
        return lambda run, oid: _fmt(
            run["quality"]["engine"]["openings"][oid]["strict"][who][key], spec, suffix
        )

    rows = [
        (
            "position coverage (non-fallback)",
            lambda run, oid: _fmt(
                run["coverage"]["openings"][oid]["positions"]["strict"]["pct"], ".1f", "%"
            ),
        ),
        ("Top-1 agreement with master", agree("top1")),
        ("Top-3 agreement with master", agree("top3")),
        ("mean CPL, top move", engine("top_move", "mean_cpl", ".1f")),
        ("median CPL, top move", engine("top_move", "median_cpl", ".1f")),
        ("blunder rate (CPL >= 100), top move", engine("top_move", "blunder_rate", ".1f", "%")),
        ("mean WDL expectation loss, top move", engine("top_move", "mean_wdl_loss", ".3f")),
        ("mean CPL, sampling from the policy", engine("policy", "mean_cpl", ".1f")),
        ("mean CPL of the master's move (reference)", engine("master", "mean_cpl", ".1f")),
    ]
    head = "| | " + " | ".join(f"{OPENINGS[o].name}: {name}" for o, name, _ in columns) + " |\n"
    head += "|---|" + "---:|" * len(columns) + "\n"
    body = "".join(
        f"| {label} | " + " | ".join(get(run, oid) for oid, _, run in columns) + " |\n"
        for label, get in rows
    )
    counts = ours["quality"]["agreement"]["openings"]
    sizes = ", ".join(f"{OPENINGS[o].name} {counts[o]['strict']['n']:,}" for o in openings)
    return (
        f"## {ours['agent']} vs {baseline['agent']} on strict positions "
        f"({ours['split']} split, never seen in train)\n\n"
        f"Agreement and coverage use every strict position ({sizes}); the Stockfish rows use "
        f"the same seeded sample for both agents.\n\n" + head + body
    )
