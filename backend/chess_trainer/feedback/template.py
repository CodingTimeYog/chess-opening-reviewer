"""Deterministic feedback text. Every sentence restates a field of the facts dict."""

MAX_COMMENT = 300
ARTICLES = {"inaccuracy": "an", "mistake": "a", "blunder": "a"}


def _quote(comment: str) -> str:
    comment = (
        comment if len(comment) <= MAX_COMMENT else comment[: MAX_COMMENT - 3].rstrip() + "..."
    )
    return f'"{comment}"'


def _engine(facts: dict) -> str | None:
    label, played = facts["classification"], facts["played"]["san"]
    if label is None:
        return None
    best = facts["best"]["san"]
    if facts["best"]["uci"] == facts["played"]["uci"] or facts["cp_loss"] == 0:
        return f"{played} is the engine's first choice."
    pawns = f"{facts['cp_loss'] / 100:.1f} pawns"
    if label == "good":
        return f"{played} is fine: within {pawns} of the engine's first choice, {best}."
    return (
        f"{played} is {ARTICLES[label]} {label}: it loses about {pawns}. Stockfish prefers {best}."
    )


def _master(facts: dict) -> str:
    master, played = facts["master"], facts["played"]
    if not master["in_book"]:
        return "No master game in the corpus reached this position."
    top = master["top"][0]
    text = (
        f"Masters most often played {top['san']} "
        f"({top['games']} of {master['games']} games, {top['share']:.0f}%)"
    )
    if top["uci"] == played["uci"]:
        return text + "."
    if master["played_games"]:
        return f"{text}; {played['san']} was played in {master['played_games']}."
    return f"{text}; {played['san']} was not played."


def _repertoire(facts: dict) -> str | None:
    repertoire, played = facts["repertoire"], facts["played"]["san"]
    parts = []
    if repertoire["covered"]:
        moves = repertoire["moves"]
        if repertoire["played_in_repertoire"]:
            mine = next(m for m in moves if m["uci"] == facts["played"]["uci"])
            parts.append(f"{played} is your repertoire move.")
            if mine["comment"]:
                parts.append(f"Prep note: {_quote(mine['comment'])}")
        else:
            names = " and ".join(m["san"] for m in moves)
            noun = "move is" if len(moves) == 1 else "moves are"
            sentence = f"Your repertoire {noun} {names}"
            comment = next((m["comment"] for m in moves if m["comment"]), "")
            parts.append(f"{sentence}: {_quote(comment)}" if comment else f"{sentence}.")
    note = repertoire["played_note"]
    if note is not None:
        marks = "".join(note["nags"])
        sentence = (
            f"Your prep marks {played} as {marks}" if marks else f"Your prep discusses {played}"
        )
        parts.append(
            f"{sentence}: {_quote(note['comment'])}" if note["comment"] else f"{sentence}."
        )
    return " ".join(parts) or None


def _citations(facts: dict) -> str | None:
    if not facts["citations"]:
        return None
    games = "; ".join(
        f"{c['white']} - {c['black']}, {c['event']}"
        f"{'' if c['year'] is None else ' ' + str(c['year'])} ({c['result']}), "
        f"continued {c['move_san']}"
        for c in facts["citations"]
    )
    return f"Example games: {games}."


def _join(*parts: str | None) -> str:
    return " ".join(part for part in parts if part)


def render_move(facts: dict) -> str:
    """Full feedback on one move."""
    return _join(_engine(facts), _master(facts), _repertoire(facts), _citations(facts))


def render_event(event_type: str, facts: dict) -> str:
    played = facts.get("played", {}).get("san")
    if event_type == "left_repertoire":
        return _join(f"{played} left your repertoire.", _repertoire(facts), _citations(facts))
    if event_type == "left_book":
        games = facts["master"]["games"]
        lead = (
            f"{played} has not been played in the {games} master "
            f"game{'s' if games != 1 else ''} that reached this position."
        )
        return _join(lead, _master(facts), _citations(facts))
    if event_type == "eval_swing":
        return render_move(facts)
    if event_type == "opening_end":
        if facts["reason"] == "ply_cap":
            return f"The opening phase is over: move {facts['max_ply'] // 2} has been reached."
        return (
            "The opening phase is over: the last "
            f"{facts['out_limit']} moves were outside both the book and your repertoire."
        )
    raise ValueError(f"unknown event type: {event_type}")
