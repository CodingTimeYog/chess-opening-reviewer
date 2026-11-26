"""Trigger engine: turns the facts of each move into events. One instance per game.

Built so far: `left_repertoire`, `left_book` (each once per game per side), `eval_swing`
(centipawn loss >= 50 / 100 / 200) and `opening_end` (ply cap, or the agent out of both
book and repertoire for three consecutive own moves). `structure_shift`, `castled`,
`queens_off` and `minor_piece_trade` are deferred until the API is up.
"""

from dataclasses import dataclass

import chess

from chess_trainer.config import MAX_PLY
from chess_trainer.feedback.template import render_event

OUT_LIMIT = 3


@dataclass(frozen=True)
class Event:
    type: str
    ply: int
    severity: str  # info, inaccuracy, mistake or blunder
    side: str  # white or black: who made the move
    facts: dict
    feedback: dict  # {text, facts, citations}

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "ply": self.ply,
            "severity": self.severity,
            "side": self.side,
            "feedback": self.feedback,
        }


class TriggerEngine:
    def __init__(
        self,
        agent_color: chess.Color | None = None,
        max_ply: int = MAX_PLY,
        out_limit: int = OUT_LIMIT,
    ) -> None:
        """`agent_color`: whose out-of-book streak ends the opening (None: either side)."""
        self.agent_side = (
            None if agent_color is None else ("white" if agent_color == chess.WHITE else "black")
        )
        self.max_ply = max_ply
        self.out_limit = out_limit
        self._fired: set[tuple[str, str]] = set()
        self._out_streak = {"white": 0, "black": 0}
        self._ended = False

    def _event(self, kind: str, severity: str, facts: dict, move_facts: dict) -> Event:
        feedback = {
            "text": render_event(kind, facts),
            "facts": facts,
            "citations": facts.get("citations", []),
        }
        return Event(kind, move_facts["ply"], severity, move_facts["side"], facts, feedback)

    def on_move(self, facts: dict) -> list[Event]:
        """`facts` is `FeedbackBuilder.move_facts` for the move just played."""
        side, events = facts["side"], []
        repertoire, master = facts["repertoire"], facts["master"]

        if repertoire["covered"] and not repertoire["played_in_repertoire"]:
            if ("left_repertoire", side) not in self._fired:
                self._fired.add(("left_repertoire", side))
                events.append(self._event("left_repertoire", "info", facts, facts))
        if master["in_book"] and master["played_games"] == 0:
            if ("left_book", side) not in self._fired:
                self._fired.add(("left_book", side))
                events.append(self._event("left_book", "info", facts, facts))
        if facts["classification"] not in (None, "good"):
            events.append(self._event("eval_swing", facts["classification"], facts, facts))

        outside = not master["in_book"] and not repertoire["covered"]
        self._out_streak[side] = self._out_streak[side] + 1 if outside else 0
        if not self._ended:
            reason = None
            if facts["ply"] + 1 >= self.max_ply:
                reason = "ply_cap"
            elif self.agent_side in (None, side) and self._out_streak[side] >= self.out_limit:
                reason = "out_of_book"
            if reason:
                self._ended = True
                end = {"reason": reason, "max_ply": self.max_ply, "out_limit": self.out_limit}
                events.append(self._event("opening_end", "info", end, facts))
        return events
