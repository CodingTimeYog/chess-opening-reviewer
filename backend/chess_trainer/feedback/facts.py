"""Structured facts about one move: engine verdict, master practice, repertoire, citations.

Everything the feedback text says comes from this dict, and nothing else. It is plain JSON,
so the optional LLM layer can be handed exactly the same facts.
"""

import chess

from chess_trainer.agents.book import BookAgent
from chess_trainer.data.openings import fen_norm
from chess_trainer.engine import clipped_cp

MAX_CITATIONS = 3
THRESHOLDS = ((200, "blunder"), (100, "mistake"), (50, "inaccuracy"))  # centipawn loss
NAG_SYMBOLS = {1: "!", 2: "?", 3: "!!", 4: "??", 5: "!?", 6: "?!"}


def classify_loss(cp_loss: int) -> str:
    return next((label for bound, label in THRESHOLDS if cp_loss >= bound), "good")


def _move(board: chess.Board, uci: str) -> dict:
    return {"uci": uci, "san": board.san(chess.Move.from_uci(uci))}


class FeedbackBuilder:
    """Collects the facts for a move and renders the deterministic template text."""

    def __init__(self, book: BookAgent, tree: dict, retriever, analyser) -> None:
        self.book = book
        self.tree = tree
        self.retriever = retriever
        self.analyser = analyser

    def move_facts(self, board: chess.Board, move: chess.Move, analyse: bool = True) -> dict:
        fen, uci = fen_norm(board), move.uci()
        facts: dict = {
            "fen": fen,
            "ply": board.ply(),
            "side": "white" if board.turn == chess.WHITE else "black",
            "played": _move(board, uci),
            "classification": None,
            "cp_loss": None,
            "best": None,
            "eval_best_cp": None,
            "eval_played_cp": None,
        }
        if analyse:
            verdict = self.analyser.analyse(board, move)
            facts.update(
                classification=classify_loss(verdict.cp_loss),
                cp_loss=verdict.cp_loss,
                best=_move(board, verdict.best_move),
                eval_best_cp=clipped_cp(*verdict.best),
                eval_played_cp=clipped_cp(*verdict.played),
            )

        counts = self.book.book.get(fen, {})
        total = sum(counts.values())
        top = sorted(counts, key=lambda m: (-counts[m], m))[:3]
        facts["master"] = {
            "in_book": bool(counts),
            "games": round(total),
            "top": [
                {
                    **_move(board, m),
                    "games": round(counts[m]),
                    "share": round(100 * counts[m] / total, 1),
                }
                for m in top
            ],
            "played_games": round(counts.get(uci, 0)),
        }

        entries = self.tree.get(fen, {})
        recommended = sorted(m for m, info in entries.items() if info["in_repertoire"])
        note = entries.get(uci)
        facts["repertoire"] = {
            "covered": bool(recommended),
            "moves": [
                {**_move(board, m), "comment": entries[m].get("comment", "")} for m in recommended
            ],
            "played_in_repertoire": uci in recommended,
            "played_note": None
            if note is None or note["in_repertoire"]
            else {
                "nags": [NAG_SYMBOLS[n] for n in note.get("nags", []) if n in NAG_SYMBOLS],
                "comment": note.get("comment", ""),
            },
        }

        facts["citations"] = [
            {
                "game_id": hit.game_id,
                "white": hit.white,
                "black": hit.black,
                "white_elo": hit.white_elo,
                "black_elo": hit.black_elo,
                "year": hit.year,
                "event": hit.event,
                "result": hit.result,
                "move_uci": hit.move_uci,
                "move_san": board.san(chess.Move.from_uci(hit.move_uci)),
            }
            for hit in self.retriever.exact(fen, MAX_CITATIONS)
        ]
        return facts

    def move_feedback(self, board: chess.Board, move: chess.Move, analyse: bool = True) -> dict:
        from chess_trainer.feedback.template import render_move

        facts = self.move_facts(board, move, analyse)
        return {"text": render_move(facts), "facts": facts, "citations": facts["citations"]}
