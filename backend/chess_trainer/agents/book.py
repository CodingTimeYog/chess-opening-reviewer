"""Opening-book baseline: sample moves in proportion to how often masters played them."""

import chess
import pandas as pd

from chess_trainer.agents.base import Agent
from chess_trainer.config import SEED
from chess_trainer.data.openings import fen_norm

# Weight of a move by the outcome for the player who made it (same as the policy loss).
WIN, DRAW, LOSS = 1.0, 0.75, 0.5


def _result_weights(positions: pd.DataFrame) -> pd.Series:
    white_to_move = positions["ply"] % 2 == 0
    winner_is_white = positions["result"].map({"1-0": True, "0-1": False})
    weights = pd.Series(DRAW, index=positions.index)
    decisive = winner_is_white.notna()
    mover_won = winner_is_white[decisive].astype(bool) == white_to_move[decisive]
    weights[decisive] = mover_won.map({True: WIN, False: LOSS})
    return weights


class BookAgent(Agent):
    name = "book"

    def __init__(self, book: dict[str, dict[str, float]], seed: int = SEED) -> None:
        super().__init__(seed)
        self.book = book  # fen_norm -> {move_uci: weight}

    @classmethod
    def from_positions(
        cls, positions: pd.DataFrame, result_weighted: bool = False, seed: int = SEED
    ) -> "BookAgent":
        """Build the book from the train split only; val and test rows are ignored."""
        train = positions[positions["split"] == "train"]
        weights = _result_weights(train) if result_weighted else pd.Series(1.0, index=train.index)
        totals = weights.groupby([train["fen_norm"], train["move_uci"]]).sum()
        book: dict[str, dict[str, float]] = {}
        for (fen, uci), weight in totals.items():
            book.setdefault(fen, {})[uci] = float(weight)
        return cls(book, seed=seed)

    def moves(self, fen: str) -> set[str]:
        return set(self.book.get(fen, ()))

    def is_fallback(self, board: chess.Board) -> bool:
        return fen_norm(board) not in self.book

    def policy(self, board: chess.Board) -> dict[str, float]:
        entry = self.book.get(fen_norm(board))
        if entry is None:  # out of book: uniform over legal moves
            legal = [move.uci() for move in board.legal_moves]
            return {uci: 1 / len(legal) for uci in legal}
        total = sum(entry.values())
        return {uci: weight / total for uci, weight in entry.items()}
