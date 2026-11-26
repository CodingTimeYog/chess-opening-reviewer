"""Hybrid agent: book where the book knows the position, the network everywhere else.

Order of precedence at each position:
1. book       - the position occurs in the train games: master move frequencies;
2. repertoire - out of book, but the prep tree has in-repertoire moves: uniform over them;
3. policy     - otherwise the learned network, instead of the book's random fallback.
"""

import chess

from chess_trainer.agents.base import Agent
from chess_trainer.agents.book import BookAgent
from chess_trainer.config import SEED
from chess_trainer.data.openings import fen_norm
from chess_trainer.data.repertoire import repertoire_moves


class HybridAgent(Agent):
    name = "hybrid"

    def __init__(
        self, book: BookAgent, network: Agent, tree: dict | None = None, seed: int = SEED
    ) -> None:
        super().__init__(seed)
        self.book = book
        self.network = network
        self.tree = tree or {}

    def source(self, board: chess.Board) -> str:
        """Which component answers for this position: book, repertoire or policy."""
        fen = fen_norm(board)
        if fen in self.book.book:
            return "book"
        if repertoire_moves(self.tree, fen):
            return "repertoire"
        return "policy"

    def policy(self, board: chess.Board) -> dict[str, float]:
        source = self.source(board)
        if source == "book":
            return self.book.policy(board)
        if source == "repertoire":
            moves = sorted(repertoire_moves(self.tree, fen_norm(board)))
            return {uci: 1 / len(moves) for uci in moves}
        return self.network.policy(board)

    def is_fallback(self, board: chess.Board) -> bool:
        return False  # the network always has an opinion

    def choose(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        if temperature > 0 and self.source(board) == "policy":
            return self.network.choose(board, temperature)  # keeps the network's top-k sampling
        return super().choose(board, temperature)
