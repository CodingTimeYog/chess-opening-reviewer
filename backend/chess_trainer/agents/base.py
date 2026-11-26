"""The interface every agent implements."""

import random
from abc import ABC, abstractmethod

import chess

from chess_trainer.config import SEED


class Agent(ABC):
    name: str

    def __init__(self, seed: int = SEED) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def policy(self, board: chess.Board) -> dict[str, float]:
        """Probability of each move (UCI) from this position. Sums to 1."""

    def is_fallback(self, board: chess.Board) -> bool:
        """True when the agent has no knowledge of this position and is guessing."""
        return False

    def best(self, board: chess.Board) -> str | None:
        """Most probable move; ties broken by UCI string so the result is deterministic."""
        policy = self.policy(board)
        return max(sorted(policy), key=policy.__getitem__) if policy else None

    def choose(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Sample a move. Temperature 0 picks the most probable move."""
        policy = self.policy(board)
        if not policy:
            raise ValueError("no legal moves")
        if temperature <= 0:
            return chess.Move.from_uci(self.best(board))
        moves = sorted(policy)
        weights = [policy[m] ** (1 / temperature) for m in moves]
        return chess.Move.from_uci(self._rng.choices(moves, weights=weights)[0])
