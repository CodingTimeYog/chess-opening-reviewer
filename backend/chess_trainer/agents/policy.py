"""The learned agent: a trained PolicyNet behind the Agent interface."""

from pathlib import Path

import chess
import numpy as np
import torch

from chess_trainer.agents.base import Agent
from chess_trainer.agents.encoding import (
    encode_bitboards,
    encode_board,
    legal_indices,
    planes_from_bitboards,
)
from chess_trainer.agents.model import PolicyNet, legal_logits
from chess_trainer.config import SEED
from chess_trainer.data.openings import board_from

TOP_K = 5


class PolicyAgent(Agent):
    name = "policy"

    def __init__(self, model: PolicyNet, seed: int = SEED, top_k: int = TOP_K) -> None:
        super().__init__(seed)
        self.model = model.eval()
        self.top_k = top_k
        self._cache: dict[tuple[str, int], tuple[dict[str, float], np.ndarray]] = {}

    @classmethod
    def from_checkpoint(cls, path: Path, seed: int = SEED) -> "PolicyAgent":
        saved = torch.load(path, weights_only=False)
        model = PolicyNet(**saved["config"])
        model.load_state_dict(saved["model"])
        return cls(model, seed=seed)

    def _forward(self, board: chess.Board) -> tuple[dict[str, float], np.ndarray]:
        key = (board.epd(), board.ply())
        if key not in self._cache:
            moves, from_to, promo = legal_indices(board)
            with torch.no_grad():
                out = self.model(torch.from_numpy(encode_board(board))[None])
                if moves:
                    logits = legal_logits(
                        out.from_to,
                        out.promo,
                        torch.from_numpy(from_to)[None],
                        torch.from_numpy(promo)[None],
                    )
                    probs = torch.softmax(logits[0].double(), dim=0).tolist()
                else:
                    probs = []
            policy = {move.uci(): p for move, p in zip(moves, probs, strict=True)}
            self._cache[key] = (policy, out.features[0].numpy())
        return self._cache[key]

    def policy(self, board: chess.Board) -> dict[str, float]:
        return dict(self._forward(board)[0])

    def embed(self, board: chess.Board) -> np.ndarray:
        """Pooled trunk features, used for nearest-neighbour retrieval."""
        return self._forward(board)[1]

    def embed_batch(self, fens: list[str], plies: list[int], batch_size: int = 1024) -> np.ndarray:
        """Embeddings of many positions at once: (N, channels) float32."""
        bitboards = np.stack([encode_bitboards(board_from(fen)) for fen in fens])
        plies = np.asarray(plies)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(fens), batch_size):
                stop = start + batch_size
                planes = planes_from_bitboards(bitboards[start:stop], plies[start:stop])
                chunks.append(self.model(torch.from_numpy(planes)).features.numpy())
        return np.concatenate(chunks)

    def choose(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Sample among the top-k moves for variety. Temperature 0 picks the best move."""
        policy = self.policy(board)
        if not policy:
            raise ValueError("no legal moves")
        if temperature <= 0:
            return chess.Move.from_uci(self.best(board))
        top = sorted(sorted(policy), key=policy.__getitem__, reverse=True)[: self.top_k]
        weights = [policy[move] ** (1 / temperature) for move in top]
        return chess.Move.from_uci(self._rng.choices(top, weights=weights)[0])
