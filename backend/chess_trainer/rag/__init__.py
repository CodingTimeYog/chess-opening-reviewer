"""Master-game retrieval for a position.

Exact `fen_norm` matches come first. If there are fewer than k, the list is filled with
games from the nearest indexed positions by cosine similarity of policy-network embeddings,
within the same opening. At equal score the game with the higher average rating ranks
first, then the most recent year. A game appears at most once.

The eval index holds train games only (so a held-out game cannot retrieve itself); the app
index holds every game (`splits=None`).
"""

import hashlib
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chess_trainer import config
from chess_trainer.config import SEED
from chess_trainer.data.openings import board_from

Embed = Callable[[str, int], np.ndarray]  # (fen_norm, ply) -> vector
_NEIGHBOUR_POSITIONS = 16  # nearest positions expanded into games


@dataclass(frozen=True)
class Hit:
    game_id: str
    score: float  # 1.0 for an exact match, else cosine similarity of the matched position
    exact: bool
    fen_norm: str  # the indexed position that matched
    ply: int
    move_uci: str  # the move played from that position
    opening: str
    eco: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    year: int | None
    event: str
    result: str

    def to_dict(self) -> dict:
        return asdict(self)


def _sorted_rows(positions: pd.DataFrame, splits: tuple[str, ...] | None) -> pd.DataFrame:
    rows = positions if splits is None else positions[positions["split"].isin(splits)]
    rows = rows.assign(
        avg_elo=(rows["white_elo"] + rows["black_elo"]) / 2,
        year_key=rows["year"].astype("Float64").fillna(-1.0).astype(float),
    )
    order = ["avg_elo", "year_key", "game_id", "ply"]
    return rows.sort_values(order, ascending=[False, False, True, True]).reset_index(drop=True)


def _hit(row, score: float, exact: bool) -> Hit:
    return Hit(
        game_id=row.game_id,
        score=score,
        exact=exact,
        fen_norm=row.fen_norm,
        ply=int(row.ply),
        move_uci=row.move_uci,
        opening=row.opening,
        eco=row.eco,
        white=row.white,
        black=row.black,
        white_elo=int(row.white_elo),
        black_elo=int(row.black_elo),
        year=None if pd.isna(row.year) else int(row.year),
        event=row.event,
        result=row.result,
    )


class Retriever:
    def __init__(
        self,
        positions: pd.DataFrame,
        embed: Embed | None = None,
        splits: tuple[str, ...] | None = ("train",),
    ) -> None:
        self.rows = _sorted_rows(positions, splits)  # best-ranked game first
        self._by_fen = self.rows.groupby("fen_norm", sort=False).indices
        self.embed = embed
        self._fens: list[str] = []
        self._matrix: np.ndarray | None = None
        self._in_opening: dict[str, np.ndarray] = {}

    def indexed_positions(self) -> pd.DataFrame:
        """Each distinct position with the earliest ply it occurs at."""
        return self.rows.groupby("fen_norm", sort=True)["ply"].min().reset_index()

    def set_embeddings(self, fens: list[str], matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._fens = list(fens)
        self._matrix = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
        row_of = {fen: i for i, fen in enumerate(self._fens)}
        self._in_opening = {}
        for opening, group in self.rows.groupby("opening"):
            mask = np.zeros(len(self._fens), dtype=bool)
            mask[[row_of[fen] for fen in group["fen_norm"].unique() if fen in row_of]] = True
            self._in_opening[opening] = mask

    def _games_at(self, fen: str) -> list:
        index = self._by_fen.get(fen)
        if index is None:
            return []
        return list(self.rows.iloc[np.sort(index)].drop_duplicates("game_id").itertuples())

    def exact(self, fen: str, k: int) -> list[Hit]:
        return [_hit(row, 1.0, True) for row in self._games_at(fen)[:k]]

    def retrieve(self, fen: str, k: int = 5, ply: int = 0, opening: str | None = None) -> list[Hit]:
        hits = self.exact(fen, k)
        if len(hits) >= k:
            return hits
        taken = {hit.game_id for hit in hits}
        return hits + self.similar(fen, k - len(hits), ply, opening, exclude=taken)

    def similar(
        self,
        fen: str,
        k: int = 5,
        ply: int = 0,
        opening: str | None = None,
        exclude: set[str] | frozenset[str] = frozenset(),
    ) -> list[Hit]:
        """Games from the nearest other positions by embedding; never an exact match."""
        hits: list[Hit] = []
        if k <= 0 or self._matrix is None or self.embed is None:
            return hits
        query = np.asarray(self.embed(fen, ply), dtype=np.float32)
        similarity = self._matrix @ (query / max(float(np.linalg.norm(query)), 1e-12))
        if opening in self._in_opening:
            similarity = np.where(self._in_opening[opening], similarity, -np.inf)
        nearest = np.argsort(-similarity, kind="stable")[:_NEIGHBOUR_POSITIONS]
        taken = set(exclude)
        candidates = []
        for i in nearest:
            if not np.isfinite(similarity[i]) or self._fens[i] == fen:
                continue
            score = round(float(similarity[i]), 6)
            candidates += [
                (score, row)
                for row in self._games_at(self._fens[i])
                # positions before the roots are shared by both openings; their games are not
                if opening is None or row.opening == opening
            ]
        # equal score: `rows` is already ordered by rating then year, so a stable sort keeps it
        candidates.sort(key=lambda c: (-c[0], -c[1].avg_elo, -c[1].year_key))
        for score, row in candidates:
            if len(hits) >= k:
                break
            if row.game_id not in taken:
                taken.add(row.game_id)
                hits.append(_hit(row, score, False))
        return hits


def index_embeddings(retriever: Retriever, agent, batch_size: int = 1024) -> None:
    """Embed every indexed position with the policy network and attach the query embedder."""
    indexed = retriever.indexed_positions()
    matrix = agent.embed_batch(list(indexed["fen_norm"]), list(indexed["ply"]), batch_size)
    retriever.set_embeddings(list(indexed["fen_norm"]), matrix)
    retriever.embed = lambda fen, ply: agent.embed(board_from(fen, ply))


def checkpoint_signature(path: Path) -> str:
    """Hash of a checkpoint file, stored with an index so a stale one is never reused."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def load_or_build_index(retriever: Retriever, agent, path: Path, signature: str) -> str:
    """Attach embeddings from `path` if it was built from the same checkpoint and the same
    indexed positions; otherwise embed the positions and write the file. Returns which."""
    path = Path(path)
    retriever.embed = lambda fen, ply: agent.embed(board_from(fen, ply))
    indexed = retriever.indexed_positions()
    fens = list(indexed["fen_norm"])
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) == signature and list(saved["fens"]) == fens:
                retriever.set_embeddings(fens, saved["matrix"])
                return "loaded"
    matrix = agent.embed_batch(fens, list(indexed["ply"]))
    retriever.set_embeddings(fens, matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signature=signature, fens=np.array(fens), matrix=matrix)
    return "built"


def index_path(splits: tuple[str, ...] | None) -> Path:
    """`retrieval_train.npz` for the eval index, `retrieval_all.npz` for the app index."""
    return config.PROCESSED_DIR / f"retrieval_{'all' if splits is None else '_'.join(splits)}.npz"


class RandomRetriever:
    """Control: k random indexed games of the same opening, at the query's ply."""

    def __init__(
        self,
        positions: pd.DataFrame,
        splits: tuple[str, ...] | None = ("train",),
        seed: int = SEED,
    ) -> None:
        self.rows = _sorted_rows(positions, splits)
        self._at = self.rows.groupby(["opening", "ply"], sort=False).indices
        self.seed = seed

    def retrieve(self, fen: str, k: int = 5, ply: int = 0, opening: str | None = None) -> list[Hit]:
        index = self._at.get((opening, ply))
        if index is None:
            return []
        rng = random.Random(f"{self.seed}:{fen}")
        picked = rng.sample(sorted(int(i) for i in index), min(k, len(index)))
        rows = self.rows.iloc[picked].itertuples()
        return [_hit(row, 0.0, row.fen_norm == fen) for row in rows]
