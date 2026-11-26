"""Stockfish evaluations of every legal move in a position, cached in SQLite.

A position is analysed once with MultiPV set to the number of legal moves, so the cache
holds a score for each (fen_norm, move, depth). Any agent's move can then be scored
without calling the engine again. Scores are from the side to move's point of view,
stored as centipawns or mate distance exactly as Stockfish reports them.

`LiveAnalyser` is the fast path for the UI: two short searches for one move.
"""

import sqlite3
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from chess_trainer import config

CP_CLIP = 1000
Score = tuple[int | None, int | None]  # (centipawns, mate in n); exactly one is set
MoveEvals = dict[str, Score]


class EvalCache:
    def __init__(self, path: Path | None = None) -> None:
        path = Path(path or config.CACHE_DIR / "engine.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS evals ("
            "fen TEXT NOT NULL, move TEXT NOT NULL, depth INTEGER NOT NULL, "
            "cp INTEGER, mate INTEGER, PRIMARY KEY (fen, move, depth))"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS live ("
            "fen TEXT NOT NULL, move TEXT NOT NULL, depth INTEGER NOT NULL, "
            "cp INTEGER, mate INTEGER, best INTEGER NOT NULL, PRIMARY KEY (fen, move, depth))"
        )

    def get(self, fen: str, depth: int) -> MoveEvals | None:
        rows = self._db.execute(
            "SELECT move, cp, mate FROM evals WHERE fen = ? AND depth = ?", (fen, depth)
        ).fetchall()
        return {move: (cp, mate) for move, cp, mate in rows} or None

    def put(self, fen: str, depth: int, evals: MoveEvals) -> None:
        """All moves of a position are written in one transaction, so a position is
        either fully cached or absent."""
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO evals VALUES (?, ?, ?, ?, ?)",
                [(fen, move, depth, cp, mate) for move, (cp, mate) in evals.items()],
            )

    # Live analysis stores single moves, so it has its own table: a position in `evals`
    # is always complete, a position in `live` holds only the moves that were searched.
    def get_live(self, fen: str, depth: int) -> dict[str, tuple[Score, bool]]:
        rows = self._db.execute(
            "SELECT move, cp, mate, best FROM live WHERE fen = ? AND depth = ?", (fen, depth)
        ).fetchall()
        return {move: ((cp, mate), bool(best)) for move, cp, mate, best in rows}

    def put_live(self, fen: str, depth: int, move: str, score: Score, best: bool) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO live VALUES (?, ?, ?, ?, ?, ?)",
                (fen, move, depth, score[0], score[1], int(best)),
            )

    def close(self) -> None:
        self._db.close()


def clipped_cp(cp: int | None, mate: int | None) -> int:
    """Centipawns bounded to +/-CP_CLIP; a mate counts as the bound."""
    if mate is not None:
        return CP_CLIP if mate > 0 else -CP_CLIP
    return max(-CP_CLIP, min(CP_CLIP, cp))


@dataclass(frozen=True)
class MoveAnalysis:
    """Engine verdict on one move: the best move and both scores, mover's point of view."""

    best_move: str
    best: Score
    played: Score

    @property
    def cp_loss(self) -> int:
        return max(0, clipped_cp(*self.best) - clipped_cp(*self.played))


class LiveAnalyser:
    """Fast feedback on a single move: one search for the best move and one restricted to
    the move played, instead of scoring every legal move. Results are cached by
    (fen_norm, move, depth). A position already fully analysed offline needs no search."""

    def __init__(
        self,
        cache: EvalCache | None = None,
        depth: int = config.ENGINE_DEPTH,
        stockfish_path: Path | None = None,
        use_full_cache: bool = True,
        engine: chess.engine.SimpleEngine | None = None,
    ) -> None:
        self.cache = cache or EvalCache()
        self.depth = depth
        self.use_full_cache = use_full_cache
        self._path = str(stockfish_path or config.STOCKFISH_PATH)
        self._engine = engine

    def _search(self, board: chess.Board, move: chess.Move | None) -> tuple[str, Score]:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
            self._engine.configure({"Threads": 1, "Hash": 32})
        limit = chess.engine.Limit(depth=self.depth)
        root_moves = [move] if move is not None else None
        info = self._engine.analyse(board, limit, root_moves=root_moves, game=object())
        return info["pv"][0].uci(), _score(info, board.turn)

    def analyse(self, board: chess.Board, move: chess.Move) -> MoveAnalysis:
        fen, uci = board.epd(), move.uci()
        if self.use_full_cache:
            full = self.cache.get(fen, self.depth)
            if full and uci in full:
                best = max(sorted(full), key=lambda m: clipped_cp(*full[m]))
                return MoveAnalysis(best, full[best], full[uci])
        live = self.cache.get_live(fen, self.depth)
        best = next((m for m, (_, is_best) in live.items() if is_best), None)
        if best is None:
            best, score = self._search(board, None)
            self.cache.put_live(fen, self.depth, best, score, best=True)
            live[best] = (score, True)
        if uci not in live:
            _, score = self._search(board, move)
            self.cache.put_live(fen, self.depth, uci, score, best=False)
            live[uci] = (score, False)
        return MoveAnalysis(best, live[best][0], live[uci][0])

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None


def _score(info: chess.engine.InfoDict, turn: chess.Color) -> Score:
    pov = info["score"].pov(turn)
    return pov.score(), pov.mate()


def analyse_all_moves(engine: chess.engine.SimpleEngine, fen: str, depth: int) -> MoveEvals:
    board = chess.Board(f"{fen} 0 1")
    legal = list(board.legal_moves)
    if not legal:
        return {}
    limit = chess.engine.Limit(depth=depth)
    # A fresh `game` makes python-chess send ucinewgame, which clears the hash table, so
    # a position's scores do not depend on what was analysed before it.
    infos = engine.analyse(board, limit, multipv=len(legal), game=object())
    evals = {info["pv"][0].uci(): _score(info, board.turn) for info in infos if info.get("pv")}
    for move in legal:  # anything MultiPV left out gets its own restricted search
        if move.uci() not in evals:
            info = engine.analyse(board, limit, root_moves=[move], game=object())
            evals[move.uci()] = _score(info, board.turn)
    return evals


def evaluate_positions(
    fens: Iterable[str],
    depth: int = config.ENGINE_DEPTH,
    cache: EvalCache | None = None,
    workers: int = config.ENGINE_WORKERS,
    stockfish_path: Path | None = None,
    progress: bool = True,
) -> dict[str, MoveEvals]:
    """Scores for every legal move of every position, from the cache where possible."""
    cache = cache or EvalCache()
    fens = list(dict.fromkeys(fens))
    out: dict[str, MoveEvals] = {}
    todo = []
    for fen in fens:
        hit = cache.get(fen, depth)
        if hit is None:
            todo.append(fen)
        else:
            out[fen] = hit
    if progress:
        print(
            f"engine: {len(out):,} cached, {len(todo):,} to analyse at depth {depth}",
            file=sys.stderr,
        )
    if not todo:
        return out

    path = str(stockfish_path or config.STOCKFISH_PATH)
    local = threading.local()
    engines: list[chess.engine.SimpleEngine] = []
    lock = threading.Lock()

    def work(fen: str) -> tuple[str, MoveEvals]:
        if not hasattr(local, "engine"):
            local.engine = chess.engine.SimpleEngine.popen_uci(path)
            local.engine.configure({"Threads": 1, "Hash": 32})
            with lock:
                engines.append(local.engine)
        return fen, analyse_all_moves(local.engine, fen, depth)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, fen) for fen in todo]
            for done, future in enumerate(as_completed(futures), start=1):
                fen, evals = future.result()
                cache.put(fen, depth, evals)  # written as they finish; a rerun resumes
                out[fen] = evals
                if progress and done % 100 == 0:
                    print(f"engine: {done:,} / {len(todo):,}", file=sys.stderr)
    finally:
        for engine in engines:
            engine.quit()
    return out
