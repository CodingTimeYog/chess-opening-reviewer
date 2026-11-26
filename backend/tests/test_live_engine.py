import chess
import chess.engine
import pytest

from chess_trainer import config
from chess_trainer.data.openings import fen_norm
from chess_trainer.engine import EvalCache, LiveAnalyser, MoveAnalysis


class FakeEngine:
    """Scores e2e4 +30, a2a3 -90 and everything else -20; prefers e2e4."""

    SCORES = {"e2e4": 30, "a2a3": -90}

    def __init__(self):
        self.calls = []

    def analyse(self, board, limit, root_moves=None, game=None, **_):
        move = root_moves[0] if root_moves else chess.Move.from_uci("e2e4")
        self.calls.append(move.uci() if root_moves else "best")
        cp = self.SCORES.get(move.uci(), -20)
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn), "pv": [move]}

    def quit(self):
        pass


@pytest.fixture
def analyser(tmp_path):
    engine = FakeEngine()
    return LiveAnalyser(EvalCache(tmp_path / "e.sqlite"), depth=14, engine=engine), engine


def test_two_searches_best_move_then_the_move_played(analyser):
    live, engine = analyser
    result = live.analyse(chess.Board(), chess.Move.from_uci("a2a3"))
    assert engine.calls == ["best", "a2a3"]
    assert result == MoveAnalysis(best_move="e2e4", best=(30, None), played=(-90, None))
    assert result.cp_loss == 120


def test_playing_the_best_move_needs_one_search_and_loses_nothing(analyser):
    live, engine = analyser
    result = live.analyse(chess.Board(), chess.Move.from_uci("e2e4"))
    assert engine.calls == ["best"]
    assert result.cp_loss == 0 and result.best == result.played


def test_results_are_cached_per_fen_move_and_depth(analyser, tmp_path):
    live, engine = analyser
    live.analyse(chess.Board(), chess.Move.from_uci("a2a3"))
    live.analyse(chess.Board(), chess.Move.from_uci("a2a3"))
    live.analyse(chess.Board(), chess.Move.from_uci("d2d4"))  # best is cached, d2d4 is new
    assert engine.calls == ["best", "a2a3", "d2d4"]
    # a second analyser on the same file reuses the cache
    again = FakeEngine()
    other = LiveAnalyser(EvalCache(tmp_path / "e.sqlite"), depth=14, engine=again)
    assert other.analyse(chess.Board(), chess.Move.from_uci("a2a3")).cp_loss == 120
    assert again.calls == []
    # a different depth is a different cache entry
    deeper = LiveAnalyser(EvalCache(tmp_path / "e.sqlite"), depth=16, engine=again)
    deeper.analyse(chess.Board(), chess.Move.from_uci("a2a3"))
    assert again.calls == ["best", "a2a3"]


def test_live_entries_do_not_look_like_a_fully_analysed_position(analyser, tmp_path):
    live, _ = analyser
    live.analyse(chess.Board(), chess.Move.from_uci("a2a3"))
    assert EvalCache(tmp_path / "e.sqlite").get(fen_norm(chess.Board()), 14) is None


def test_a_fully_analysed_position_is_answered_without_the_engine(tmp_path):
    cache = EvalCache(tmp_path / "e.sqlite")
    start = fen_norm(chess.Board())
    full = {m.uci(): (-20, None) for m in chess.Board().legal_moves}
    full.update({"d2d4": (40, None), "a2a3": (None, -3)})
    cache.put(start, 14, full)
    engine = FakeEngine()
    result = LiveAnalyser(cache, depth=14, engine=engine).analyse(
        chess.Board(), chess.Move.from_uci("a2a3")
    )
    assert engine.calls == []
    assert result.best_move == "d2d4" and result.cp_loss == 1040
    # the correctness eval must be able to ignore it and really search
    fresh = LiveAnalyser(cache, depth=14, engine=engine, use_full_cache=False)
    assert fresh.analyse(chess.Board(), chess.Move.from_uci("a2a3")).best_move == "e2e4"
    assert engine.calls == ["best", "a2a3"]


def test_a_played_move_that_scores_above_the_best_search_loses_nothing(tmp_path):
    class Noisy(FakeEngine):
        SCORES = {"e2e4": 30, "d2d4": 34}

    live = LiveAnalyser(EvalCache(tmp_path / "e.sqlite"), depth=14, engine=Noisy())
    assert live.analyse(chess.Board(), chess.Move.from_uci("d2d4")).cp_loss == 0


@pytest.mark.skipif(not config.STOCKFISH_PATH.exists(), reason="Stockfish not installed")
def test_real_stockfish_flags_a_queen_blunder(tmp_path):
    live = LiveAnalyser(EvalCache(tmp_path / "e.sqlite"), depth=8)
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3")
    try:
        blunder = live.analyse(board, chess.Move.from_uci("h5f7"))  # Qxf7+ loses the queen
        fine = live.analyse(board, chess.Move.from_uci("g1f3"))
    finally:
        live.close()
    assert blunder.cp_loss >= 300
    assert fine.cp_loss < 150
    assert chess.Move.from_uci(blunder.best_move) in board.legal_moves
