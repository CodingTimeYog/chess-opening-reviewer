from pathlib import Path

import chess
import pandas as pd
import pytest

from chess_trainer.data.ingest import build_corpus, fen_norm
from chess_trainer.data.openings import OPENINGS, ROOT_FENS, classify
from chess_trainer.data.repertoire import build_tree, repertoire_moves
from chess_trainer.data.split import assign_splits, mark_strict

FIXTURES = Path(__file__).parent / "fixtures"


def moves(san: str) -> list[chess.Move]:
    board, out = chess.Board(), []
    for tok in san.split():
        out.append(board.push_san(tok))
    return out


@pytest.fixture(scope="module")
def corpus():
    return build_corpus([FIXTURES / "corpus.pgn"])


def test_fen_norm_drops_counters_and_unusable_en_passant():
    board = chess.Board()
    board.push_san("e4")
    assert fen_norm(board) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"


def test_classify_by_position_includes_transpositions():
    assert classify(moves("e4 e5 Nf3 Nc6 Bb5 a6")) == "ruy_lopez"
    assert classify(moves("e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3")) == "taimanov"
    assert classify(moves("e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nc3")) == "taimanov"
    assert classify(moves("e4 e5 Nf3 Nc6 Bc4 Bc5")) is None
    assert classify(moves("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6")) is None


def test_rating_floor_differs_per_opening():
    assert OPENINGS["ruy_lopez"].min_elo == 2500
    assert OPENINGS["taimanov"].min_elo == 2400


def test_corpus_filters_and_logs_each_stage(corpus):
    games, _, stats = corpus
    assert games.groupby("opening").size().to_dict() == {"ruy_lopez": 2, "taimanov": 2}
    assert stats["parsed"] == 11
    assert stats["openings"]["ruy_lopez"]["stages"] == {
        "reached_root": 6,
        "both_rated": 5,
        "rating_ok": 4,
        "has_result": 3,
        "unique": 2,
    }
    assert stats["openings"]["taimanov"]["stages"] == {
        "reached_root": 3,
        "both_rated": 3,
        "rating_ok": 2,
        "has_result": 2,
        "unique": 2,
    }
    assert games["game_id"].is_unique


def test_games_keep_a_clean_pgn_and_positions_do_not(corpus):
    games, positions, _ = corpus
    pgn = games.loc[games["white"] == "Alpha, A", "pgn"].item()
    assert pgn.startswith('[Event "Fixture Open"]')
    assert "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6" in pgn and pgn.rstrip().endswith("1-0")
    assert "pgn" not in positions.columns


def test_positions_one_row_per_ply_with_metadata(corpus):
    games, positions, _ = corpus
    gid = games.loc[games["white"] == "Alpha, A", "game_id"].item()
    rows = positions[positions["game_id"] == gid].sort_values("ply")
    assert list(rows["ply"]) == list(range(14))
    assert rows.iloc[0]["fen_norm"] == fen_norm(chess.Board())
    assert rows.iloc[0]["move_uci"] == "e2e4"
    assert rows.iloc[4]["move_uci"] == "f1b5"
    assert rows.iloc[5]["fen_norm"] == ROOT_FENS["ruy_lopez"]
    assert set(rows["result"]) == {"1-0"}
    assert set(rows["white_elo"]) == {2600}
    assert set(rows["year"]) == {2015}
    assert set(rows["opening"]) == {"ruy_lopez"}


def test_positions_truncated_at_max_ply():
    _, positions, _ = build_corpus([FIXTURES / "corpus.pgn"], max_ply=8)
    assert positions["ply"].max() == 7


def _fake_games(n_per_opening=100):
    return pd.DataFrame(
        [
            {"game_id": f"{o}-{i:03d}", "opening": o}
            for o in ("ruy_lopez", "taimanov")
            for i in range(n_per_opening)
        ]
    )


def test_split_is_by_game_seeded_and_80_10_10_per_opening():
    games = _fake_games()
    a = assign_splits(games, seed=7)
    assert a.equals(assign_splits(games.sample(frac=1, random_state=1), seed=7).loc[games.index])
    assert not a.equals(assign_splits(games, seed=8))
    counts = games.assign(split=a).groupby(["opening", "split"]).size()
    for o in ("ruy_lopez", "taimanov"):
        assert (counts[o]["train"], counts[o]["val"], counts[o]["test"]) == (80, 10, 10)


def test_no_game_spans_two_splits(corpus):
    _, positions, _ = corpus
    assert (positions.groupby("game_id")["split"].nunique() == 1).all()
    assert set(positions["split"]) <= {"train", "val", "test"}


def test_strict_marks_held_out_positions_unseen_in_train():
    positions = pd.DataFrame(
        {
            "split": ["train", "train", "test", "test", "val"],
            "fen_norm": ["a", "b", "a", "c", "d"],
        }
    )
    assert list(mark_strict(positions)) == [False, False, False, True, True]


def test_repertoire_tree_keeps_only_root_chapters():
    tree, stats = build_tree([FIXTURES / "prep.pgn"])
    assert stats == {
        "chapters": 3,
        "kept": {"ruy_lopez": 1},
        "skipped_fen": 1,
        "skipped_off_root": 1,
    }
    root = tree[ROOT_FENS["ruy_lopez"]]
    assert root["a7a6"]["main"] is True
    assert root["a7a6"]["comment"] == "Main line. (note)"
    assert root["a7a6"]["san"] == "a6"
    assert root["g8f6"]["main"] is False
    assert root["a7a6"]["chapters"] == ["1. Ruy Lopez Test"]
    after_nf3 = fen_norm(
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    )
    assert set(tree[after_nf3]) == {"b8c6"}


def _fen_after(san: str) -> str:
    board = chess.Board()
    for tok in san.split():
        board.push_san(tok)
    return fen_norm(board)


def test_repertoire_side_only_main_moves_without_dubious_nags():
    tree, _ = build_tree([FIXTURES / "prep.pgn"])
    after_a6 = tree[_fen_after("e4 e5 Nf3 Nc6 Bb5 a6")]
    assert after_a6["b5a4"]["side"] == "own"
    assert after_a6["b5a4"]["in_repertoire"] is True
    assert after_a6["b5c6"]["in_repertoire"] is False  # side variation
    assert after_a6["b5c6"]["nags"] == [6]
    assert after_a6["b5c4"]["in_repertoire"] is False
    # a main move tagged ?! is not a recommendation either
    d4 = tree[_fen_after("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6")]["d2d4"]
    assert d4["main"] is True and d4["in_repertoire"] is False
    assert repertoire_moves(tree, _fen_after("e4 e5 Nf3 Nc6 Bb5 a6")) == {"b5a4"}
    assert repertoire_moves(tree, "not a position") == set()


def test_opponent_side_keeps_every_branch_on_the_repertoire_path():
    tree, _ = build_tree([FIXTURES / "prep.pgn"])
    root = tree[ROOT_FENS["ruy_lopez"]]
    assert root["a7a6"]["side"] == root["g8f6"]["side"] == "opp"
    assert root["a7a6"]["in_repertoire"] and root["g8f6"]["in_repertoire"]
    # the recommended reply inside an opponent side branch is repertoire
    assert tree[_fen_after("e4 e5 Nf3 Nc6 Bb5 Nf6")]["e1g1"]["in_repertoire"] is True
    # moves below a rejected own-side move are kept for reference but are off the path
    assert tree[_fen_after("e4 e5 Nf3 Nc6 Bb5 a6 Bxc6")]["d7c6"]["in_repertoire"] is False
    assert tree[_fen_after("e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6")]["e1g1"]["in_repertoire"] is False
