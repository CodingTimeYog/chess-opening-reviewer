import chess
import pytest
from helpers import RUY_A, RUY_C, game_rows, positions_frame
from test_eval_quality import FixedAgent

from chess_trainer.agents.book import BookAgent
from chess_trainer.agents.hybrid import HybridAgent
from chess_trainer.data.openings import fen_norm
from chess_trainer.eval.coverage import evaluate_coverage
from chess_trainer.eval.quality import agreement


class SpyAgent(FixedAgent):
    """Stands in for the network: always prefers the first legal move, and counts calls."""

    name = "spy"

    def __init__(self):
        super().__init__({})
        self.calls = 0

    def policy(self, board):
        self.calls += 1
        legal = sorted(m.uci() for m in board.legal_moves)
        rest = 0.1 / (len(legal) - 1)
        return {uci: (0.9 if i == 0 else rest) for i, uci in enumerate(legal)}


def board_after(san: str) -> chess.Board:
    board = chess.Board()
    for tok in san.split():
        board.push_san(tok)
    return board


@pytest.fixture
def positions():
    return positions_frame(game_rows("train1", RUY_A, "train"), game_rows("test1", RUY_C, "test"))


@pytest.fixture
def agents(positions):
    book, net = BookAgent.from_positions(positions), SpyAgent()
    out_of_book = board_after("e4 e5 Nf3 Nc6 Bb5 Nf6")
    tree = {
        fen_norm(out_of_book): {
            "e1g1": {"in_repertoire": True},
            "d2d3": {"in_repertoire": True},
            "b5c6": {"in_repertoire": False},
        },
        # a repertoire entry for a book position must not override the book
        fen_norm(chess.Board()): {"d2d4": {"in_repertoire": True}},
    }
    return HybridAgent(book, net, tree), book, net


def test_in_book_positions_use_the_book_and_never_call_the_network(agents):
    hybrid, book, net = agents
    for san in ("", "e4", "e4 e5 Nf3 Nc6 Bb5"):
        board = board_after(san)
        assert hybrid.source(board) == "book"
        assert hybrid.policy(board) == book.policy(board)
    assert net.calls == 0


def test_out_of_book_with_prep_plays_the_repertoire_moves(agents):
    hybrid, _, net = agents
    board = board_after("e4 e5 Nf3 Nc6 Bb5 Nf6")
    assert hybrid.source(board) == "repertoire"
    assert hybrid.policy(board) == {"d2d3": 0.5, "e1g1": 0.5}
    assert net.calls == 0


def test_out_of_book_and_out_of_prep_hands_off_to_the_network(agents):
    hybrid, book, net = agents
    board = board_after("e4 e5 Nf3 Nc6 Bb5 Nf6 O-O")
    assert book.is_fallback(board)
    assert hybrid.source(board) == "policy"
    assert hybrid.policy(board) == net.policy(board)
    # the network's answer, not the book's uniform fallback
    assert max(hybrid.policy(board).values()) == 0.9
    assert hybrid.is_fallback(board) is False


def test_handoff_happens_at_the_first_position_the_book_has_not_seen(agents):
    hybrid, _, _ = agents
    board, sources = chess.Board(), []
    for san in RUY_C.split():
        sources.append(hybrid.source(board))
        board.push_san(san)
    assert sources == ["book"] * 6 + ["repertoire"] + ["policy"] * 3


def test_without_a_repertoire_tree_the_network_takes_over_directly(positions):
    hybrid = HybridAgent(BookAgent.from_positions(positions), SpyAgent())
    assert hybrid.source(board_after("e4 e5 Nf3 Nc6 Bb5 Nf6")) == "policy"


def test_choose_is_seeded_and_argmax_at_temperature_zero(positions):
    def make():
        return HybridAgent(BookAgent.from_positions(positions), SpyAgent(), seed=2)

    off_book = board_after("e4 e5 Nf3 Nc6 Bb5 Nf6 O-O")
    one, two = make(), make()
    assert [one.choose(off_book) for _ in range(30)] == [two.choose(off_book) for _ in range(30)]
    first_legal = min(m.uci() for m in off_book.legal_moves)
    assert one.choose(off_book, temperature=0).uci() == first_legal
    assert one.choose(chess.Board(), temperature=0).uci() == "e2e4"


def test_eval_harness_scores_the_hybrid_like_the_book_when_seen_and_the_network_when_not(
    positions,
):
    book, net = BookAgent.from_positions(positions), SpyAgent()
    hybrid = HybridAgent(book, net)
    coverage = evaluate_coverage(hybrid, positions, {})["openings"]["ruy_lopez"]["positions"]
    assert coverage["all"]["pct"] == 100.0 and coverage["strict"]["pct"] == 100.0
    ours = agreement(hybrid, positions)["openings"]["ruy_lopez"]
    books = agreement(book, positions)["openings"]["ruy_lopez"]
    nets = agreement(net, positions)["openings"]["ruy_lopez"]
    assert ours["strict"] == nets["strict"]
    seen_hits = lambda block: block["all"]["top1"] * 10 - block["strict"]["top1"] * 4  # noqa: E731
    # percentages are rounded to 2 decimals, so allow for that
    assert seen_hits(ours) == pytest.approx(seen_hits(books), abs=0.1)
    assert agreement(hybrid, positions)["sanity"]["illegal_moves"] == 0
