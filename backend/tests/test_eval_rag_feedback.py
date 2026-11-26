import chess
import pytest
from helpers import RUY_A, RUY_C, game_rows, positions_frame

from chess_trainer.agents.book import BookAgent
from chess_trainer.agents.hybrid import HybridAgent
from chess_trainer.engine import MoveAnalysis, clipped_cp
from chess_trainer.eval.coverage import evaluate_coverage, metrics_markdown
from chess_trainer.eval.feedback import build_cases, evaluate_feedback, feedback_markdown
from chess_trainer.eval.quality import sample_positions
from chess_trainer.eval.retrieval import evaluate_retrieval, is_relevant, retrieval_markdown
from chess_trainer.feedback import FeedbackBuilder
from chess_trainer.rag import Hit, RandomRetriever, Retriever

TEST_LINE = "e4 e5 Nf3 Nc6 Bb5 Nf6 d3 Bc5"  # leaves the train games after 4.d3


@pytest.fixture
def positions():
    return positions_frame(
        game_rows("a1", RUY_A, "train", elo=2700),
        game_rows("a2", RUY_A, "train", elo=2650),
        game_rows("c1", RUY_C, "train", elo=2600, eco="C67"),
        game_rows("t", TEST_LINE, "test", eco="C65"),
    )


def _hit(**changes) -> Hit:
    base = dict(
        game_id="g", score=0.9, exact=False, fen_norm="f", ply=6, move_uci="e1g1",
        opening="ruy_lopez", eco="C67", white="W", black="B", white_elo=2600, black_elo=2600,
        year=2015, event="E", result="1-0",
    )  # fmt: skip
    return Hit(**{**base, **changes})


def test_relevance_rule_exact_position_or_same_eco_and_same_next_move():
    assert is_relevant(_hit(exact=True, eco="A00", move_uci="a2a3"), "C67", "e1g1")
    assert is_relevant(_hit(), "C67", "e1g1")
    assert not is_relevant(_hit(move_uci="d2d3"), "C67", "e1g1")
    assert not is_relevant(_hit(eco="C65"), "C67", "e1g1")


def test_precision_and_mrr_arithmetic(positions):
    class Stub:
        def retrieve(self, fen, k, ply, opening):
            return [_hit(eco="X"), _hit(eco="C65", move_uci=MOVE), _hit(eco="X")]

    sample = sample_positions(positions, n_per_stratum=50)
    strict = sample[sample["stratum"] == "strict"]
    assert len(strict) == 1  # the position after 4.d3
    global MOVE
    MOVE = strict.iloc[0]["move_uci"]
    block = evaluate_retrieval(Stub(), strict)["openings"]["ruy_lopez"]["strict"]
    assert block == {"n": 1, "precision": 20.0, "mrr": 0.5, "mean_returned": 3.0}


def test_exact_lookup_scores_on_seen_positions_and_nothing_on_strict_ones(positions):
    sample = sample_positions(positions, n_per_stratum=50)
    exact = evaluate_retrieval(Retriever(positions), sample)["openings"]["ruy_lopez"]
    assert exact["seen"]["n"] == 7 and exact["seen"]["mrr"] == 1.0
    # plies 0-5 are reached by all three train games, ply 6 only by c1: (6 * 3 + 1) / (7 * 5)
    assert exact["seen"]["precision"] == pytest.approx(100 * 19 / 35, abs=0.01)
    assert exact["strict"] == {"n": 1, "precision": 0.0, "mrr": 0.0, "mean_returned": 0.0}
    control = evaluate_retrieval(RandomRetriever(positions), sample)
    text = retrieval_markdown({"random": control, "exact": {"k": 5, "openings": {
        "ruy_lopez": exact, "taimanov": control["openings"]["taimanov"]}}},
        {"split": "test", "seed": 0, "git_sha": "abc"})  # fmt: skip
    assert "precision@5, exact" in text and "MRR, random" in text and "n/a" in text


class CacheAnalyser:
    """A perfect live engine: answers from the same scores the ground truth uses."""

    def __init__(self, evals):
        self.evals = evals

    def analyse(self, board, move):
        scores = self.evals[board.epd()]
        best = max(sorted(scores), key=lambda m: clipped_cp(*scores[m]))
        return MoveAnalysis(best, scores[best], scores[move.uci()])


def _evals(sample):
    """Every move scores 0 except the alphabetically first (+40) and last (-300)."""
    evals = {}
    for row in sample.itertuples():
        moves = sorted(m.uci() for m in chess.Board(f"{row.fen_norm} 0 1").legal_moves)
        evals[row.fen_norm] = {m: (0, None) for m in moves}
        evals[row.fen_norm][moves[0]] = (40, None)
        evals[row.fen_norm][moves[-1]] = (-300, None)
    return evals


def test_cases_cover_master_best_and_random_moves_with_ground_truth(positions):
    sample = sample_positions(positions, n_per_stratum=50)
    evals = _evals(sample)
    cases = build_cases(sample, evals, seed=3)
    assert cases.equals(build_cases(sample, evals, seed=3))
    assert not cases.duplicated(["fen_norm", "move_uci"]).any()
    assert set(cases["kind"]) <= {"master", "best", "random"}
    assert len(cases) > len(sample)
    best = cases[cases["kind"] == "best"]
    assert (best["truth_cp_loss"] == 0).all() and (best["truth_class"] == "good").all()
    start = cases[(cases["ply"] == 0) & (cases["kind"] == "master")].iloc[0]
    # 1.e4 scores 0 against a best of +40: 40 cp lost, still "good"
    assert (start["move_uci"], start["truth_cp_loss"], start["truth_class"]) == ("e2e4", 40, "good")
    assert start["truth_best"] == ["a2a3"]


def test_feedback_from_a_perfect_engine_scores_100_with_no_hallucinated_citations(positions):
    sample = sample_positions(positions, n_per_stratum=50)
    evals = _evals(sample)
    cases = build_cases(sample, evals)
    builder = FeedbackBuilder(
        BookAgent.from_positions(positions), {}, Retriever(positions, splits=None),
        CacheAnalyser(evals),
    )  # fmt: skip
    metrics = evaluate_feedback(cases, builder, positions, evals)
    overall = metrics["overall"]
    assert overall["n"] == len(cases)
    assert overall["classification_accuracy"] == 100.0
    assert overall["best_move_correct"] == 100.0 and overall["text_consistent"] == 100.0
    assert overall["citations"] > 0 and overall["hallucinated_citations"] == 0
    assert metrics["by_stratum"]["ruy_lopez"]["strict"]["n"] >= 1
    text = feedback_markdown(metrics, {"layer": "template", "split": "test", "seed": 0,
                                       "depth": 14, "git_sha": "abc"})  # fmt: skip
    assert "classification accuracy" in text and "Confusion matrix" in text


def test_a_wrong_engine_and_an_invented_citation_are_caught(positions):
    sample = sample_positions(positions, n_per_stratum=50)
    evals = _evals(sample)
    cases = build_cases(sample, evals)

    class AlwaysHappy:
        def analyse(self, board, move):
            return MoveAnalysis(move.uci(), (0, None), (0, None))

    class Inventing(FeedbackBuilder):
        def move_feedback(self, board, move, analyse=True):
            feedback = super().move_feedback(board, move, analyse)
            feedback["citations"] = [*feedback["citations"], {"game_id": "made-up"}]
            return feedback

    builder = Inventing(
        BookAgent.from_positions(positions), {}, Retriever(positions, splits=None), AlwaysHappy()
    )
    metrics = evaluate_feedback(cases, builder, positions, evals)
    assert metrics["overall"]["classification_accuracy"] < 100.0
    assert metrics["overall"]["best_move_correct"] < 100.0
    assert metrics["overall"]["hallucinated_citations"] == len(cases)
    assert metrics["confusion"]["blunder"] == {
        "good": int((cases["truth_class"] == "blunder").sum())
    }


def test_hybrid_tier_breakdown_is_reported_for_all_and_strict_positions(positions):
    class Net(BookAgent):
        name = "net"

        def is_fallback(self, board):
            return False

    book = BookAgent.from_positions(positions)
    after_d3 = positions[(positions["game_id"] == "t") & (positions["ply"] == 7)].iloc[0]
    tree = {after_d3["fen_norm"]: {"f8c5": {"in_repertoire": True}}}
    hybrid = HybridAgent(book, Net(book.book), tree)
    coverage = evaluate_coverage(hybrid, positions, tree)
    tiers = coverage["openings"]["ruy_lopez"]["tiers"]
    assert tiers["all"] == {"n": 8, "book": 87.5, "prep": 12.5, "resnet": 0.0}
    assert tiers["strict"] == {"n": 1, "book": 0.0, "prep": 100.0, "resnet": 0.0}
    without_prep = evaluate_coverage(HybridAgent(book, Net(book.book)), positions, {})
    assert without_prep["openings"]["ruy_lopez"]["tiers"]["strict"]["resnet"] == 100.0
    text = metrics_markdown(coverage)
    assert "% book, all positions" in text and "% resnet, strict positions" in text
    # agents without tiers have no such section
    plain = evaluate_coverage(book, positions, tree)
    assert "tiers" not in plain["openings"]["ruy_lopez"]
    assert "Which tier" not in metrics_markdown(plain)
