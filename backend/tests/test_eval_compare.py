import chess
from helpers import RUY_A, RUY_C, game_rows, positions_frame
from test_eval_quality import FixedAgent

from chess_trainer.agents.book import BookAgent
from chess_trainer.eval.compare import comparison_markdown
from chess_trainer.eval.coverage import evaluate_coverage
from chess_trainer.eval.quality import agreement, engine_quality, sample_positions


def _metrics(agent, name, positions, sample, evals):
    return {
        "agent": name,
        "split": "test",
        "coverage": evaluate_coverage(agent, positions, {}),
        "quality": {
            "agreement": agreement(agent, positions),
            "engine": engine_quality(agent, sample, evals),
        },
    }


def test_comparison_table_puts_both_agents_side_by_side_on_strict_positions():
    positions = positions_frame(game_rows("tr", RUY_A, "train"), game_rows("te", RUY_C, "test"))
    sample = sample_positions(positions, n_per_stratum=50)
    evals = {}
    for row in sample.itertuples():
        board = chess.Board(f"{row.fen_norm} 0 1")
        evals[row.fen_norm] = {m.uci(): (0, None) for m in board.legal_moves}
    book = _metrics(BookAgent.from_positions(positions), "book", positions, sample, evals)

    class Uniform(FixedAgent):
        def policy(self, board):
            legal = [m.uci() for m in board.legal_moves]
            return {uci: 1 / len(legal) for uci in legal}

    ours = _metrics(Uniform({}), "resnet-main", positions, sample, evals)
    text = comparison_markdown(ours, book)
    assert "resnet-main vs book on strict positions" in text
    assert "Ruy Lopez: book | Ruy Lopez: resnet-main" in text
    assert "Ruy Lopez 4" in text  # four strict positions in the fixture
    coverage_row = next(
        line for line in text.splitlines() if line.startswith("| position coverage")
    )
    # the book falls back on every strict position; an agent without a fallback covers all
    assert coverage_row.split("|")[2:4] == [" 0.0% ", " 100.0% "]
    assert "n/a" in text  # the Taimanov columns are empty in this fixture
