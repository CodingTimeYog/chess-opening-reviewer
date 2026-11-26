import chess
from helpers import RUY_A, board_after, game_rows, positions_frame

from chess_trainer.agents.book import BookAgent
from chess_trainer.engine import MoveAnalysis
from chess_trainer.feedback import FeedbackBuilder
from chess_trainer.feedback.llm import OllamaRewriter, rewriter_from_env
from chess_trainer.rag import Retriever


class Analyser:
    def analyse(self, board, move):
        return MoveAnalysis("b5a4", (40, None), (-90, None))


def _builder(rewriter=None):
    positions = positions_frame(game_rows("g1", RUY_A, "train"))
    return FeedbackBuilder(
        BookAgent.from_positions(positions), {}, Retriever(positions), Analyser(), rewriter
    )


BOARD = board_after("e4 e5 Nf3 Nc6 Bb5 a6")
MOVE = chess.Move.from_uci("b5e2")


def test_llm_is_off_by_default(monkeypatch):
    monkeypatch.delenv("CHESS_TRAINER_LLM", raising=False)
    assert rewriter_from_env() is None
    feedback = _builder().move_feedback(BOARD, MOVE)
    assert feedback["llm_text"] is None and feedback["text"].startswith("Be2 is a mistake")


def test_flag_turns_it_on_with_the_configured_model(monkeypatch):
    monkeypatch.setenv("CHESS_TRAINER_LLM", "ollama")
    monkeypatch.setenv("CHESS_TRAINER_LLM_MODEL", "some-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://box:1234/")
    rewriter = rewriter_from_env()
    assert (rewriter.model, rewriter.url) == ("some-model", "http://box:1234/api/generate")


def test_rewriter_sends_only_facts_and_template_and_returns_the_prose():
    sent = {}

    def post(url, payload, timeout):
        sent.update(payload, url=url)
        return {"response": "  Be2 is passive; Ba4 keeps the pressure.  "}

    feedback = _builder(OllamaRewriter(post=post)).move_feedback(BOARD, MOVE)
    assert feedback["llm_text"] == "Be2 is passive; Ba4 keeps the pressure."
    assert feedback["text"].startswith("Be2 is a mistake")  # the template is always kept
    assert sent["model"] == "qwen2.5:3b" and sent["stream"] is False
    assert sent["url"] == "http://localhost:11434/api/generate"
    assert '"classification": "mistake"' in sent["prompt"]
    assert "White g1" in sent["prompt"] and feedback["text"] in sent["prompt"]
    assert "game_id" not in sent["prompt"] and "fen" not in sent["prompt"]
    assert "Use only the information" in sent["system"]


def test_an_unreachable_or_empty_llm_falls_back_to_the_template():
    def down(url, payload, timeout):
        raise OSError("connection refused")

    assert _builder(OllamaRewriter(post=down)).move_feedback(BOARD, MOVE)["llm_text"] is None
    empty = OllamaRewriter(post=lambda *_: {"response": "   "})
    assert _builder(empty).move_feedback(BOARD, MOVE)["llm_text"] is None


def test_moves_that_were_not_analysed_are_not_sent_to_the_llm():
    calls = []
    rewriter = OllamaRewriter(post=lambda *a: calls.append(a) or {"response": "x"})
    feedback = _builder(rewriter).move_feedback(BOARD, MOVE, analyse=False)
    assert feedback["llm_text"] is None and calls == []
