import chess
import chess.pgn
import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import RUY_A, RUY_C, board_after, game_rows, positions_frame

from chess_trainer.agents.base import Agent
from chess_trainer.api.app import create_app
from chess_trainer.api.service import TrainerService
from chess_trainer.data.openings import ROOT_FENS, fen_norm
from chess_trainer.engine import MoveAnalysis
from chess_trainer.rag import Retriever

TAIMANOV = "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6"
LINES = {
    "ruy1": (RUY_A, "ruy_lopez", 2750),
    "ruy2": (RUY_A, "ruy_lopez", 2650),
    "berlin": (RUY_C, "ruy_lopez", 2600),
    "tai1": (TAIMANOV, "taimanov", 2600),
    "heldout": (RUY_C, "ruy_lopez", 2800),
}
AFTER_A6 = fen_norm(board_after("e4 e5 Nf3 Nc6 Bb5 a6"))


class StubNetwork(Agent):
    """Prefers the alphabetically first legal move."""

    name = "resnet_main"

    def policy(self, board):
        legal = sorted(m.uci() for m in board.legal_moves)
        rest = 0.1 / max(len(legal) - 1, 1)
        return {uci: (0.9 if i == 0 else rest) for i, uci in enumerate(legal)}


class StubAnalyser:
    LOSSES = {"b5a6": 420, "b5e2": 130}

    def best(self, board):
        return (
            "b5a4" if fen_norm(board) == AFTER_A6 else min(m.uci() for m in board.legal_moves)
        ), (
            40,
            None,
        )

    def analyse(self, board, move):
        best, score = self.best(board)
        loss = 0 if move.uci() == best else self.LOSSES.get(move.uci(), 10)
        return MoveAnalysis(best, score, (40 - loss, None))


class StubRewriter:
    def rewrite(self, facts, template_text):
        return f"LLM: {facts['played']['san']}"


def _pgn(san: str, game_id: str) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = f"Event {game_id}"
    node = game
    board = chess.Board()
    for tok in san.split():
        node = node.add_variation(board.push_san(tok))
    return str(game)


def make_service(**kwargs) -> TrainerService:
    positions = positions_frame(
        *[
            game_rows(gid, san, "test" if gid == "heldout" else "train", opening=opening, elo=elo)
            for gid, (san, opening, elo) in LINES.items()
        ]
    )
    games = positions.drop_duplicates("game_id")[
        [
            "game_id",
            "opening",
            "result",
            "white_elo",
            "black_elo",
            "eco",
            "event",
            "year",
            "white",
            "black",
        ]
    ].copy()
    games["pgn"] = [_pgn(LINES[gid][0], gid) for gid in games["game_id"]]
    tree = {
        AFTER_A6: {
            "b5a4": {"in_repertoire": True, "nags": [], "comment": "Keep the bishop."},
        }
    }
    retriever = Retriever(positions, splits=None, embed=lambda fen, ply: np.eye(16)[min(ply, 15)])
    indexed = retriever.indexed_positions()
    retriever.set_embeddings(list(indexed["fen_norm"]), np.eye(16)[indexed["ply"].to_numpy()])
    defaults = dict(analyser=StubAnalyser(), network=StubNetwork(), retriever=retriever)
    return TrainerService(positions, games, tree, **{**defaults, **kwargs})


@pytest.fixture
def client():
    return TestClient(create_app(make_service()))


def new_game(client, **body):
    payload = {"opening_id": "ruy_lopez", "agent": "hybrid", "temperature": 0, **body}
    response = client.post("/games", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_health_openings_and_agents(client):
    assert client.get("/health").json() == {"status": "ok"}
    openings = client.get("/openings").json()
    assert [(o["id"], o["user_color"]) for o in openings] == [
        ("ruy_lopez", "white"),
        ("taimanov", "black"),
    ]
    assert openings[0]["root_moves"] == ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    assert openings[0]["root_fen"] == ROOT_FENS["ruy_lopez"]
    agents = client.get("/agents").json()
    assert agents == {
        "agents": ["book", "hybrid", "resnet_main"],
        "default": "hybrid",
        "llm": False,
    }


def test_new_game_starts_at_the_opening_root_and_the_agent_replies_first(client):
    state = new_game(client)
    assert state["opening_id"] == "ruy_lopez" and state["mode"] == "play"
    assert state["user_color"] == "white" and state["agents"] == {"white": None, "black": "hybrid"}
    assert [m["san"] for m in state["moves"]] == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert [m["by"] for m in state["moves"]] == ["setup"] * 5 + ["agent"]
    reply = state["moves"][-1]
    assert (reply["agent"], reply["tier"], reply["side"], reply["ply"]) == (
        "hybrid",
        "book",
        "black",
        5,
    )
    assert state["status"] == "active" and state["turn"] == "white" and state["ply"] == 6
    assert "b5a4" in state["legal_moves"]
    assert state["pgn"] == "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    assert client.get(f"/games/{state['game_id']}").json() == state


def test_taimanov_as_black_and_starting_from_move_one(client):
    state = new_game(client, opening_id="taimanov")
    assert state["user_color"] == "black" and state["agents"] == {"white": "hybrid", "black": None}
    assert len(state["moves"]) == 9 and state["turn"] == "black"  # 8 setup moves + White's reply
    assert state["moves"][-1]["san"] == "Nc3"

    scratch = new_game(client, start="initial")
    assert scratch["moves"] == [] and scratch["turn"] == "white" and scratch["ply"] == 0
    flipped = new_game(client, start="initial", user_color="black")
    assert [m["by"] for m in flipped["moves"]] == ["agent"] and flipped["turn"] == "black"


def test_a_good_move_returns_feedback_citations_similar_games_and_the_reply(client):
    game = new_game(client)
    response = client.post(f"/games/{game['game_id']}/moves", json={"uci": "b5a4"})
    assert response.status_code == 200, response.text
    body = response.json()
    move = body["move"]
    assert (move["san"], move["by"], move["ply"], move["side"]) == ("Ba4", "user", 6, "white")
    feedback = move["feedback"]
    assert feedback["facts"]["classification"] == "good" and feedback["facts"]["cp_loss"] == 0
    assert feedback["text"].startswith("Ba4 is the engine's first choice")
    assert "Keep the bishop." in feedback["text"]  # the repertoire comment
    assert feedback["llm_text"] is None
    # exact-match citations: the games that reached this position, best rated first
    assert [c["game_id"] for c in feedback["citations"]] == ["ruy1", "ruy2"]
    assert all(not hit["exact"] for hit in move["similar_games"])
    assert {hit["game_id"] for hit in move["similar_games"]} == {"heldout", "berlin"}
    assert body["agent_reply"]["san"] == "Nf6" and body["agent_reply"]["tier"] == "book"
    assert body["events"] == []
    assert body["state"]["ply"] == 8 and body["state"]["turn"] == "white"
    assert body["state"]["moves"][-2:] == [move, body["agent_reply"]]


def test_a_blunder_triggers_events_and_the_network_takes_over_out_of_book(client):
    game = new_game(client)
    body = client.post(f"/games/{game['game_id']}/moves", json={"uci": "b5a6"}).json()
    assert body["move"]["feedback"]["facts"]["classification"] == "blunder"
    kinds = {(e["type"], e["severity"], e["ply"]) for e in body["events"]}
    assert ("eval_swing", "blunder", 6) in kinds
    assert ("left_book", "info", 6) in kinds and ("left_repertoire", "info", 6) in kinds
    event = next(e for e in body["events"] if e["type"] == "eval_swing")
    assert set(event) == {"type", "ply", "severity", "side", "feedback"}
    assert event["feedback"]["text"].startswith("Bxa6 is a blunder")
    assert body["agent_reply"]["tier"] == "policy" and body["agent_reply"]["uci"] == "a8a6"
    assert body["state"]["events"] == body["events"]


def test_illegal_and_malformed_moves_are_rejected_with_the_legal_moves(client):
    game = new_game(client)
    url = f"/games/{game['game_id']}/moves"
    illegal = client.post(url, json={"uci": "b5b6"})
    assert illegal.status_code == 400
    detail = illegal.json()["detail"]
    assert detail["error"] == "illegal_move" and "b5a4" in detail["legal_moves"]
    assert client.post(url, json={"uci": "zz"}).status_code == 400
    assert client.post(url, json={}).status_code == 422
    assert client.get(f"/games/{game['game_id']}").json()["ply"] == 6  # nothing was played


def test_unknown_ids_and_bad_options(client):
    assert client.get("/games/nope").status_code == 404
    assert client.post("/games/nope/moves", json={"uci": "e2e4"}).status_code == 404
    assert client.post("/games", json={"opening_id": "italian"}).status_code == 400
    assert (
        client.post("/games", json={"opening_id": "ruy_lopez", "agent": "gpt"}).status_code == 400
    )
    assert client.post("/games", json={"opening_id": "ruy_lopez", "mode": "x"}).status_code == 422


def test_each_agent_can_be_the_opponent(client):
    for agent in ("book", "hybrid", "resnet_main"):
        state = new_game(client, agent=agent)
        assert state["agents"]["black"] == agent
        assert state["moves"][-1]["agent"] == agent
    assert new_game(client, agent="resnet_main")["moves"][-1]["uci"] == "a7a5"


def test_agent_vs_agent_advances_on_request_and_takes_no_user_moves(client):
    state = new_game(client, mode="watch", white_agent="hybrid", black_agent="book")
    assert state["user_color"] is None
    assert state["agents"] == {"white": "hybrid", "black": "book"}
    assert len(state["moves"]) == 5  # setup only: nothing moves until asked
    url = f"/games/{state['game_id']}"
    step = client.post(f"{url}/agent-move", params={"plies": 3}).json()
    assert [m["san"] for m in step["moves"]] == ["a6", "Ba4", "Nf6"]
    assert [m["agent"] for m in step["moves"]] == ["book", "hybrid", "book"]
    assert step["state"]["ply"] == 8
    assert client.post(f"{url}/moves", json={"uci": "e1g1"}).status_code == 409
    # in play mode the agent cannot be asked to move for the user
    play = new_game(client)
    assert client.post(f"/games/{play['game_id']}/agent-move").status_code == 409


def test_reset_restarts_the_same_session(client):
    game = new_game(client)
    url = f"/games/{game['game_id']}"
    client.post(f"{url}/moves", json={"uci": "b5a6"})
    fresh = client.post(f"{url}/reset").json()
    assert fresh["game_id"] == game["game_id"]
    assert fresh["ply"] == 6 and fresh["events"] == [] and len(fresh["moves"]) == 6
    assert fresh["agents"] == game["agents"]


def test_opening_end_closes_the_game_and_review_summarises_it():
    client = TestClient(create_app(make_service(max_ply=10)))
    game = new_game(client)
    url = f"/games/{game['game_id']}"
    client.post(f"{url}/moves", json={"uci": "b5e2"})  # a mistake
    last = client.post(f"{url}/moves", json={"uci": "e1g1"}).json()
    assert ("opening_end", 9) in {(e["type"], e["ply"]) for e in last["events"]}
    assert last["state"]["status"] == "opening_over" and last["state"]["legal_moves"] == []
    assert client.post(f"{url}/moves", json={"uci": "d2d3"}).status_code == 409

    review = client.get(f"{url}/review").json()
    assert review["game_id"] == game["game_id"]
    assert [(m["san"], m["classification"], m["cp_loss"]) for m in review["moves"]] == [
        ("Be2", "mistake", 130),
        ("O-O", "good", 10),
    ]
    assert review["moves"][0]["best_san"] == "Ba4" and review["moves"][0]["text"]
    assert review["summary"] == {
        "analysed": 2,
        "mean_cp_loss": 70.0,
        "counts": {"good": 1, "inaccuracy": 0, "mistake": 1, "blunder": 0},
    }
    assert {e["type"] for e in review["events"]} >= {"eval_swing", "opening_end"}


def test_retrieve_returns_exact_and_similar_master_games(client):
    body = client.get(
        "/positions/retrieve", params={"fen": AFTER_A6, "k": 2, "opening": "ruy_lopez"}
    ).json()
    assert body["fen"] == AFTER_A6
    assert [(h["game_id"], h["exact"], h["move_uci"]) for h in body["exact"]] == [
        ("ruy1", True, "b5a4"),
        ("ruy2", True, "b5a4"),
    ]
    assert len(body["similar"]) == 2 and not any(h["exact"] for h in body["similar"])
    assert set(body["exact"][0]) >= {
        "white",
        "black",
        "white_elo",
        "year",
        "event",
        "result",
        "eco",
    }
    full_fen = board_after("e4 e5 Nf3 Nc6 Bb5 a6").fen()
    assert client.get("/positions/retrieve", params={"fen": full_fen}).json()["fen"] == AFTER_A6
    assert client.get("/positions/retrieve", params={"fen": "not a fen"}).status_code == 400


def test_analyze_gives_engine_book_and_policy_views(client):
    body = client.get("/positions/analyze", params={"fen": AFTER_A6}).json()
    assert body["best"] == {"uci": "b5a4", "san": "Ba4"} and body["eval_cp"] == 40
    assert body["book"]["in_book"] is True and body["book"]["games"] == 2
    assert body["book"]["moves"] == [{"uci": "b5a4", "san": "Ba4", "prob": 1.0}]
    assert body["policy"]["agent"] == "resnet_main"
    assert len(body["policy"]["moves"]) == 5 and body["policy"]["moves"][0]["prob"] == 0.9
    assert body["repertoire"] == [{"uci": "b5a4", "san": "Ba4", "comment": "Keep the bishop."}]


def test_master_game_returns_metadata_and_the_full_pgn(client):
    body = client.get("/master-games/ruy1").json()
    assert body["game_id"] == "ruy1" and body["white"] == "White ruy1" and body["white_elo"] == 2750
    assert body["opening"] == "ruy_lopez" and body["year"] == 2015
    assert "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6" in body["pgn"] and body["pgn"].startswith("[Event")
    assert client.get("/master-games/missing").status_code == 404


def test_cors_allows_the_local_react_dev_server(client):
    headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    preflight = client.options("/games", headers=headers)
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.options("/games", headers={**headers, "Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_llm_text_appears_only_when_a_rewriter_is_configured():
    client = TestClient(create_app(make_service(rewriter=StubRewriter())))
    assert client.get("/agents").json()["llm"] is True
    game = new_game(client)
    body = client.post(f"/games/{game['game_id']}/moves", json={"uci": "b5a4"}).json()
    assert body["move"]["feedback"]["llm_text"] == "LLM: Ba4"
    assert body["move"]["feedback"]["text"].startswith("Ba4 is the engine's first choice")
    assert body["agent_reply"]["feedback"]["llm_text"] is None  # agent moves are not rewritten


def test_without_a_network_only_the_book_is_offered():
    client = TestClient(create_app(make_service(network=None)))
    assert client.get("/agents").json() == {"agents": ["book"], "default": "book", "llm": False}
    assert (
        client.post("/games", json={"opening_id": "ruy_lopez", "agent": "hybrid"}).status_code
        == 400
    )
    assert client.post("/games", json={"opening_id": "ruy_lopez"}).status_code == 201
    assert client.get("/positions/analyze", params={"fen": AFTER_A6}).json()["policy"] is None


def test_openapi_schema_documents_every_route(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/health", "/openings", "/agents", "/games", "/games/{game_id}",
        "/games/{game_id}/moves", "/games/{game_id}/agent-move", "/games/{game_id}/reset",
        "/games/{game_id}/review", "/positions/retrieve", "/positions/analyze",
        "/master-games/{game_id}",
    }  # fmt: skip
    assert {"MoveResponse", "GameState", "MoveRecord", "Event"} <= set(
        schema["components"]["schemas"]
    )
