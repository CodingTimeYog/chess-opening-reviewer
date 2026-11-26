"""The trainer behind the HTTP routes: sessions, moves, feedback, retrieval.

Sessions live in memory (no persistence in v1). One lock serialises every call, because
the engine process, the SQLite cache and the network are shared.
"""

import threading
import uuid
from dataclasses import dataclass, field

import chess
import pandas as pd

from chess_trainer import config
from chess_trainer.agents.base import Agent
from chess_trainer.agents.book import BookAgent
from chess_trainer.agents.hybrid import HybridAgent
from chess_trainer.data.openings import OPENINGS, ROOT_FENS, fen_norm
from chess_trainer.data.repertoire import repertoire_moves
from chess_trainer.engine import clipped_cp
from chess_trainer.feedback import FeedbackBuilder, TriggerEngine
from chess_trainer.rag import Retriever

SIMILAR_GAMES = 3
TOP_MOVES = 5
CLASSES = ("good", "inaccuracy", "mistake", "blunder")


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str, **extra) -> None:
        super().__init__(message)
        self.status = status
        self.detail = {"error": error, "message": message, **extra}


def _side(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


@dataclass
class Game:
    id: str
    opening_id: str
    mode: str
    start: str
    user_color: chess.Color | None
    agents: dict[str, str | None]  # side -> agent name
    temperature: float
    analyse_agent_moves: bool
    board: chess.Board = field(default_factory=chess.Board)
    moves: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    status: str = "active"
    triggers: TriggerEngine | None = None


class TrainerService:
    def __init__(
        self,
        positions: pd.DataFrame,
        games: pd.DataFrame,
        tree: dict,
        analyser,
        network: Agent | None = None,
        network_name: str = "resnet_main",
        retriever: Retriever | None = None,
        rewriter=None,
        max_ply: int = config.MAX_PLY,
        seed: int = config.SEED,
    ) -> None:
        self.lock = threading.RLock()
        self.tree = tree
        self.max_ply = max_ply
        self.master_games = games.set_index("game_id")
        self.book = BookAgent.from_positions(positions, seed=seed)  # train games, as evaluated
        self.agents: dict[str, Agent] = {"book": self.book}
        self.network_name = network_name if network is not None else None
        if network is not None:
            self.agents[network_name] = network
            self.agents["hybrid"] = HybridAgent(self.book, network, tree, seed=seed)
        self.default_agent = "hybrid" if network is not None else "book"
        self.retriever = retriever or Retriever(positions, splits=None)  # every game
        self.analyser = analyser
        self.llm = rewriter is not None
        self.builder = FeedbackBuilder(self.book, tree, self.retriever, analyser, rewriter)
        self._games: dict[str, Game] = {}

    # ------------------------------------------------------------ catalogue

    def openings(self) -> list[dict]:
        return [
            {
                "id": o.id,
                "name": o.name,
                "user_color": _side(o.side),
                "root_moves": list(o.root_san),
                "root_fen": ROOT_FENS[o.id],
            }
            for o in OPENINGS.values()
        ]

    def agent_names(self) -> dict:
        return {"agents": sorted(self.agents), "default": self.default_agent, "llm": self.llm}

    def _agent(self, name: str | None) -> str:
        name = name or self.default_agent
        if name not in self.agents:
            raise ApiError(
                400, "unknown_agent", f"agent must be one of {sorted(self.agents)}", agent=name
            )
        return name

    # ------------------------------------------------------------ sessions

    def create_game(self, request: dict, game_id: str | None = None) -> dict:
        with self.lock:
            opening = OPENINGS.get(request["opening_id"])
            if opening is None:
                raise ApiError(
                    400, "unknown_opening", f"opening_id must be one of {sorted(OPENINGS)}"
                )
            mode = request.get("mode", "play")
            if mode == "play":
                colour = request.get("user_color")
                user_color = opening.side if colour is None else colour == "white"
                agents = {_side(user_color): None}
                agents[_side(not user_color)] = self._agent(request.get("agent"))
            else:
                user_color = None
                agents = {
                    "white": self._agent(request.get("white_agent") or request.get("agent")),
                    "black": self._agent(request.get("black_agent") or request.get("agent")),
                }
            game = Game(
                id=game_id or uuid.uuid4().hex[:12],
                opening_id=opening.id,
                mode=mode,
                start=request.get("start", "root"),
                user_color=user_color,
                agents={side: agents[side] for side in ("white", "black")},
                temperature=request.get("temperature", 1.0),
                analyse_agent_moves=request.get("analyse_agent_moves", False),
            )
            game.triggers = TriggerEngine(
                agent_color=None if user_color is None else not user_color, max_ply=self.max_ply
            )
            game.request = dict(request)  # kept for reset
            if game.start == "root":
                for san in opening.root_san:
                    move = game.board.parse_san(san)
                    self._record(game, move, by="setup")
            self._games[game.id] = game
            if mode == "play":
                self._agent_turn(game)
            return self.state(game.id)

    def reset(self, game_id: str) -> dict:
        with self.lock:
            return self.create_game(self._game(game_id).request, game_id=game_id)

    def _game(self, game_id: str) -> Game:
        game = self._games.get(game_id)
        if game is None:
            raise ApiError(404, "unknown_game", f"no game with id {game_id}")
        return game

    def state(self, game_id: str) -> dict:
        with self.lock:
            game = self._game(game_id)
            board = game.board
            active = game.status == "active"
            return {
                "game_id": game.id,
                "opening_id": game.opening_id,
                "mode": game.mode,
                "start": game.start,
                "user_color": None if game.user_color is None else _side(game.user_color),
                "agents": dict(game.agents),
                "status": game.status,
                "fen": board.fen(),
                "ply": board.ply(),
                "turn": _side(board.turn),
                "legal_moves": sorted(m.uci() for m in board.legal_moves) if active else [],
                "moves": list(game.moves),
                "events": list(game.events),
                "pgn": chess.Board().variation_san(board.move_stack),
            }

    # ------------------------------------------------------------ moves

    def _record(
        self,
        game: Game,
        move: chess.Move,
        by: str,
        agent: str | None = None,
        tier: str | None = None,
        analyse: bool = False,
    ) -> tuple[dict, list[dict]]:
        board = game.board
        record = {
            "ply": board.ply(),
            "side": _side(board.turn),
            "uci": move.uci(),
            "san": board.san(move),
            "by": by,
            "agent": agent,
            "tier": tier,
            "feedback": None,
            "similar_games": [],
        }
        events: list[dict] = []
        if by != "setup":
            feedback = self.builder.move_feedback(board, move, analyse=analyse)
            record["feedback"] = feedback
            cited = {c["game_id"] for c in feedback["citations"]}
            record["similar_games"] = [
                hit.to_dict()
                for hit in self.retriever.similar(
                    fen_norm(board), SIMILAR_GAMES, board.ply(), game.opening_id, exclude=cited
                )
            ]
            events = [event.to_dict() for event in game.triggers.on_move(feedback["facts"])]
        board.push(move)
        record["fen"] = board.fen()
        game.moves.append(record)
        game.events += events
        if any(event["type"] == "opening_end" for event in events):
            game.status = "opening_over"
        elif board.is_game_over():
            game.status = "game_over"
        return record, events

    def _agent_turn(self, game: Game) -> tuple[dict, list[dict]] | None:
        """Play the agent's move if the game is active and it is an agent's turn."""
        name = game.agents[_side(game.board.turn)]
        if game.status != "active" or name is None:
            return None
        agent = self.agents[name]
        tier = agent.source(game.board) if hasattr(agent, "source") else None
        move = agent.choose(game.board, temperature=game.temperature)
        return self._record(
            game, move, by="agent", agent=name, tier=tier, analyse=game.analyse_agent_moves
        )

    def _require_active(self, game: Game) -> None:
        if game.status != "active":
            raise ApiError(
                409, "game_not_active", f"the game is {game.status}; reset to play again"
            )

    def user_move(self, game_id: str, uci: str) -> dict:
        with self.lock:
            game = self._game(game_id)
            self._require_active(game)
            if game.mode != "play" or game.board.turn != game.user_color:
                raise ApiError(409, "not_your_turn", "it is not the user's turn in this game")
            legal = sorted(m.uci() for m in game.board.legal_moves)
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                raise ApiError(
                    400, "invalid_uci", f"{uci!r} is not a UCI move", legal_moves=legal
                ) from None
            if move not in game.board.legal_moves:
                raise ApiError(400, "illegal_move", f"{uci} is not legal here", legal_moves=legal)
            record, events = self._record(game, move, by="user", analyse=True)
            reply = self._agent_turn(game)
            return {
                "move": record,
                "agent_reply": reply[0] if reply else None,
                "events": events + (reply[1] if reply else []),
                "state": self.state(game_id),
            }

    def agent_move(self, game_id: str, plies: int = 1) -> dict:
        with self.lock:
            game = self._game(game_id)
            self._require_active(game)
            if game.mode != "watch":
                raise ApiError(409, "not_watch_mode", "agents move on their own in play mode")
            records, events = [], []
            for _ in range(plies):
                played = self._agent_turn(game)
                if played is None:
                    break
                records.append(played[0])
                events += played[1]
            return {"moves": records, "events": events, "state": self.state(game_id)}

    def review(self, game_id: str) -> dict:
        with self.lock:
            game = self._game(game_id)
            rows = []
            for record in game.moves:
                facts = (record["feedback"] or {}).get("facts")
                if not facts or facts["classification"] is None:
                    continue
                rows.append(
                    {
                        "ply": record["ply"],
                        "side": record["side"],
                        "uci": record["uci"],
                        "san": record["san"],
                        "by": record["by"],
                        "classification": facts["classification"],
                        "cp_loss": facts["cp_loss"],
                        "best_san": facts["best"]["san"],
                        "text": record["feedback"]["text"],
                    }
                )
            losses = [row["cp_loss"] for row in rows]
            return {
                "game_id": game.id,
                "summary": {
                    "analysed": len(rows),
                    "mean_cp_loss": round(sum(losses) / len(losses), 1) if losses else None,
                    "counts": {
                        label: sum(row["classification"] == label for row in rows)
                        for label in CLASSES
                    },
                },
                "moves": rows,
                "events": list(game.events),
            }

    # ------------------------------------------------------------ positions

    @staticmethod
    def _board(fen: str) -> chess.Board:
        try:
            board = chess.Board(fen)
        except ValueError:
            raise ApiError(400, "invalid_fen", f"{fen!r} is not a FEN") from None
        if not board.is_valid():
            raise ApiError(400, "invalid_fen", f"{fen!r} is not a legal position")
        return board

    def retrieve(self, fen: str, k: int = 5, opening: str | None = None) -> dict:
        with self.lock:
            board = self._board(fen)
            key = fen_norm(board)
            exact = self.retriever.exact(key, k)
            taken = {hit.game_id for hit in exact}
            similar = self.retriever.similar(key, k, board.ply(), opening, exclude=taken)
            return {
                "fen": key,
                "exact": [hit.to_dict() for hit in exact],
                "similar": [hit.to_dict() for hit in similar],
            }

    def _top(self, board: chess.Board, policy: dict[str, float]) -> list[dict]:
        top = sorted(sorted(policy), key=policy.__getitem__, reverse=True)[:TOP_MOVES]
        return [
            {"uci": m, "san": board.san(chess.Move.from_uci(m)), "prob": round(policy[m], 4)}
            for m in top
        ]

    def analyze(self, fen: str) -> dict:
        with self.lock:
            board = self._board(fen)
            key = fen_norm(board)
            best, score = self.analyser.best(board)
            counts = self.book.book.get(key, {})
            entries = self.tree.get(key, {})
            network = self.agents.get(self.network_name) if self.network_name else None
            return {
                "fen": key,
                "best": {"uci": best, "san": board.san(chess.Move.from_uci(best))},
                "eval_cp": clipped_cp(*score),
                "book": {
                    "in_book": bool(counts),
                    "games": round(sum(counts.values())),
                    "moves": self._top(board, self.book.policy(board)) if counts else [],
                },
                "policy": None
                if network is None
                else {"agent": self.network_name, "moves": self._top(board, network.policy(board))},
                "repertoire": [
                    {
                        "uci": m,
                        "san": board.san(chess.Move.from_uci(m)),
                        "comment": entries[m].get("comment", ""),
                    }
                    for m in sorted(repertoire_moves(self.tree, key))
                ],
            }

    def master_game(self, game_id: str) -> dict:
        if game_id not in self.master_games.index:
            raise ApiError(404, "unknown_master_game", f"no master game with id {game_id}")
        row = self.master_games.loc[game_id]
        return {
            "game_id": game_id,
            "opening": row["opening"],
            "eco": row["eco"],
            "white": row["white"],
            "black": row["black"],
            "white_elo": int(row["white_elo"]),
            "black_elo": int(row["black_elo"]),
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "event": row["event"],
            "result": row["result"],
            "pgn": row["pgn"],
        }
