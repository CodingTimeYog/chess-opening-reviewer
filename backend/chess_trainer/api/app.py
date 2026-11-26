"""FastAPI app.

uv run uvicorn chess_trainer.api.app:create_app --factory --port 8000

Interactive docs at http://localhost:8000/docs. CORS is open to the local React dev
server; set CHESS_TRAINER_CORS to a comma-separated list of origins to change that.
"""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chess_trainer import config
from chess_trainer.api import models
from chess_trainer.api.service import ApiError, TrainerService

DEFAULT_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
)
ERRORS = {
    400: {"model": models.IllegalMove, "description": "Bad request; `detail` says why."},
    404: {"description": "Unknown id."},
    409: {"description": "Not allowed in the game's current state."},
}


def default_service() -> TrainerService:
    """The real thing: processed data, trained checkpoint, Stockfish, optional LLM."""
    from chess_trainer.data.repertoire import load_tree
    from chess_trainer.engine import EvalCache, LiveAnalyser
    from chess_trainer.feedback.llm import rewriter_from_env
    from chess_trainer.rag import Retriever, checkpoint_signature, index_path, load_or_build_index

    positions = pd.read_parquet(config.PROCESSED_DIR / "positions.parquet")
    games = pd.read_parquet(config.PROCESSED_DIR / "games.parquet")
    tree = load_tree(config.PROCESSED_DIR / "repertoire.json")
    retriever = Retriever(positions, splits=None)
    name = os.environ.get("CHESS_TRAINER_CHECKPOINT", "resnet_main")
    checkpoint = config.CHECKPOINT_DIR / f"{name}.pt"
    network = None
    if checkpoint.exists():
        from chess_trainer.agents.policy import PolicyAgent  # imports torch

        network = PolicyAgent.from_checkpoint(checkpoint)
        load_or_build_index(retriever, network, index_path(None), checkpoint_signature(checkpoint))
    return TrainerService(
        positions,
        games,
        tree,
        analyser=LiveAnalyser(EvalCache()),
        network=network,
        network_name=name,
        retriever=retriever,
        rewriter=rewriter_from_env(),
    )


def create_app(service: TrainerService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = default_service()
        yield
        analyser = getattr(app.state.service, "analyser", None)
        if hasattr(analyser, "close"):
            analyser.close()

    app = FastAPI(
        title="Chess Opening Reviewer",
        version="0.1.0",
        description="Opening trainer: play a repertoire against a learned agent, with "
        "engine-checked feedback grounded in retrieved master games.",
        lifespan=lifespan,
    )
    app.state.service = service
    origins = os.environ.get("CHESS_TRAINER_CORS", DEFAULT_ORIGINS).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content={"detail": error.detail})

    def svc() -> TrainerService:
        return app.state.service

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/openings", response_model=list[models.Opening], tags=["meta"])
    def openings():
        """Openings the user can train."""
        return svc().openings()

    @app.get("/agents", response_model=models.Agents, tags=["meta"])
    def agents():
        """Agents available as opponents, and whether the LLM rewrite is on."""
        return svc().agent_names()

    @app.post(
        "/games", response_model=models.GameState, status_code=201, responses=ERRORS, tags=["games"]
    )
    def create_game(body: models.GameCreate):
        """Start a session. In play mode the agent moves first if it is its turn."""
        return svc().create_game(body.model_dump())

    @app.get("/games/{game_id}", response_model=models.GameState, responses=ERRORS, tags=["games"])
    def get_game(game_id: str):
        """Full state: moves with feedback, events, legal moves."""
        return svc().state(game_id)

    @app.post(
        "/games/{game_id}/moves",
        response_model=models.MoveResponse,
        responses=ERRORS,
        tags=["games"],
    )
    def post_move(game_id: str, body: models.MoveIn):
        """Play the user's move. Returns its feedback, any events, and the agent's reply.
        An illegal move is a 400 whose `detail.legal_moves` lists what can be played."""
        return svc().user_move(game_id, body.uci)

    @app.post(
        "/games/{game_id}/agent-move",
        response_model=models.AgentMoveResponse,
        responses=ERRORS,
        tags=["games"],
    )
    def post_agent_move(game_id: str, plies: int = Query(1, ge=1, le=80)):
        """Watch mode: let the agents play the next `plies` half-moves."""
        return svc().agent_move(game_id, plies)

    @app.post(
        "/games/{game_id}/reset", response_model=models.GameState, responses=ERRORS, tags=["games"]
    )
    def reset_game(game_id: str):
        """Restart the session with the same settings and id."""
        return svc().reset(game_id)

    @app.get(
        "/games/{game_id}/review", response_model=models.Review, responses=ERRORS, tags=["games"]
    )
    def review(game_id: str):
        """Per-move centipawn loss and feedback for every analysed move, with a summary."""
        return svc().review(game_id)

    @app.get(
        "/positions/retrieve", response_model=models.Retrieval, responses=ERRORS, tags=["positions"]
    )
    def retrieve(
        fen: str = Query(description="Full FEN or the 4-field normalised FEN."),
        k: int = Query(5, ge=1, le=20),
        opening: str | None = Query(None, description="Restrict similar games to an opening."),
    ):
        """Master games that reached this position, and games from similar positions."""
        return svc().retrieve(fen, k, opening)

    @app.get(
        "/positions/analyze", response_model=models.Analysis, responses=ERRORS, tags=["positions"]
    )
    def analyze(fen: str = Query(description="Full FEN or the 4-field normalised FEN.")):
        """Engine best move and eval, the book's and the network's top moves, repertoire moves."""
        return svc().analyze(fen)

    @app.get(
        "/master-games/{game_id}",
        response_model=models.MasterGame,
        responses=ERRORS,
        tags=["positions"],
    )
    def master_game(game_id: str):
        """Metadata and full PGN of a master game, for the embedded viewer."""
        return svc().master_game(game_id)

    return app
