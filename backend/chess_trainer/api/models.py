"""Request and response shapes. These are the contract the web app is built against."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Color = Literal["white", "black"]
Severity = Literal["info", "inaccuracy", "mistake", "blunder"]


class Opening(BaseModel):
    id: str
    name: str
    user_color: Color = Field(description="The side whose repertoire this is.")
    root_moves: list[str] = Field(description="SAN moves leading to the opening's root position.")
    root_fen: str


class Agents(BaseModel):
    agents: list[str]
    default: str
    llm: bool = Field(description="True when the optional LLM rewrite is switched on.")


class GameCreate(BaseModel):
    opening_id: str
    mode: Literal["play", "watch"] = Field(
        "play", description="play: user against an agent. watch: agent against agent."
    )
    agent: str | None = Field(None, description="Opponent in play mode. Default: see GET /agents.")
    user_color: Color | None = Field(None, description="Play mode only. Default: the opening's.")
    white_agent: str | None = Field(None, description="Watch mode. Default: `agent`.")
    black_agent: str | None = Field(None, description="Watch mode. Default: `agent`.")
    start: Literal["root", "initial"] = Field(
        "root", description="root: begin at the opening's root position. initial: move one."
    )
    temperature: float = Field(
        1.0, ge=0, le=2, description="Agent sampling temperature; 0 always plays the top move."
    )
    analyse_agent_moves: bool = Field(
        False, description="Also run the engine on agent moves (slower; mainly for watch mode)."
    )


class MoveIn(BaseModel):
    uci: str = Field(examples=["b5a4"])


class Citation(BaseModel):
    game_id: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    year: int | None
    event: str
    result: str
    move_uci: str
    move_san: str


class Feedback(BaseModel):
    text: str = Field(description="Deterministic template text.")
    llm_text: str | None = Field(None, description="LLM rewrite; null unless switched on.")
    facts: dict[str, Any] = Field(description="Structured facts the text was built from.")
    citations: list[Citation] = Field(description="Master games that reached this exact position.")


class Hit(BaseModel):
    game_id: str
    score: float = Field(description="1.0 for an exact match, else cosine similarity.")
    exact: bool
    fen_norm: str = Field(description="The indexed position that matched.")
    ply: int
    move_uci: str = Field(description="The move the master played from that position.")
    opening: str
    eco: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    year: int | None
    event: str
    result: str


class MoveRecord(BaseModel):
    ply: int = Field(description="Half-moves played before this move; 0 is White's first.")
    side: Color
    uci: str
    san: str
    by: Literal["setup", "user", "agent"]
    agent: str | None = None
    tier: str | None = Field(
        None, description="Hybrid agent only: book, repertoire or policy answered."
    )
    fen: str = Field(description="Position after the move.")
    feedback: Feedback | None = Field(None, description="Null for setup moves.")
    similar_games: list[Hit] = Field(
        default_factory=list, description="Games from similar positions (never exact matches)."
    )


class EventFeedback(BaseModel):
    text: str
    facts: dict[str, Any]
    citations: list[Citation]


class Event(BaseModel):
    type: Literal[
        "left_repertoire", "left_book", "eval_swing", "castled", "queens_off",
        "minor_piece_trade", "structure_shift", "opening_end",
    ]  # fmt: skip
    ply: int
    severity: Severity
    side: Color
    feedback: EventFeedback


class GameState(BaseModel):
    game_id: str
    opening_id: str
    mode: Literal["play", "watch"]
    start: Literal["root", "initial"]
    user_color: Color | None
    agents: dict[Color, str | None]
    status: Literal["active", "opening_over", "game_over"]
    fen: str
    ply: int
    turn: Color
    legal_moves: list[str] = Field(description="UCI; empty once the game is no longer active.")
    moves: list[MoveRecord]
    events: list[Event]
    pgn: str = Field(description="Movetext so far.")


class MoveResponse(BaseModel):
    move: MoveRecord
    agent_reply: MoveRecord | None
    events: list[Event] = Field(description="Events from the user move and the agent reply.")
    state: GameState


class AgentMoveResponse(BaseModel):
    moves: list[MoveRecord]
    events: list[Event]
    state: GameState


class ReviewMove(BaseModel):
    ply: int
    side: Color
    uci: str
    san: str
    by: str
    classification: str
    cp_loss: int
    best_san: str
    text: str


class ReviewSummary(BaseModel):
    analysed: int
    mean_cp_loss: float | None
    counts: dict[str, int]


class Review(BaseModel):
    game_id: str
    summary: ReviewSummary
    moves: list[ReviewMove] = Field(description="Every move the engine analysed.")
    events: list[Event]


class Retrieval(BaseModel):
    fen: str
    exact: list[Hit]
    similar: list[Hit]


class MoveProb(BaseModel):
    uci: str
    san: str
    prob: float


class BookView(BaseModel):
    in_book: bool
    games: int
    moves: list[MoveProb]


class PolicyView(BaseModel):
    agent: str
    moves: list[MoveProb]


class RepertoireMove(BaseModel):
    uci: str
    san: str
    comment: str


class Analysis(BaseModel):
    fen: str
    best: dict[str, str]
    eval_cp: int = Field(description="Side to move's point of view, clipped to +/-1000.")
    book: BookView
    policy: PolicyView | None
    repertoire: list[RepertoireMove]


class MasterGame(BaseModel):
    game_id: str
    opening: str
    eco: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    year: int | None
    event: str
    result: str
    pgn: str


class IllegalMove(BaseModel):
    error: str
    message: str
    legal_moves: list[str] = []
