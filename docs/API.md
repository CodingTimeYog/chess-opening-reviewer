# Backend API

The contract between the FastAPI backend and the React app. The machine-readable version is
`web/src/api/openapi.json`, regenerated with `uv run python -m chess_trainer.api.export`.
Field-level descriptions live in `backend/chess_trainer/api/models.py`.

## Running it

From `backend/`:

```
uv run uvicorn chess_trainer.api.app:create_app --factory --port 8000
```

- Base URL `http://localhost:8000`. Interactive docs at `/docs`, schema at `/openapi.json`.
- Startup takes about 11 seconds (it loads the games, the checkpoint and the retrieval index).
- A move request takes about 1 second, because Stockfish runs on every user move.
- CORS allows `http://localhost:5173` and `http://localhost:3000` (and the `127.0.0.1`
  forms). Set `CHESS_TRAINER_CORS` to a comma-separated list of origins to change that.
- Sessions are held in memory and are lost when the server restarts.
- Needs `data/processed/` (run `python -m chess_trainer.data`), Stockfish at
  `STOCKFISH_PATH`, and `data/checkpoints/resnet_main.pt`. Without the checkpoint only the
  `book` agent is offered.

## Conventions

- Moves are UCI strings (`b5a4`, `e1g1` for castling, `e7e8q` for promotion). Responses also
  carry SAN (`Ba4`).
- `ply` counts half-moves already played: 0 is White's first move.
- Colours are `"white"` and `"black"`.
- Errors are `{"detail": {"error": "<code>", "message": "..."}}`. Request-validation errors
  (a missing or mistyped field) are FastAPI's standard 422.

| Status | `detail.error` | When |
|---|---|---|
| 400 | `unknown_opening`, `unknown_agent` | bad `opening_id` or agent name |
| 400 | `illegal_move`, `invalid_uci` | `detail.legal_moves` lists what can be played |
| 400 | `invalid_fen` | bad `fen` query parameter |
| 404 | `unknown_game`, `unknown_master_game` | unknown id |
| 409 | `game_not_active` | the opening is over or the game ended; reset to play again |
| 409 | `not_your_turn`, `not_watch_mode` | wrong endpoint for the game's mode or turn |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/openings` | openings to train |
| GET | `/agents` | available opponents and whether the LLM rewrite is on |
| POST | `/games` | start a session |
| GET | `/games/{game_id}` | full session state |
| POST | `/games/{game_id}/moves` | play the user's move |
| POST | `/games/{game_id}/agent-move?plies=1` | watch mode: advance the agents |
| POST | `/games/{game_id}/reset` | restart with the same settings and id |
| GET | `/games/{game_id}/review` | per-move review and summary |
| GET | `/positions/retrieve?fen=&k=5&opening=` | master games for a position |
| GET | `/positions/analyze?fen=` | engine, book, network and repertoire views |
| GET | `/master-games/{game_id}` | metadata and full PGN of a master game |

### GET /openings

```json
[
  {"id": "ruy_lopez", "name": "Ruy Lopez", "user_color": "white",
   "root_moves": ["e4", "e5", "Nf3", "Nc6", "Bb5"],
   "root_fen": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -"},
  {"id": "taimanov", "name": "Sicilian Taimanov", "user_color": "black",
   "root_moves": ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6"],
   "root_fen": "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq -"}
]
```

### GET /agents

```json
{"agents": ["book", "hybrid", "resnet_main"], "default": "hybrid", "llm": false}
```

`hybrid` is the book where the position is known, then the prep repertoire, then the network.

### POST /games

Request (only `opening_id` is required):

```json
{
  "opening_id": "ruy_lopez",
  "mode": "play",
  "agent": "hybrid",
  "user_color": null,
  "white_agent": null,
  "black_agent": null,
  "start": "root",
  "temperature": 1.0,
  "analyse_agent_moves": false
}
```

| Field | Values | Default | Meaning |
|---|---|---|---|
| `mode` | `play`, `watch` | `play` | user against an agent, or agent against agent |
| `agent` | name from `/agents` | `/agents` default | the opponent in play mode |
| `user_color` | `white`, `black`, `null` | the opening's side | play mode only |
| `white_agent`, `black_agent` | agent names | `agent` | watch mode |
| `start` | `root`, `initial` | `root` | start at the opening's root position, or at move one |
| `temperature` | 0 to 2 | 1.0 | 0 makes the agent always play its top move |
| `analyse_agent_moves` | boolean | false | also engine-check agent moves (slower) |

Response: 201 with a **GameState**. With `start: "root"` the root moves are already in
`moves` with `by: "setup"`. In play mode, if the agent is to move it has already moved, so
the state comes back with the user to play. In watch mode nothing moves until
`/agent-move` is called.

### GameState

```json
{
  "game_id": "3f9c1b2a7d4e",
  "opening_id": "ruy_lopez",
  "mode": "play",
  "start": "root",
  "user_color": "white",
  "agents": {"white": null, "black": "hybrid"},
  "status": "active",
  "fen": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
  "ply": 6,
  "turn": "white",
  "legal_moves": ["a2a3", "a2a4", "b5a4", "b5c6", "..."],
  "moves": [ "MoveRecord, ..." ],
  "events": [ "Event, ..." ],
  "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
}
```

- `status`: `active`, `opening_over` (an `opening_end` event fired) or `game_over`
  (mate or draw). `legal_moves` is empty unless the status is `active`.
- `agents`: the agent playing each side; `null` marks the user's side.

### MoveRecord

```json
{
  "ply": 6,
  "side": "white",
  "uci": "b5a4",
  "san": "Ba4",
  "by": "user",
  "agent": null,
  "tier": null,
  "fen": "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",
  "feedback": { "Feedback" },
  "similar_games": [ "Hit, ..." ]
}
```

- `by`: `setup` (root moves; `feedback` is null), `user` or `agent`.
- `agent` and `tier` are set on agent moves. `tier` is set only for `hybrid` and says
  which part answered: `book`, `repertoire` or `policy`.
- `similar_games`: up to 3 games from similar positions. These are never exact matches and
  are for browsing only; they are not cited in feedback.

### Feedback

```json
{
  "text": "Ba4 is the engine's first choice. Masters most often played Ba4 (2270 of 2398 games, 95%). Ba4 is your repertoire move. Example games: Caruana, Fabiano - Carlsen, Magnus, Croatia GCT 2019 (1/2-1/2), continued Ba4; ...",
  "llm_text": null,
  "facts": {
    "fen": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
    "ply": 6,
    "side": "white",
    "played": {"uci": "b5a4", "san": "Ba4"},
    "classification": "good",
    "cp_loss": 0,
    "best": {"uci": "b5a4", "san": "Ba4"},
    "eval_best_cp": 42,
    "eval_played_cp": 42,
    "move": {"castled": null, "captured": null, "piece": "bishop",
             "queens_off": false, "structure": []},
    "master": {"in_book": true, "games": 2398, "played_games": 2270,
               "top": [{"uci": "b5a4", "san": "Ba4", "games": 2270, "share": 94.7}]},
    "repertoire": {"covered": true, "played_in_repertoire": true, "played_note": null,
                   "moves": [{"uci": "b5a4", "san": "Ba4", "comment": ""}]},
    "citations": [ "Citation, ..." ]
  },
  "citations": [
    {"game_id": "e033440470774d9c", "white": "Caruana, Fabiano", "black": "Carlsen, Magnus",
     "white_elo": 2819, "black_elo": 2875, "year": 2019, "event": "Croatia GCT",
     "result": "1/2-1/2", "move_uci": "b5a4", "move_san": "Ba4"}
  ]
}
```

- `text` is always the deterministic template text. `llm_text` is null unless the server
  was started with the LLM rewrite on; show `llm_text` when present, else `text`.
- `classification`: `good`, `inaccuracy` (loss >= 50 cp), `mistake` (>= 100), `blunder`
  (>= 200). It is `null`, with `cp_loss` and `best` also null, on moves the engine did not
  check (agent moves, unless `analyse_agent_moves` was set).
- Evals are centipawns from the mover's point of view, clipped to +/-1000.
- `citations` (at most 3) are only ever games that reached this exact position. Fetch the
  full game with `/master-games/{game_id}`.
- `repertoire.played_note` is set when the prep files discuss the played move but do not
  recommend it: `{"nags": ["?!"], "comment": "..."}`.

### POST /games/{game_id}/moves

Request: `{"uci": "b5a4"}`. Response:

```json
{
  "move": { "MoveRecord for the user's move, with feedback" },
  "agent_reply": { "MoveRecord, or null if the game ended" },
  "events": [ "Event, ... from both moves" ],
  "state": { "GameState after both moves" }
}
```

An illegal move changes nothing and returns 400:

```json
{"detail": {"error": "illegal_move", "message": "a1a8 is not legal here",
            "legal_moves": ["a2a3", "a2a4", "..."]}}
```

### Event

```json
{
  "type": "eval_swing",
  "ply": 10,
  "severity": "blunder",
  "side": "white",
  "feedback": {"text": "Bb5 is a blunder: it loses about 5.4 pawns. Stockfish prefers O-O. ...",
               "facts": { "same shape as Feedback.facts" },
               "citations": [ "Citation, ..." ]}
}
```

| `type` | `severity` | Fires |
|---|---|---|
| `left_repertoire` | `info` | first move by a side that is not a repertoire move where prep has one |
| `left_book` | `info` | first move by a side that no master played from a known position |
| `eval_swing` | `inaccuracy`, `mistake`, `blunder` | an engine-checked move losing >= 50 cp |
| `castled` | `info` | `facts.move.castled` is `kingside` or `queenside` |
| `queens_off` | `info` | once, when the last queen leaves the board |
| `minor_piece_trade` | `info` | a minor piece is captured right after another; `facts.trade` names both |
| `structure_shift` | `info` | `facts.move.structure` lists the changes in words |
| `opening_end` | `info` | `facts.reason` is `ply_cap` (move 20) or `out_of_book`; the game stops |

### POST /games/{game_id}/agent-move?plies=1

Watch mode only. Plays up to `plies` half-moves (1 to 80) and stops early if the game ends.

```json
{"moves": [ "MoveRecord, ..." ], "events": [ "Event, ..." ], "state": { "GameState" }}
```

### GET /games/{game_id}/review

```json
{
  "game_id": "3f9c1b2a7d4e",
  "summary": {"analysed": 3, "mean_cp_loss": 188.3,
              "counts": {"good": 2, "inaccuracy": 0, "mistake": 0, "blunder": 1}},
  "moves": [
    {"ply": 6, "side": "white", "uci": "b5a4", "san": "Ba4", "by": "user",
     "classification": "good", "cp_loss": 0, "best_san": "Ba4", "text": "Ba4 is the engine's first choice. ..."}
  ],
  "events": [ "every Event of the game" ]
}
```

`moves` holds only engine-checked moves.

### GET /positions/retrieve

Query: `fen` (full FEN or the 4-field form), `k` (1 to 20, default 5), `opening` (optional;
restricts similar games to `ruy_lopez` or `taimanov`).

```json
{
  "fen": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
  "exact": [
    {"game_id": "e033440470774d9c", "score": 1.0, "exact": true,
     "fen_norm": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
     "ply": 6, "move_uci": "b5a4", "opening": "ruy_lopez", "eco": "C84",
     "white": "Caruana, Fabiano", "black": "Carlsen, Magnus", "white_elo": 2819,
     "black_elo": 2875, "year": 2019, "event": "Croatia GCT", "result": "1/2-1/2"}
  ],
  "similar": [ "Hit with exact: false and score < 1, ..." ]
}
```

`exact` is ordered by average rating, then most recent year. In `similar`, `fen_norm` and
`ply` describe the neighbouring position that matched, and `move_uci` is the move played
from it.

### GET /positions/analyze

```json
{
  "fen": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
  "best": {"uci": "b5a4", "san": "Ba4"},
  "eval_cp": 42,
  "book": {"in_book": true, "games": 2398,
           "moves": [{"uci": "b5a4", "san": "Ba4", "prob": 0.9466},
                     {"uci": "b5c6", "san": "Bxc6", "prob": 0.0534}]},
  "policy": {"agent": "resnet_main",
             "moves": [{"uci": "b5a4", "san": "Ba4", "prob": 0.9458}]},
  "repertoire": [{"uci": "b5a4", "san": "Ba4", "comment": ""}]
}
```

`book.moves` and `policy.moves` hold the top 5. `policy` is null when no checkpoint is
loaded.

### GET /master-games/{game_id}

```json
{
  "game_id": "e033440470774d9c", "opening": "ruy_lopez", "eco": "C84",
  "white": "Caruana, Fabiano", "black": "Carlsen, Magnus", "white_elo": 2819,
  "black_elo": 2875, "year": 2019, "event": "Croatia GCT", "result": "1/2-1/2",
  "pgn": "[Event \"Croatia GCT\"]\n[Site \"Zagreb\"]\n...\n\n1. e4 e5 2. Nf3 Nc6 ... 1/2-1/2"
}
```

`pgn` is headers plus the main line, with no comments or variations.

## Optional LLM rewrite

Off by default. Start the server with `CHESS_TRAINER_LLM=ollama` (and `ollama pull
qwen2.5:3b` once) to fill `llm_text` on engine-checked moves. `CHESS_TRAINER_LLM_MODEL`
and `OLLAMA_HOST` override the model and server. The model is given only the facts and the
template text. If Ollama is unreachable, `llm_text` stays null and nothing else changes.
`GET /agents` reports the flag as `llm`.

## Typical play flow

1. `GET /openings`, `GET /agents` to fill the pickers.
2. `POST /games` and render `state.fen`, `state.moves`, `state.legal_moves`.
3. On each user move, `POST /games/{id}/moves`. Show `move.feedback`, `events`, and
   `agent_reply`; replace the local state with `state`.
4. When `state.status` is no longer `active`, `GET /games/{id}/review`.
5. For a cited or similar game, `GET /master-games/{game_id}` and step through `pgn`.
