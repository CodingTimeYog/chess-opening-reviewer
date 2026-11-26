# Chess Opening Reviewer — project spec

RAG-augmented chess opening trainer for CS4710 (UVA, Fall 2026). Two-person team.
Deadlines: feature freeze **Sun Dec 7**, presentation **Dec 1 / 3 / 8** (TBD), report **Wed Dec 10**.
Detailed milestones live in `docs/PLAN.md`. This file is the design contract; change it
deliberately and in its own commit.

## 1. Problem statement (for the report)

Opening study tools are either static books (frequency lookup, silent one move out of book)
or raw engines (accurate, no explanation). We build an agent that (a) **learns** a policy
over opening positions from GM games so it plays varied, strong lines and keeps playing
sensibly out of book, (b) detects player mistakes and thematic shifts, and (c) grounds its
feedback in **retrieved master games**. We compare the learned agent against an
opening-book baseline on held-out positions using four metrics.

## 2. Scope

**Repertoire (v1):**
- White: Ruy Lopez (1.e4 e5 2.Nf3 Nc6 3.Bb5).
- Black: Sicilian Taimanov (1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nc6).
- A game belongs to an opening when it **reaches the root position** (matched by
  `fen_norm`) within its first 12 plies, so transpositions such as 2...Nc6 then 4...e6
  count. Sub-lines are whatever the prep PGNs cover.
- Agent plays both colors. Opening phase ends at ply 40 (move 20) or when the agent is out
  of both repertoire and book for 3 consecutive own moves, whichever comes first.

**In scope:** data pipeline, learned policy, book baseline, retrieval, mistake/trigger
detection, template + optional LLM feedback, eval harness, FastAPI backend, React UI
(human vs agent, move-list annotations, embedded retrieved-game viewer).
**Out of scope (v1):** the Italian and any other opening, Lichess data, RL/self-play, accounts/persistence, hosting, agent-vs-agent in UI
(CLI only, used for eval), middlegame/endgame play.

## 3. Data

Two sources, both PGN, both in `data/raw/` (gitignored, never committed). The corpus is
OTB-only: ChessBase exports, no Lichess.

| Source | Role | Filter |
|---|---|---|
| Opening prep PGNs (`data/raw/prep/`) | **Repertoire tree**: ground truth for "in repertoire", top-priority reference in feedback | only chapters whose main line reaches a repertoire root; puzzle chapters that start from a FEN are skipped |
| OTB master PGNs (`data/raw/masters/`) | Training + retrieval corpus, source of citations | reaches a repertoire root; both players rated, Ruy Lopez both >= 2500, Taimanov both >= 2400; decisive or drawn result; exact duplicates removed |

`data/raw/archive/` holds PGNs we deliberately do not use (part 2 of the 1.e4 course, which
is White against the Sicilian and would pollute the Black Taimanov repertoire).
ChessBase `.cbh`/`.cbv` files must be exported to PGN first; we do not parse them.

**"In repertoire" rule (`chess_trainer.data.repertoire`):** the tree stores every move the
prep files discuss, each flagged `in_repertoire`.
- Repertoire side (White in the Ruy Lopez, Black in the Taimanov): only a chapter's first
  choice at a position counts, and never a move annotated `?`, `??` or `?!`.
- Opponent side: every branch counts, so the trainer covers all opponent replies.
- A move counts only if the position it is played from is itself on the repertoire path.
  Lines below a rejected repertoire-side move stay in the tree for feedback but are off it.

**Pipeline (`chess_trainer.data`):** `uv run python -m chess_trainer.data`. Parse with
python-chess, classify by root position, filter, truncate at ply 40, emit one row per
position: `game_id, opening, split, strict, ply, fen_norm, move_uci, result, white_elo,
black_elo, eco, source, event, year, white, black`. `fen_norm` is the FEN with
halfmove/fullmove counters stripped (en-passant square only when a capture is legal).
Stored as Parquet in `data/processed/` (`games.parquet`, `positions.parquet`,
`repertoire.json`). The exact filters and per-opening game counts at every filter stage are
written to `results/data_stats.md` for the report.

**Split:** by **game**, never by position, seeded, 80/10/10 train/val/test within each
opening. Metrics are reported on
(a) all test positions and (b) the **strict** subset whose `fen_norm` never appears in train.
State this split in the report; it is what makes "learning vs lookup" measurable.

## 4. Agents

Both implement `Agent.policy(board) -> dict[move_uci, prob]`, `Agent.choose(board, temperature)`
and `Agent.is_fallback(board)` (true when the agent has no knowledge of the position).

**Book baseline (`chess_trainer.agents.book`):** `dict[fen_norm -> Counter[move_uci]]`
built from the train split. Samples proportional to frequency (optional result weighting).
Out of book: uniform random legal move, and `fallback=True` is recorded. This is the
"frequency sampling" control.

**Learned policy (`chess_trainer.agents.policy`):**
- Input: 19 planes x 8x8: 12 piece planes, side to move, 4 castling rights, en-passant
  file, normalized ply. Board is always oriented from the side to move.
- Model: ResNet, 6 residual blocks, 64 channels, policy head to 4096 from-to logits
  plus 4x8 promotion logits (4128 total), masked to legal moves. About 0.4M params.
- Objective: cross-entropy vs the GM move, with per-sample weight 1.0 for the winner's
  moves, 0.75 for draws, 0.5 for the loser's moves (ablation: unweighted).
- Auxiliary value head (tanh, predicts result) with 0.25 loss weight. Ablation: off.
- Training: AdamW, cosine LR, at most 10 epochs, early stop on val top-1. PyTorch CPU on
  16 threads; university cluster is the fallback, not the plan.
- Play: sample from top-k=5 with temperature 1.0 for variety; eval uses argmax.
- Penultimate pooled features are exposed as `embed(board) -> np.ndarray` for retrieval.

What the agent learns, in one sentence for the report: a mapping from board state to a
distribution over GM-preferred moves that generalizes to positions never seen in training.

## 5. Retrieval, triggers, feedback

**Retrieval (`chess_trainer.rag`):** hybrid. Exact `fen_norm` lookup first; if fewer than
k hits, nearest neighbors by cosine on policy embeddings (numpy brute force; corpus is
small). Each hit returns game metadata, the move played from this position, and the full
PGN for the embedded viewer. At equal score the game with the higher average rating ranks first, then the most recent
year.

**Engine (`chess_trainer.engine`):** Stockfish via python-chess UCI. Depth 14, one
thread per analysis, cached by `fen_norm` in `data/cache/engine.sqlite`.
Path from `STOCKFISH_PATH`; default
`C:/Users/aswin/Desktop/Chess_Engines/stockfish/stockfish-windows-x86-64-avx2.exe`.

**Trigger engine (`chess_trainer.feedback.triggers`):** feedback fires on events, not every
move. Events, each with a structured payload:
- `left_repertoire` / `left_book` (first time only per game, per side)
- `eval_swing`: centipawn loss >= 50 (inaccuracy), >= 100 (mistake), >= 200 (blunder)
- `structure_shift`: pawn-structure signature changed (central exchange, new isolated or
  doubled pawn, pawn chain locked, open file created)
- `castled`, `queens_off`, `minor_piece_trade`
- `opening_end` (ply cap or book exhaustion), which triggers the full-game review

**Feedback (`chess_trainer.feedback`):** two layers.
1. **Template layer** (always on): deterministic text from the structured facts:
   classification, cp loss, engine best move, most common GM move, repertoire move if any,
   and up to 3 cited games. This is the ground truth for the correctness metric.
2. **LLM layer** (optional, `--llm ollama`): local Ollama model rewrites the template facts
   into prose explaining the idea. It receives only the structured JSON and must cite
   nothing outside it. Model: `qwen2.5:3b` (pull on first use). No paid APIs.

## 6. Evaluation harness (`chess_trainer.eval`)

`uv run python -m chess_trainer.eval --agent {book,policy} --split test`
writes `results/<agent>-<split>/metrics.json` and `metrics.md`. Every run reports each
opening separately, on all positions and on the strict subset. Plots via
`chess_trainer.eval.plots`.

| Metric | Definition |
|---|---|
| Opening coverage | % held-out positions with a non-fallback move; % test lines, replayed along the master moves, where the agent's first 10 own moves are all non-fallback and its top move is theory (played from that position in train, or in repertoire). Lines with fewer than 10 own moves are excluded. Reported with the agent on each side |
| Move quality | mean/median centipawn loss vs Stockfish best; top-1/top-3 agreement with the GM move; top-1 agreement with Stockfish top-3 |
| Retrieval relevance | precision@5 and MRR; relevant = retrieved game reaches the exact position, or shares ECO and the next GM move |
| Feedback correctness | on a labeled set of (position, player move) with engine ground truth: classification accuracy, best move named correctly, zero hallucinated citations (every cited game exists in the corpus and reached the position) |

Headline experiment: book vs policy on strict held-out positions, plus ablations
(result weighting, value head). Engine evals of the test set are precomputed and cached.

## 7. Backend API (FastAPI, `chess_trainer.api`)

Serves on `localhost:8000`, OpenAPI at `/docs`, schema exported to `web/src/api/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/openings` | repertoire roots the user can pick |
| POST | `/games` | new game `{opening_id, user_color, agent}` returns `{game_id, fen}` |
| POST | `/games/{id}/moves` | user move `{uci}` returns `{fen, legal_moves, events[], agent_reply?}` |
| GET | `/games/{id}` | full state: moves, annotations, events |
| GET | `/games/{id}/review` | post-game review: per-move cp loss plus feedback |
| GET | `/positions/retrieve?fen=&k=` | retrieved master games with PGN |
| GET | `/positions/analyze?fen=` | engine eval, best move, book and policy distributions |

Events in move responses carry `{type, ply, severity, feedback: {text, facts, citations}}`.
The frontend never calls Stockfish or Ollama directly.

## 8. Repo layout and ownership

```
backend/            Python package `chess_trainer` (uv, pytest)   -- owner: Yogi
  chess_trainer/{data,agents,rag,engine,feedback,eval,api}/
  tests/
  pyproject.toml
web/                React + Vite + TypeScript (pnpm, vitest)       -- owner: teammate
  src/{api,components,pages}/
data/               raw/ processed/ cache/ gitignored; tiny fixtures live in backend/tests
results/            eval outputs (committed: metrics.md and plots only)
docs/               PLAN.md, API.md, report notes
scripts/            one-off data download/convert helpers
```

Conflict rules: each person edits only their tree. Shared files are `CLAUDE.md`,
`docs/API.md`, `web/src/api/openapi.json`; announce before touching them.
Until the API is live the web app runs against MSW mocks derived from `openapi.json`.

## 9. Workflow

- Both on `main`, small commits, commit messages `area: what` (e.g. `data: game-level split`).
- **TDD**: write the failing test first. Backend `uv run pytest`, web `pnpm test`. Tests
  must pass before every commit. No CI, no hooks.
- Python 3.13, `uv` manages the venv and lockfile. `ruff` for lint and format.
- Tiny fixture PGNs (<= 20 games) live in `backend/tests/fixtures/`; real data never in git.
- Determinism: every sampler and split takes a seed; eval runs record seed and git sha.
- Don't optimize prematurely. Brute-force numpy and SQLite are fine at this scale.

## 10. Environment facts

- Dev box: Ryzen 7 8845HS, 16 threads, about 14 GB RAM, Radeon 780M (no CUDA, PyTorch CPU).
- Stockfish 18 at the path above. Ollama installed; only `llama3.2-vision` pulled so far.
- uv 0.11, Node 26, pnpm 10.
