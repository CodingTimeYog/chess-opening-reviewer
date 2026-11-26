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
- Model: ResNet, 6 residual blocks, 64 channels, 465,441 params. Policy head: a 1x1 conv
  giving 4096 from-to logits (index `to * 64 + from`) plus 4x8 promotion logits (piece x
  target file) from the pooled features. A move's logit is its from-to logit plus, for a
  promotion, its promotion logit; the softmax runs over the legal moves only.
- Objective: cross-entropy vs the GM move, with per-sample weight 1.0 for the winner's
  moves, 0.75 for draws, 0.5 for the loser's moves (ablation: unweighted).
- Auxiliary value head (tanh, predicts result) with 0.25 loss weight. Ablation: off.
- Training: `uv run python -m chess_trainer.agents.train --name resnet_main --resume`.
  AdamW (lr 1e-3, weight decay 1e-4), batch 256, cosine LR, at most 10 epochs, early stop
  after 2 epochs without a val top-1 gain. Best checkpoint by val top-1 goes to
  `data/checkpoints/<name>.pt` (gitignored); the full state after every epoch goes to
  `<name>_last.pt`, which `--resume` continues from with the original hyperparameters. PyTorch CPU; university cluster is the
  fallback, not the plan.
- Play: sample from top-k=5 with temperature 1.0 for variety; eval uses argmax.
- Penultimate pooled features are exposed as `embed(board) -> np.ndarray` for retrieval.

**Hybrid (`chess_trainer.agents.hybrid`):** at each position, the book if the position
occurs in train; else uniform over the prep tree's in-repertoire moves if it has any; else
the learned policy. Never falls back to random. Evaluated as a third agent.

What the agent learns, in one sentence for the report: a mapping from board state to a
distribution over GM-preferred moves that generalizes to positions never seen in training.

## 5. Retrieval, triggers, feedback

**Retrieval (`chess_trainer.rag`):** hybrid. Exact `fen_norm` lookup first; if fewer than
k hits, nearest neighbors by cosine on policy embeddings (numpy brute force; corpus is
small), within the same opening. Each hit returns game metadata and the move played from
the matched position; the full PGN for the embedded viewer is added with the API. One hit
per game. The eval index holds train games only; the app index holds every game. At equal score the game with the higher average rating ranks first, then the most recent
year.

**Engine (`chess_trainer.engine`):** Stockfish via python-chess UCI. Depth 14, one
thread per analysis, several engines in parallel. A position is analysed once with MultiPV
equal to its number of legal moves, and every move's score is cached by
`(fen_norm, move, depth)` in `data/cache/engine.sqlite`, so scoring a new agent or
checkpoint on already-analysed positions needs no engine calls. Live feedback uses
`LiveAnalyser` instead: two short searches per move (best move, then the move played),
cached in a separate `live` table, because scoring every legal move takes about 10 s.
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

All eight are built. `minor_piece_trade` fires when a minor piece is captured on the move
right after another one was; `queens_off` fires once per game.

**Feedback (`chess_trainer.feedback`):** two layers.
1. **Template layer** (always on): deterministic text from the structured facts:
   classification, cp loss, engine best move, most common GM move, repertoire move if any,
   the prep author's comment, and up to 3 cited games. Citations are only ever games that
   reached the exact position. This is the ground truth for the correctness metric.
2. **LLM layer** (optional, off by default): local Ollama model rewrites the template facts
   into prose explaining the idea. It receives only the structured JSON and the template
   text and must cite nothing outside them. Model: `qwen2.5:3b` (`ollama pull` it first).
   No paid APIs. Switched on for the API with `CHESS_TRAINER_LLM=ollama` and for the
   feedback eval with `--llm ollama`; the template text is always returned alongside, and
   is what the user sees if Ollama is unreachable.

## 6. Evaluation harness (`chess_trainer.eval`)

`uv run python -m chess_trainer.eval --agent {book,book-weighted,policy,hybrid} --split test`
(`--checkpoint <name>` for the policy and hybrid, `--compare book resnet-main` to prepend
side-by-side tables against earlier runs)
writes `results/<agent>-<split>/metrics.json` and `metrics.md`. Every run reports each
opening separately, on all positions and on the strict subset. Plots via
`chess_trainer.eval.plots`.

| Metric | Definition |
|---|---|
| Opening coverage | % held-out positions with a non-fallback move; % test lines, replayed along the master moves, where the agent's first 10 own moves are all non-fallback and its top move is theory (played from that position in train, or in repertoire). Lines with fewer than 10 own moves are excluded. Reported with the agent on each side |
| Move quality | On every held-out position: top-1/top-3 agreement with the GM move (policy ties scored as the expected hit under a random tie-break) plus legal-move sanity checks. On a seeded sample of distinct positions, 500 seen + 500 strict per opening: mean/median centipawn loss vs Stockfish best (scores clipped to +/-1000 cp), blunder rate (CPL >= 100), and WDL expectation loss. Reported for the agent's top move, for sampling from its policy, and for the GM move as reference |
| Retrieval relevance | `python -m chess_trainer.eval.retrieval [--checkpoint <name>]`. precision@5 and MRR on the move-quality sample, train-only index; relevant = retrieved game reaches the exact position, or shares ECO and the next GM move. Methods: random same-opening same-ply control, exact lookup, hybrid (exact + embeddings) |
| Feedback correctness | `python -m chess_trainer.eval.feedback`. Labeled set: for each sample position the GM move, the engine's best move and a seeded random move, with ground truth from the full-move Stockfish cache. The live two-search feedback path is scored on classification accuracy, best move named correctly, text consistent with facts, and zero hallucinated citations (every cited game exists in the corpus and reached the position) |

Headline experiment: book vs policy on strict held-out positions, plus ablations
(result weighting, value head). Engine evals of the test set are precomputed and cached.

## 7. Backend API (FastAPI, `chess_trainer.api`)

`uv run uvicorn chess_trainer.api.app:create_app --factory --port 8000`. Serves on
`localhost:8000`, interactive docs at `/docs`. The contract is documented in `docs/API.md`
and exported to `web/src/api/openapi.json` with `python -m chess_trainer.api.export`.
Sessions are in memory. CORS allows the local React dev server (ports 5173 and 3000).

| Method | Path | Purpose |
|---|---|---|
| GET | `/openings`, `/agents` | repertoire roots; available opponents (`book`, `hybrid`, `resnet_main`) |
| POST | `/games` | new session `{opening_id, mode: play or watch, agent, user_color, start, temperature}`; returns the full state, with the agent's first move already played if it is to move |
| POST | `/games/{id}/moves` | user move `{uci}` returns `{move, agent_reply, events[], state}`; an illegal move is a 400 listing the legal moves |
| POST | `/games/{id}/agent-move?plies=` | watch mode: advance agent vs agent |
| POST | `/games/{id}/reset` | restart with the same settings |
| GET | `/games/{id}` | full state: moves with feedback, events, legal moves |
| GET | `/games/{id}/review` | per-move cp loss plus feedback, and a summary |
| GET | `/positions/retrieve?fen=&k=&opening=` | exact-match and similar master games |
| GET | `/positions/analyze?fen=` | engine eval, best move, book and policy distributions, repertoire moves |
| GET | `/master-games/{game_id}` | metadata and full PGN of a master game |

A session starts at the opening's root position by default (the book mixes both openings
before the roots, so starting from move one can leave the chosen opening). Each move record
carries `feedback: {text, llm_text, facts, citations}` and `similar_games`. Events carry
`{type, ply, severity, side, feedback: {text, facts, citations}}`. Only user moves are
engine-checked unless the session asks otherwise. The frontend never calls Stockfish or
Ollama directly.

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
