# Milestone plan: Nov 19 to Dec 10, 2026

Two tracks run in parallel: **ML/backend** (Yogi) and **UI** (teammate). Each phase has a
definition of done. Course priority: agent, baseline, eval and feedback first, UI second.
Thanksgiving is Thu Nov 26; the plan assumes a light Nov 26 and 27.

## Phase 0: Scaffold (Wed Nov 19 to Thu Nov 20)
Backend: uv project, pytest, ruff, `.gitignore` for data, config module, Stockfish wrapper
with cache, fixture PGNs, `docs/API.md` plus exported `openapi.json` from stub routes.
UI: Vite + React + TS scaffold, react-chessboard + chess.js, MSW mocks from `openapi.json`,
playable local board (no agent yet).
**Done when:** `uv run pytest` and `pnpm test` are both green; stub API returns mocked shapes.

## Phase 1: Data, baseline, harness skeleton (Fri Nov 21 to Mon Nov 24)
- Repertoire tree from prep PGNs; ingest Lichess Elite and OTB PGNs; rating filter;
  truncate; Parquet; seeded game-level split; strict-subset computation.
- Book baseline agent with fallback logging.
- Eval harness CLI with **opening coverage** implemented end to end; engine cache
  precompute for the test split started in the background.
UI: opening picker, game page, move list component, mock events rendered as annotations.
**Done when:** coverage numbers for the book exist in `results/`; a data stats table
(games, positions, strict subset size per opening) exists for the report.

## Phase 2: Learned policy (Tue Nov 25 to Fri Nov 28)
- Board encoder, ResNet policy plus value head, training script, checkpointing, `embed()`.
- Policy agent; **move quality** metric; first book-vs-policy run on all and strict test sets.
- Ablation flags: result weighting on/off, value head on/off.
UI: retrieved-games embedded viewer (step through PGN on a mini board), review page.
**Done when:** a trained checkpoint exists and the headline table (coverage plus move
quality, book vs policy, all vs strict) is generated.

## Phase 3: Retrieval, triggers, feedback (Sat Nov 29 to Tue Dec 2)
- Hybrid retrieval (exact plus embedding); **retrieval relevance** metric.
- Trigger engine (repertoire/book exit, eval swings, structure shifts, trades, castling).
- Template feedback; labeled correctness set; **feedback correctness** metric.
- Slides v1 by **Sun Nov 30** (problem, importance, approach, progress so far).
UI: event/feedback panel wired to mock events; polish move annotations.
**Done when:** all four metrics run from one command on both agents.

## Phase 4: Integration (Wed Dec 3 to Fri Dec 5)
- Real FastAPI routes replace stubs; UI switches from MSW to the live API.
- Ollama LLM layer behind a flag; correctness metric rerun with LLM on vs off.
- CLI agent-vs-agent self-play used to generate extra eval games (not in UI).
**Done when:** a full human-vs-agent game with live feedback and retrieved games works
end to end on localhost.

## Phase 5: Experiments and freeze (Sat Dec 6 to Sun Dec 7)
- Final eval runs with fixed seeds; ablations; plots; `results/` tidy.
- **Feature freeze Sun Dec 7.** Rehearse demo and talk.

## Phase 6: Report (Mon Dec 8 to Wed Dec 10)
- IEEE two-column, at most 8 pages including references. Sections: Introduction (problem,
  importance), Related Work, Approach (data, learned agent, baseline, RAG, feedback),
  Evaluation (4 metrics, strict vs all, ablations), Discussion (difficulties), Conclusion.
- Q&A prep: why behavior cloning and not RL, how the strict split avoids leakage, why
  hybrid retrieval, what the LLM may and may not say.

## Risk register
| Risk | Mitigation |
|---|---|
| Too few 2500+ games in some Sicilian sub-lines | fall back to 2400+ for that line and report it; prep PGNs still define the repertoire |
| CPU training too slow | shrink to 4 blocks / 48 channels; subsample train set; cluster as last resort |
| Ollama output wrong or slow | template layer is always the shipped default; LLM is an ablation |
| Teammate blocked on API | MSW mocks from day one; API contract frozen in `docs/API.md` |
| Stockfish eval time on test set | depth 14, cached, precomputed in Phase 1 in the background |
