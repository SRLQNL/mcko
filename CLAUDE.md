# MCKO Handoff for Claude Code

## Current Runtime
This repository is a small Python 3.8 X11/Tkinter desktop app for solving tasks
from text, screenshots, clipboard images, and pasted images.

The active runtime is the three-model solver in `app/geometry_solver.py`.
Do not switch it back to the old single-model API client without an explicit
request.

Model roles (env var → default):

- `MODEL_PARSER=qwen/qwen3-vl-32b-instruct`
  - parser, OCR, visual extraction
  - focused option arbiter for hard "select option numbers" tasks
- `MODEL_SOLVER=deepseek/deepseek-v4-flash`
  - primary solver (text-only; does NOT receive images)
- `MODEL_VERIFIER=meta-llama/llama-4-maverick`
  - independent verifier
  - JSON repair model for malformed solver output

Text-only requests go through a solver fast path. Image and image+text requests
go through parser -> solver+verifier (parallel) -> consensus.

The user-facing output contract is strict answer-only:

```text
1) 69
```

For several independent tasks:

```text
1) ...
2) ...
```

Do not expose reasoning, JSON, model names, tracebacks, or explanations in the UI/clipboard answer.

## Product Requirements
The app must solve arbitrary school/exam-style and general tasks, not only
geometry or math. It should handle:

- math screenshots
- geometry diagrams
- option-selection tasks
- Russian language tasks
- foreign language tasks
- informatics tasks
- several images with one task per image
- one image containing several tasks

Avoid task-specific hardcoded answers. A previous exact-solver experiment was
removed because it made the project less general and misled testing.

## Key Files
- `main.py` - application startup, Tk root, hotkey callbacks, response worker.
- `app/geometry_solver.py` - active solver, request building, JSON repair, consensus, answer rendering.
- `app/config.py` - `.env` loading, base64 API-key decoding, model/env defaults.
- `app/scenario2.py` - background clipboard request flow.
- `app/hotkeys.py` - global hotkeys via `pynput`.
- `app/clipboard.py` - clipboard text/image helpers.
- `app/screenshot.py` - screenshot capture.
- `ui/window.py` - small chat window and hidden-window image insertion.
- `ui/input_field.py` - text input and image labels.
- `ui/chat_view.py` - assistant placeholder and answer rendering.
- `tests/test_geometry_solver.py` - unit tests for selection/repair/normalization logic.
- `tests/run_solver_fixture_suite.py` - live model regression runner.

## Solver Pipeline
High-level flow in `GeometryPhotoSolver.solve_content_blocks()`:

1. Extract text blocks and image data URLs.
2. If text-only, call `_solve_text_only()`.
3. If images exist:
   - `_prepare_variants()` builds image variants (full + optional text/diagram crops).
   - `_call_parser()` OCRs and extracts task structure (parser model, gets images).
   - `_call_parallel_solvers()` runs solver and verifier concurrently:
     - Solver receives parser OCR extract as text only (no images — model is text-only).
     - Verifier receives parser OCR extract plus original images, solves independently.
     - EMS: verifier is skipped if solver confidence ≥ 0.95 on non-option tasks.
   - `_compare_results()` scores consistency; agreement boost when solver+verifier match.
   - `_should_run_option_arbiter()` may run a focused parser option arbiter pass.
   - `_should_run_self_check()` may trigger a second model round.
   - `_pick_user_answer()` chooses a final answer or returns fallback.
4. On unexpected failures, return `1) Не удалось определить ответ`.

All API requests use:
- `"provider": {"allow_fallbacks": True, "data_collection": "allow", "sort": "throughput"}`
- `"plugins": [{"id": "response-healing"}]` — free OpenRouter JSON repair, 80-99.8% defect reduction.

## Known Failure Modes
These are real issues seen during testing:

- Parser (Qwen) can return upstream `429`. When that happens, parser quality drops and
  downstream answers can degrade.
- Solver can return prose, reasoning, or malformed JSON despite JSON prompts.
  Response healing plugin reduces this significantly server-side.
- Verifier (Llama) can confidently verify wrong option subsets.
- Local salvage from prose is dangerous for option-selection tasks because it
  can extract the last numbered option explanation instead of the final digit set.
- Some valid answers appear in equivalent math forms, for example
  `a*sqrt(3)/3`, `a/sqrt(3)`, or LaTeX fractions.

Current mitigations:

- Recoverable provider failures degrade instead of crashing.
- Solver JSON repair uses verifier model as formatter (not the solver itself).
- Option-style prose prefers remote repair before local salvage.
- Parser option arbiter only runs for detected option-selection tasks with model disagreement.
- The final renderer enforces answer-only formatting.
- `"sort": "throughput"` reduces 429/timeout errors.
- Response healing plugin eliminates most JSON defects before repair is needed.

## What Not To Do
- Do not add hardcoded answers for the user's screenshots.
- Do not add a broad deterministic "exact engine" unless the user explicitly
  changes the product direction.
- Do not reintroduce `app/api_client.py` as the active runtime.
- Do not reintroduce active `OPENROUTER_MODEL`, `OPENROUTER_MODELS`,
  `SYSTEM_PROMPT_1`, or `SYSTEM_PROMPT_2`.
- Do not loosen UI output to include explanations.
- Do not call Tkinter methods directly from hotkey listener threads.
- Do not break the small hidden X11 window behavior.
- Do not pass images to the solver model (DeepSeek V4 Flash is text-only).

## Test Commands
Local checks:

```bash
python3 -m py_compile main.py app/*.py ui/*.py tests/*.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

Live model regression suite:

```bash
python3 tests/run_solver_fixture_suite.py user-regression
```

The live suite requires network and a valid OpenRouter key. It creates
screenshot-like image fixtures and sends image-only blocks to the real
`GeometryPhotoSolver`, matching the model path more closely than text-only unit
tests. It is not a full GUI test.

Expected answers in `user-regression`:

- `101` -> `16`
- `102` -> `25`
- `103` -> `23`
- `104` -> `124`
- `105` -> `120/17`
- `106` -> `a/sqrt(3)` or equivalent
- `107` -> `69`
- `108` -> `80`
- `109` -> `0.8`
- `110` -> `6`

Manual GUI checks after UI changes:

- startup via `bash run.sh`
- `Ctrl+Space` toggle
- `Ctrl+Shift+Space` show
- `Ctrl+Alt+Space` clipboard solve
- `Ctrl+Shift+S` screenshot insertion
- paste one or several images
- close through `×`
- answer-only output

## Release and Secrets
`.env` is tracked in git. It may contain a local test key encoded in base64.
Release branches must sanitize `.env` so:

```text
OPENROUTER_API_KEY=
```

Do not put raw API keys in docs. There is no `.env.example` — users on a clean
clone insert their own key directly into `.env`.

## Operational Notes
`run.sh` installs dependencies from `packages/` using offline mode. Any new
dependency must include a matching artifact under `packages/`.

The target environment is Linux/RosaOS with X11. Keep Python 3.8 compatibility.

The UI is intentionally small and hidden by default. Preserve:

- global hotkeys
- small window placement/focus behavior
- close via `×`
- screenshot flow
- image paste/insertion labels
- answer-only output
