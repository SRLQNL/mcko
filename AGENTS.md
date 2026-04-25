# Repository Guidelines

## Current State
`main` is the active branch. The runtime is the three-model solver in
`app/geometry_solver.py`. The old single-model `APIClient` runtime is removed.

The app is a small X11/Tkinter desktop tool for solving tasks from text,
screenshots, clipboard images, and pasted images through OpenRouter.

Active model roles (env var → default):

- `MODEL_PARSER=qwen/qwen3-vl-32b-instruct` — parser, OCR, visual extraction,
  and option arbiter for hard multiple-choice image tasks.
- `MODEL_SOLVER=deepseek/deepseek-v4-flash` — primary text solver. Receives
  parser OCR extract only; does NOT receive images (text-only model).
- `MODEL_VERIFIER=meta-llama/llama-4-maverick` — independent verifier and JSON
  repair model. Receives images and OCR; solves independently of solver.

Do not reintroduce `OPENROUTER_MODEL`, `OPENROUTER_MODELS`, `SYSTEM_PROMPT_1`,
or `SYSTEM_PROMPT_2` as active runtime dependencies without an explicit migration.

## Project Structure
Use the repository root for commands.

- `main.py` - application entrypoint, Tk root, hotkey callbacks, UI wiring.
- `app/geometry_solver.py` - active solver pipeline and OpenRouter calls.
- `app/config.py` - `.env` loading and base64 API-key decoding.
- `app/scenario2.py` - background clipboard flow.
- `app/hotkeys.py` - global X11 hotkeys via `pynput`.
- `app/clipboard.py` - text/image clipboard helpers.
- `app/screenshot.py` - screenshot capture flow.
- `app/session.py` - in-memory chat history.
- `ui/` - Tkinter window, input, chat view, image labels.
- `tests/test_geometry_solver.py` - unit/regression tests for solver selection and payload normalization.
- `tests/run_solver_fixture_suite.py` - live model fixture runner. It creates screenshot-like image fixtures and sends image-only blocks to `GeometryPhotoSolver`.

Operational scripts:

- `bash run.sh` - installs offline dependencies from `packages/`, starts watchdog.
- `bash kill.sh` - stops watchdog/app processes.
- `python3 main.py` - direct local run when dependencies are installed.

## Runtime Contract
All user-facing answers must be answer-only, usually:

```text
1) 69
```

For multiple independent tasks:

```text
1) ...
2) ...
```

Internal reasoning can be full, but it must not be displayed in the UI or
clipboard result.

Text-only requests use the solver fast path. Image or image+text requests use
the three-model pipeline:

1. `Parser` (Qwen) OCRs and extracts task structure from source images.
2. `Solver` (DeepSeek) solves from parser OCR extract (text only, no images).
3. `Verifier` (Llama) solves independently from parser OCR plus source images.
   EMS: skipped if solver confidence ≥ 0.95 on non-option tasks.
4. The consensus stage compares answers, confidence, parser ambiguity, repair
   source, and verifier independence.
5. For hard option-selection tasks, parser may run a focused option arbiter pass.

All API requests include:
- `"provider": {"allow_fallbacks": True, "data_collection": "allow", "sort": "throughput"}`
- `"plugins": [{"id": "response-healing"}]`

Do not add hardcoded task answers or formula-specific exact solvers.

## Important Solver Risks
The biggest live failure modes seen so far:

- OpenRouter/Parser `429` upstream rate limits.
- Solver returning prose or malformed JSON instead of strict JSON.
- Local answer salvage extracting a numbered option explanation instead of the final answer.
- Verifier confidently verifying a wrong option subset such as `14` instead of `124`.
- Low-confidence or missing parser output causing `1) Не удалось определить ответ`.

Current mitigations:

- Recoverable provider errors degrade instead of crashing the worker thread.
- Response healing plugin reduces server-side JSON defects 80-99.8%.
- JSON repair for solver uses verifier model as formatter, not the solver itself.
- Option-style prose prefers remote JSON repair before local salvage.
- Parser option arbiter is limited to detected "select/write option numbers" tasks with model disagreement.
- `"sort": "throughput"` reduces timeout and rate-limit errors.
- Top-level solver errors return `1) Не удалось определить ответ` instead of raw exceptions.

## Python and Style
Target Python 3.8 on Linux/RosaOS.

- Use 4-space indentation.
- Use `snake_case` for functions/modules and `PascalCase` for classes.
- Prefer `List`, `Dict`, `Optional` style hints where needed.
- Do not use `match`, `X | Y`, or other Python 3.9+ syntax.
- Keep UI edits minimal and local.
- Preserve X11 hotkeys, small window behavior, `×` close hiding, screenshot flow, paste/image labels, and `overrideredirect(True)` focus timing.
- Log changed runtime paths with existing `logger.info`, `logger.warning`, `logger.error` patterns.

## Editing Rules
Do not revert unrelated user changes.

`AGENTS.md` and `CLAUDE.md` are context files. Runtime code must not depend on them.

Configuration lives in `.env` (tracked in git).

- `.env` may contain a local test key encoded in base64.
- Release branches must sanitize `.env` so `OPENROUTER_API_KEY=` is empty.
- There is no `.env.example`. Users on a clean clone insert their key directly.
- `app/config.py` auto-decodes base64 `OPENROUTER_API_KEY` values.

## Testing
Always run after code changes:

```bash
python3 -m py_compile main.py app/*.py ui/*.py tests/*.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

Live solver regression suite:

```bash
python3 tests/run_solver_fixture_suite.py user-regression
```

That suite requires network and a valid OpenRouter key. It sends image-only
blocks to the same `GeometryPhotoSolver` path used by GUI/clipboard flows, but
it is still not a full X11 GUI test.

Manual target checks after UI changes:

- app startup through `bash run.sh`
- `Ctrl+Space`, `Ctrl+Shift+Space`, `Ctrl+Alt+Space`, `Ctrl+Shift+S`
- small chat window show/hide
- close via `×`
- screenshot insertion
- clipboard image/text flow
- paste multiple images
- answer-only output

## Git
Commit messages should be short, imperative, and specific, for example:

- `Implement research findings: response healing, EMS, V4 Flash`
- `Rename model roles: kimi/qwen/llama → solver/parser/verifier`
- `Fix solver 404: remove images from text-only model request`

Do not amend commits unless explicitly requested. Do not use destructive git
commands unless explicitly requested.
