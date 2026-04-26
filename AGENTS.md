# Repository Guidelines

## Current State
`main` is the active branch. The runtime is the **lite** single-model solver
in `app/geometry_solver.py`. Previous three-model parser/solver/verifier
runtime is removed.

The app is a small X11/Tkinter desktop tool for solving tasks from text,
screenshots, clipboard images, and pasted images through OpenRouter.

A single env var controls the model:

- `MODEL` — any OpenRouter model id (the user sets this in `.env`). Default
  is empty; the solver returns a fallback message until configured.

Do not reintroduce `MODEL_SOLVER`, `MODEL_PARSER`, `MODEL_VERIFIER`,
`OPENROUTER_MODEL`, `OPENROUTER_MODELS`, `SYSTEM_PROMPT_1`, or
`SYSTEM_PROMPT_2` as active runtime dependencies without an explicit
migration.

## Project Structure
Use the repository root for commands.

- `main.py` - application entrypoint, Tk root, hotkey callbacks, UI wiring.
- `app/geometry_solver.py` - lite single-model solver and OpenRouter call.
- `app/config.py` - `.env` loading and base64 API-key decoding.
- `app/scenario2.py` - background clipboard flow.
- `app/hotkeys.py` - global X11 hotkeys via `pynput`.
- `app/clipboard.py` - text/image clipboard helpers.
- `app/screenshot.py` - screenshot capture flow.
- `app/session.py` - in-memory chat history.
- `ui/` - Tkinter window, input, chat view, image labels.
- `tests/test_geometry_solver.py` - unit tests for the lite solver.
- `tests/run_solver_fixture_suite.py` - live model fixture runner.

Operational scripts:

- `bash run.sh` - installs offline dependencies from `packages/`, starts watchdog.
- `bash kill.sh` - stops watchdog/app processes.
- `python3 main.py` - direct local run when dependencies are installed.

## Runtime Contract
All user-facing answers are plain text. Whatever the model returns is
post-processed to strip residual markdown markers (`**`, `*`, `_`, `__`,
`` ` ``, `~~`, headers, blockquotes, code fences).

The system prompt is intentionally minimal: it tells the model to reply in
plain text without markdown formatting (in both English and Russian). There
is no JSON schema, no role description, no examples.

All API requests include:
- `"provider": {"allow_fallbacks": True, "data_collection": "allow", "sort": "throughput"}`
- `"plugins": [{"id": "response-healing"}]`

Retries on 408/429/500/502/503/504 with exponential backoff. Any other
failure returns `Не удалось определить ответ`.

Do not add hardcoded task answers, formula-specific exact solvers, or
multi-model consensus logic.

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
- `MODEL` may be left empty in releases — the user fills it in.
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

That suite requires network, a valid OpenRouter key, and a configured
`MODEL` value in `.env`.

Manual target checks after UI changes:

- app startup through `bash run.sh`
- `Ctrl+Space`, `Ctrl+Shift+Space`, `Ctrl+Alt+Space`, `Ctrl+Shift+S`
- small chat window show/hide
- close via `×`
- screenshot insertion
- clipboard image/text flow
- paste multiple images
- plain-text output

## Git
Commit messages should be short, imperative, and specific, for example:

- `Convert runtime to lite single-model solver`
- `Replace MODEL_SOLVER/PARSER/VERIFIER with single MODEL env var`
- `Strip remaining markdown markers from model output`

Do not amend commits unless explicitly requested. Do not use destructive git
commands unless explicitly requested.
