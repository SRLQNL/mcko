# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the entrypoint for a small X11/Tkinter desktop app that solves text and image math tasks through OpenRouter. The active runtime path is centered on `app/geometry_solver.py`: `Qwen` parses, `Kimi` solves, `Llama` verifies. Core support modules live in `app/` (`config.py`, `session.py`, `hotkeys.py`, `scenario2.py`, `screenshot.py`, `clipboard.py`, `logger.py`, `restart.py`). Tkinter UI code lives in `ui/`. Root scripts are operational: `run.sh` installs deps and starts the app, `watchdog.sh` restarts `main.py`, and `kill.sh` stops running processes. Offline Python artifacts are committed under `packages/`.

## Build, Test, and Development Commands
Use the repo root for all commands.

- `bash run.sh`: installs system deps, installs Python packages from `packages/`, creates `.env` from `.env.example`, and launches the app in the background.
- `bash kill.sh`: stops the watchdog and app processes.
- `python3 main.py`: runs the app directly when dependencies are already installed.
- `python3 -m py_compile main.py app/*.py ui/*.py`: quick syntax check for touched files.

Target environment is RosaOS/Linux with offline dependency installation. `run.sh` uses `--no-index --find-links=./packages`, so any dependency change must include the matching wheel or sdist in `packages/`.

## Coding Style & Naming Conventions
Target Python 3.8 compatibility. Use 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and explicit imports. Prefer 3.8-safe hints (`List`, `Dict`, `Optional`) unless `from __future__ import annotations` is already present. Do not use `match`, `X | Y`, or other 3.9+ conveniences. Logging is mandatory in changed runtime paths; follow the existing `logger.info(...)` / `logger.warning(...)` / `logger.error(...)` pattern.

Keep UI changes minimal and local. Do not regress X11 hotkeys, small window behavior, close-via-`×`, image insertion/paste, screenshot flow, or the current `overrideredirect(True)`/focus timing logic.

## Runtime Expectations
The old single-model `APIClient` path is removed. Do not reintroduce `OPENROUTER_MODEL`, `OPENROUTER_MODELS`, or legacy `SYSTEM_PROMPT_1/2` as active runtime dependencies without a clear migration plan. The current app solves all tasks through the three-model solver:

- `Qwen` = parser / OCR / visual extraction
- `Kimi` = primary solver
- `Llama` = verifier

Internal reasoning may be full, but user-facing output must stay answer-only, typically `1) 69`.

## Testing Guidelines
This project currently relies on manual validation. After changes, run `python3 -m py_compile main.py app/*.py ui/*.py`, then verify the affected flow locally: app startup, X11 hotkeys, small chat window, close button, clipboard/image handling, screenshot flow, and log output in `mcko.log`. For solver changes, test both `text-only` and `image` tasks and watch for malformed JSON, provider `429`, and fallback/degraded runs.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Handle solver timeouts explicitly`, `Harden JSON salvage for solver`, and `Retry parser and verifier on rate limits`. Keep commit messages concise, specific, and action-oriented. PRs should describe the user-visible behavior change, list manual verification steps, mention config or dependency updates, and include screenshots only for UI changes.

## Security & Configuration Tips
Configuration lives in `.env` and `.env.example`. `OPENROUTER_API_KEY` is stored in `.env` as base64 and decoded by `app/config.py`; preserve that behavior. Keep secrets out of docs and scripts. Prefer resilient provider handling: retry transient failures, degrade gracefully on parser/verifier outages, and avoid hard-failing the whole solver on temporary upstream rate limits unless the primary solver is unavailable.
