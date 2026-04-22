# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the entrypoint for a small X11/Tkinter desktop app that sends text and image prompts to OpenRouter. The main user flow is taking or pasting photos of math problems and getting a short answer back. Core runtime logic lives in `app/` (`api_client.py`, `config.py`, `session.py`, `hotkeys.py`, `scenario2.py`, `screenshot.py`). Tkinter UI code lives in `ui/`. Root scripts are operational: `run.sh` installs deps and starts the app, `watchdog.sh` restarts `main.py`, and `kill.sh` stops running processes. Offline Python artifacts are committed under `packages/`.

## Build, Test, and Development Commands
Use the repo root for all commands.

- `bash run.sh`: installs system deps, installs Python packages from `packages/`, creates `.env` from `.env.example`, and launches the app in the background.
- `bash kill.sh`: stops the watchdog and app processes.
- `python3 main.py`: runs the app directly when dependencies are already installed.
- `python3 -m py_compile main.py app/*.py ui/*.py`: quick syntax check for touched files.

Target environment is RosaOS/Linux with offline dependency installation. `run.sh` uses `--no-index --find-links=./packages`, so any dependency change must include the matching wheel or sdist in `packages/`. There is no Makefile or formal test runner.

## Coding Style & Naming Conventions
Target Python 3.8 compatibility. Use 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and explicit imports. Prefer 3.8-safe hints (`List`, `Dict`, `Union`) unless `from __future__ import annotations` is already present. Do not use `match`, `X | Y`, or other 3.9+ conveniences. Logging is mandatory in changed runtime paths; follow the existing `logger.info(...)` / `logger.error(...)` pattern.

Keep UI changes minimal and local. Do not regress X11 hotkeys, small window behavior, close-via-`×`, image insertion/paste, or screenshot flow. Preserve the current `overrideredirect(True)` and hotkey timing logic unless the task explicitly targets them.

## Testing Guidelines
This project currently relies on manual validation. After changes, run `python3 -m py_compile main.py app/*.py ui/*.py`, then verify the affected flow locally: app startup, X11 hotkeys, small chat window, close button, clipboard/image handling, and log output in `mcko.log`. If you add automated tests, place them in `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects, for example `Preserve external env over tracked config` and `Sanitize tracked config defaults`. Keep commit messages concise, specific, and action-oriented. PRs should describe the user-visible behavior change, list manual verification steps, mention config or dependency updates, and include screenshots only for UI changes.

## Security & Configuration Tips
Configuration lives in `.env` and `.env.example`. `OPENROUTER_API_KEY` is stored in `.env` as base64 and decoded by `app/config.py`; preserve that behavior. Keep secrets out of docs and scripts. When changing models or provider logic, prefer additive changes such as fallbacks over breaking the current single-model path.
