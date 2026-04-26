# MCKO Handoff for Claude Code

## Current Runtime
This repository is a small Python 3.8 X11/Tkinter desktop app for solving tasks
from text, screenshots, clipboard images, and pasted images.

The active runtime is **lite**: a single-model solver in
`app/geometry_solver.py`. One configured model handles every request directly.
Do not reintroduce the previous three-model parser/solver/verifier consensus
pipeline without an explicit request.

## Configuration
Single env var controls the model:

- `MODEL` — any OpenRouter model id, for example a top-tier multimodal model.
  The user sets this in `.env`. Default is empty; if unset, the solver returns
  the fallback message and logs a warning.

There are no parser/solver/verifier roles, no JSON schema, no consensus, no
EMS, no option arbiter, no self-check round, no remote JSON repair.

## User-Facing Output
Strict plain text. Whatever the model returns is post-processed to strip any
remaining markdown markers (`**`, `*`, `_`, `__`, `` ` ``, `~~`, headers,
blockquotes, code fences). The prompt also instructs the model to emit plain
text in both English and Russian.

Do not expose model names, tracebacks, or system internals in the UI/clipboard
answer.

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

Avoid task-specific hardcoded answers.

## Key Files
- `main.py` - application startup, Tk root, hotkey callbacks, response worker.
- `app/geometry_solver.py` - lite single-model solver.
- `app/config.py` - `.env` loading, base64 API-key decoding, `MODEL` default.
- `app/scenario2.py` - background clipboard request flow.
- `app/hotkeys.py` - global hotkeys via `pynput`.
- `app/clipboard.py` - clipboard text/image helpers.
- `app/screenshot.py` - screenshot capture.
- `ui/window.py` - small chat window and hidden-window image insertion.
- `ui/input_field.py` - text input and image labels.
- `ui/chat_view.py` - assistant placeholder and answer rendering.
- `tests/test_geometry_solver.py` - unit tests for the lite solver.
- `tests/run_solver_fixture_suite.py` - live model regression runner.

## Solver Pipeline
`GeometryPhotoSolver.solve_content_blocks(content_blocks)`:

1. Validate `MODEL` is set and `content_blocks` is non-empty.
2. Build a single chat-completions payload:
   - `system`: minimal instruction to return plain text without markdown.
   - `user`: the raw content blocks (text + image_url) the caller passed in.
3. POST to `https://openrouter.ai/api/v1/chat/completions` with retries on
   408/429/500/502/503/504.
4. Extract the assistant text, run `_strip_markdown`, return the result.
5. On any failure return `Не удалось определить ответ`.

All API requests use:
- `"provider": {"allow_fallbacks": True, "data_collection": "allow", "sort": "throughput"}`
- `"plugins": [{"id": "response-healing"}]`

## What Not To Do
- Do not add hardcoded answers for the user's screenshots.
- Do not reintroduce parser/solver/verifier roles or consensus logic.
- Do not reintroduce `MODEL_SOLVER`, `MODEL_PARSER`, `MODEL_VERIFIER`,
  `OPENROUTER_MODEL`, `OPENROUTER_MODELS`, `SYSTEM_PROMPT_1`, or
  `SYSTEM_PROMPT_2` as active runtime dependencies.
- Do not reintroduce `app/api_client.py` as the active runtime.
- Do not add elaborate system prompts. The lite runtime keeps the prompt
  minimal: "plain text without markdown".
- Do not call Tkinter methods directly from hotkey listener threads.
- Do not break the small hidden X11 window behavior.

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

The live suite requires network and a valid OpenRouter key plus a `MODEL`
value in `.env`. It sends image-only blocks to `GeometryPhotoSolver`.

Manual GUI checks after UI changes:

- startup via `bash run.sh`
- `Ctrl+Space` toggle
- `Ctrl+Shift+Space` show
- `Ctrl+Alt+Space` clipboard solve
- `Ctrl+Shift+S` screenshot insertion
- paste one or several images
- close through `×`
- plain-text output

## Release and Secrets
`.env` is tracked in git. It may contain a local test key encoded in base64.
Release branches must sanitize `.env` so:

```text
OPENROUTER_API_KEY=
```

`MODEL` may also be left empty in releases — the user fills it in. Do not put
raw API keys in docs. There is no `.env.example` — users on a clean clone edit
`.env` directly.

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
- plain-text output
