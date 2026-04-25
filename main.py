import signal
import sys
import threading
import time

from app.logger import logger
from app.config import Config
from app.geometry_solver import GeometryPhotoSolver
from app.session import Session
from app.hotkeys import HotkeyManager
from app.scenario2 import handle_clipboard_hotkey


def main():
    logger.info("MCKO starting up")

    # Ignore SIGHUP so closing the terminal doesn't kill the process
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        logger.info("SIGHUP ignored — terminal close will not stop MCKO")
    except (AttributeError, OSError):
        pass  # SIGHUP not available on Windows

    # ── Config ──────────────────────────────────────────────────────────────
    config = Config()
    config.load()

    # ── Core components ─────────────────────────────────────────────────────
    geometry_solver = GeometryPhotoSolver(
        config.api_key,
        kimi_model=config.model_solver,
        qwen_model=config.model_parser,
        llama_model=config.model_verifier,
    )
    session = Session()
    logger.info("Core components initialized")

    # ── tkinter root (hidden) ────────────────────────────────────────────────
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    # Prevent root from appearing in taskbar
    try:
        root.wm_overrideredirect(True)
    except Exception as exc:
        logger.warning("Could not set overrideredirect: %s", exc)

    # ── UI ───────────────────────────────────────────────────────────────────
    from ui.window import ChatWindow

    pending_window_toggle = {"job": None}
    screenshot_hotkey_state = {"last_trigger_at": 0.0}
    screenshot_hotkey_suppress_seconds = 1.0
    solver_mode = {"multi_model": True}

    def _should_suppress_window_hotkey() -> bool:
        elapsed = time.monotonic() - screenshot_hotkey_state["last_trigger_at"]
        if elapsed < screenshot_hotkey_suppress_seconds:
            logger.info(
                "Suppressing window/show hotkey because screenshot hotkey fired %.3fs ago",
                elapsed,
            )
            return True
        return False

    def _is_api_error_text(text: str) -> bool:
        return text.startswith("[Ошибка ") or text.startswith("\n[Ошибка ")

    def on_send(content_blocks: list) -> None:
        """Called when user submits a message from the chat window."""
        logger.info("User submitted message: %d blocks", len(content_blocks))
        has_images = any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in content_blocks
        )

        # Determine display text for chat view
        text_parts = [
            b["text"] for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        display_text = " ".join(text_parts) if text_parts else "[изображение]"

        chat_window.chat_view.append_user(display_text)

        session.add_user(content_blocks)

        chat_window.chat_view.begin_assistant()
        # Disable input while waiting for response
        chat_window._input_field.configure(state="disabled")

        def _stream_in_thread():
            logger.info("Starting response worker: has_images=%s", has_images)
            full_text = ""
            had_stream_error = False
            try:
                result = geometry_solver.solve_content_blocks(
                    content_blocks, multi_model=solver_mode["multi_model"]
                )
                if isinstance(result, str):
                    root.after(0, lambda t=result: chat_window.chat_view.append_assistant_chunk(t))
                    if result.startswith("[Ошибка ") or result.startswith("\n[Ошибка "):
                        had_stream_error = True
                    else:
                        full_text = result
                else:
                    for chunk in result:
                        if _is_api_error_text(chunk):
                            had_stream_error = True
                            root.after(0, lambda t=chunk: chat_window.chat_view.append_assistant_chunk(t))
                            continue
                        full_text += chunk
                        root.after(0, lambda t=chunk: chat_window.chat_view.append_assistant_chunk(t))
            except Exception as exc:
                had_stream_error = True
                logger.error("Response worker failed: %s", exc, exc_info=True)
                error_text = "1) Не удалось определить ответ"
                root.after(0, lambda t=error_text: chat_window.chat_view.append_assistant_chunk(t))
            finally:
                root.after(0, chat_window.chat_view.end_assistant)
                if full_text:
                    session.add_assistant(full_text)
                    logger.info(
                        "Streaming response complete: %d chars (stream_error=%s)",
                        len(full_text),
                        had_stream_error,
                    )
                else:
                    logger.warning("Assistant response was not added to session due to API/stream error")

                def _reenable():
                    chat_window._input_field.configure(state="normal")
                    chat_window._input_field.focus()

                root.after(0, _reenable)

        t = threading.Thread(target=_stream_in_thread, daemon=True, name="stream-response")
        t.start()

    chat_window = ChatWindow(
        root,
        on_send_callback=on_send,
        on_mode_change=lambda v: solver_mode.update({"multi_model": v}),
    )
    logger.info("ChatWindow created")

    # ── Hotkeys ──────────────────────────────────────────────────────────────
    def on_window_hotkey():
        if _should_suppress_window_hotkey():
            return
        logger.info("Window hotkey dispatched to main thread (toggle)")
        def _run_toggle():
            logger.info("Executing delayed window toggle")
            pending_window_toggle["job"] = None
            chat_window.toggle()

        existing_job = pending_window_toggle["job"]
        if existing_job is not None:
            try:
                root.after_cancel(existing_job)
                logger.info("Cancelled previous pending window toggle")
            except Exception as exc:
                logger.warning("Failed to cancel pending window toggle: %s", exc)
        pending_window_toggle["job"] = root.after(120, _run_toggle)
        logger.info("Scheduled window toggle in 120ms")

    def on_show_hotkey():
        if _should_suppress_window_hotkey():
            return
        logger.info("Show hotkey dispatched to main thread")
        def _run_show():
            # Отменяем pending toggle только если окно СКРЫТО:
            # тогда SHOW сам его покажет, toggle лишний.
            # Если окно ВИДИМО — toggle должен выполниться и скрыть его.
            if not chat_window._visible:
                existing_job = pending_window_toggle["job"]
                if existing_job is not None:
                    try:
                        root.after_cancel(existing_job)
                        logger.info("Cancelled pending window toggle because SHOW hotkey fired (window was hidden)")
                    except Exception as exc:
                        logger.warning("Failed to cancel pending toggle on SHOW hotkey: %s", exc)
                    pending_window_toggle["job"] = None
            else:
                logger.info("SHOW hotkey fired but window is visible — letting pending toggle proceed")
            chat_window.show()

        root.after(0, _run_show)

    def on_clipboard_hotkey():
        logger.info("Clipboard hotkey triggered")
        handle_clipboard_hotkey(geometry_solver)

    def on_screenshot_hotkey():
        logger.info("Screenshot hotkey triggered")
        screenshot_hotkey_state["last_trigger_at"] = time.monotonic()
        from app.screenshot import take_screenshot
        png_bytes = take_screenshot()
        if png_bytes:
            logger.info("Screenshot ready (%d bytes), inserting into input field", len(png_bytes))
            root.after(0, lambda b=png_bytes: chat_window.insert_image_bytes(b))
        else:
            logger.error("Screenshot failed, nothing to insert")

    hotkeys = HotkeyManager(
        hotkey_window=config.hotkey_window,
        hotkey_show=config.hotkey_show,
        hotkey_clipboard=config.hotkey_clipboard,
        hotkey_screenshot=config.hotkey_screenshot,
        on_window=on_window_hotkey,
        on_show=on_show_hotkey,
        on_clipboard=on_clipboard_hotkey,
        on_screenshot=on_screenshot_hotkey,
    )
    hotkeys.start()
    logger.info("HotkeyManager started")

    # ── Main loop ────────────────────────────────────────────────────────────
    logger.info("All components initialized, entering main loop")
    try:
        root.mainloop()
    finally:
        hotkeys.stop()
        logger.info("MCKO shut down")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
