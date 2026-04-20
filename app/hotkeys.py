import logging
import threading
from typing import Callable, Optional

from pynput import keyboard

_log = logging.getLogger("mcko.hotkeys")


class HotkeyManager:
    """Manages global hotkeys using pynput.GlobalHotKeys.

    All tkinter interactions in callbacks must be scheduled via root.after() —
    direct calls from pynput threads are not thread-safe.
    """

    def __init__(
        self,
        hotkey_window: str,
        hotkey_show: str,
        hotkey_clipboard: str,
        hotkey_screenshot: str,
        on_window: Callable[[], None],
        on_show: Callable[[], None],
        on_clipboard: Callable[[], None],
        on_screenshot: Callable[[], None],
    ):
        self._hotkey_window = hotkey_window
        self._hotkey_show = hotkey_show
        self._hotkey_clipboard = hotkey_clipboard
        self._hotkey_screenshot = hotkey_screenshot
        self._on_window = on_window
        self._on_show = on_show
        self._on_clipboard = on_clipboard
        self._on_screenshot = on_screenshot
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self._thread: Optional[threading.Thread] = None

        _log.info(
            "HotkeyManager initialized: window=%s, show=%s, clipboard=%s, screenshot=%s",
            hotkey_window,
            hotkey_show,
            hotkey_clipboard,
            hotkey_screenshot,
        )

    def start(self) -> None:
        """Start the hotkey listener in a daemon thread."""
        hotkeys = {
            self._hotkey_window: self._handle_window,
            self._hotkey_show: self._handle_show,
            self._hotkey_clipboard: self._handle_clipboard,
            self._hotkey_screenshot: self._handle_screenshot,
        }
        _log.info("Registering hotkeys: %s", list(hotkeys.keys()))
        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._thread = threading.Thread(
            target=self._listener.run, daemon=True, name="hotkey-listener"
        )
        self._thread.start()
        _log.info("Hotkey listener started (thread: %s)", self._thread.name)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener is not None:
            _log.info("Stopping hotkey listener")
            self._listener.stop()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        _log.info("Hotkey listener stopped")

    def _handle_window(self) -> None:
        _log.info("Hotkey fired: WINDOW (toggle) — %s", self._hotkey_window)
        try:
            self._on_window()
        except Exception as exc:
            _log.error("Error in on_window callback: %s", exc)

    def _handle_show(self) -> None:
        _log.info("Hotkey fired: SHOW — %s", self._hotkey_show)
        try:
            self._on_show()
        except Exception as exc:
            _log.error("Error in on_show callback: %s", exc)

    def _handle_clipboard(self) -> None:
        _log.info("Hotkey fired: CLIPBOARD — %s", self._hotkey_clipboard)
        try:
            t = threading.Thread(
                target=self._on_clipboard, daemon=True, name="clipboard-handler"
            )
            t.start()
        except Exception as exc:
            _log.error("Error starting clipboard handler thread: %s", exc)

    def _handle_screenshot(self) -> None:
        _log.info("Hotkey fired: SCREENSHOT — %s", self._hotkey_screenshot)
        try:
            t = threading.Thread(
                target=self._on_screenshot, daemon=True, name="screenshot-handler"
            )
            t.start()
        except Exception as exc:
            _log.error("Error starting screenshot handler thread: %s", exc)
