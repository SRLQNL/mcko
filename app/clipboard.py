import base64
import io
import logging
import subprocess
import sys
from typing import Literal, Optional

_log = logging.getLogger("mcko.clipboard")

ClipboardType = Literal["text", "image", "empty"]


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_mac() -> bool:
    return sys.platform == "darwin"


def detect_type() -> ClipboardType:
    """Return 'text', 'image', or 'empty' depending on clipboard contents."""
    _log.debug("Detecting clipboard type")
    result = _try_read_image()
    if result is not None:
        _log.info("Clipboard type: image, size=%d bytes", len(result))
        return "image"
    text = _try_read_text()
    if text:
        _log.info("Clipboard type: text, size=%d chars", len(text))
        return "text"
    _log.info("Clipboard type: empty")
    return "empty"


def read_text() -> Optional[str]:
    """Return clipboard text or None if not available."""
    text = _try_read_text()
    if text:
        _log.info("read_text: %d chars", len(text))
    else:
        _log.info("read_text: empty")
    return text


def read_image() -> Optional[bytes]:
    """Return clipboard image as PNG bytes or None if not available."""
    data = _try_read_image()
    if data:
        _log.info("read_image: %d bytes", len(data))
    else:
        _log.info("read_image: no image in clipboard")
    return data


def write_text(text: str) -> None:
    """Write text to clipboard."""
    _log.info("write_text: %d chars", len(text))
    if _is_linux():
        _linux_write_text(text)
    elif _is_mac():
        _mac_write_text(text)
    else:
        _log.error("Unsupported platform for clipboard write: %s", sys.platform)


def image_to_base64(image_bytes: bytes) -> str:
    """Convert raw PNG bytes to a base64 data URL for use in API requests."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ─── Internal helpers ───────────────────────────────────────────────────────

def _try_read_text() -> Optional[str]:
    try:
        if _is_linux():
            return _linux_read_text()
        elif _is_mac():
            return _mac_read_text()
    except Exception as exc:
        _log.error("Error reading text from clipboard: %s", exc)
    return None


def _try_read_image() -> Optional[bytes]:
    try:
        if _is_linux():
            return _linux_read_image()
        elif _is_mac():
            return _mac_read_image()
    except Exception as exc:
        _log.error("Error reading image from clipboard: %s", exc)
    return None


# ─── Linux (xclip / xsel) ───────────────────────────────────────────────────

def _linux_read_text() -> Optional[str]:
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        pass
    try:
        result = subprocess.run(
            ["xsel", "--clipboard", "--output"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        _log.warning("Neither xclip nor xsel found; cannot read clipboard text")
    return None


def _linux_read_image() -> Optional[bytes]:
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o", "-t", "image/png"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except FileNotFoundError:
        _log.warning("xclip not found; cannot read clipboard image")
    return None


def _linux_write_text(text: str) -> None:
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            timeout=5,
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        subprocess.run(
            ["xsel", "--clipboard", "--input"],
            input=text.encode("utf-8"),
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        _log.error("Cannot write to clipboard: %s", exc)


# ─── macOS (pbcopy / pbpaste) ───────────────────────────────────────────────

def _mac_read_text() -> Optional[str]:
    result = subprocess.run(["pbpaste"], capture_output=True, timeout=5)
    if result.returncode == 0 and result.stdout:
        return result.stdout.decode("utf-8", errors="replace")
    return None


def _mac_read_image() -> Optional[bytes]:
    # Use PIL on macOS for image clipboard
    try:
        from PIL import ImageGrab
        img = ImageGrab.grabclipboard()
        if img is None:
            return None
        buf = io.BytesIO()
        img.convert("RGBA").save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        _log.error("PIL ImageGrab error: %s", exc)
    return None


def _mac_write_text(text: str) -> None:
    subprocess.run(
        ["pbcopy"],
        input=text.encode("utf-8"),
        timeout=5,
        check=True,
    )
