import io
import logging
import sys
from typing import Optional

_log = logging.getLogger("mcko.screenshot")


def take_screenshot() -> Optional[bytes]:
    """Capture the full screen and return PNG bytes, or None on failure."""
    _log.info("Taking screenshot, platform=%s", sys.platform)
    if sys.platform.startswith("linux"):
        return _linux_screenshot()
    elif sys.platform == "darwin":
        return _mac_screenshot()
    else:
        _log.error("Unsupported platform for screenshot: %s", sys.platform)
        return None


MAX_DIMENSION = 1280  # максимальная ширина или высота скриншота


def _resize_if_needed(image):
    """Уменьшает изображение до MAX_DIMENSION по длинной стороне, сохраняя пропорции."""
    from PIL import Image
    w, h = image.size
    if w <= MAX_DIMENSION and h <= MAX_DIMENSION:
        return image
    scale = MAX_DIMENSION / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    _log.info("Screenshot resized: %dx%d → %dx%d", w, h, new_w, new_h)
    return resized


def _linux_screenshot() -> Optional[bytes]:
    try:
        from Xlib import display as xdisplay, X
        from PIL import Image

        d = xdisplay.Display()
        root = d.screen().root
        geom = root.get_geometry()
        _log.debug("Screen geometry: %dx%d", geom.width, geom.height)
        raw = root.get_image(0, 0, geom.width, geom.height, X.ZPixmap, 0xFFFFFFFF)
        image = Image.frombytes(
            "RGB", (geom.width, geom.height), raw.data, "raw", "BGRX"
        )
        d.close()
        image = _resize_if_needed(image)
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        _log.info("Screenshot captured: %dx%d, %d bytes", image.width, image.height, len(png_bytes))
        return png_bytes
    except Exception as exc:
        _log.error("Linux screenshot failed: %s", exc, exc_info=True)
        return None


def _mac_screenshot() -> Optional[bytes]:
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab()
        img = _resize_if_needed(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        _log.info("Screenshot captured: %dx%d, %d bytes", img.width, img.height, len(png_bytes))
        return png_bytes
    except Exception as exc:
        _log.error("Mac screenshot failed: %s", exc, exc_info=True)
        return None
