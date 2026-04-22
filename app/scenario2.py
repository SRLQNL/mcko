import logging

from app import clipboard as cb
from app.api_client import APIClient
from app.config import Config
from app.geometry_solver import GeometryPhotoSolver

_log = logging.getLogger("mcko.scenario2")


def handle_clipboard_hotkey(config: Config, api_client: APIClient, geometry_solver: GeometryPhotoSolver) -> None:
    """Read clipboard, send to AI, write response back to clipboard.

    This function is meant to be called in a background thread.
    """
    _log.info("Clipboard hotkey handler started")

    clip_type = cb.detect_type()
    _log.info("Clipboard content type: %s", clip_type)

    if clip_type == "empty":
        _log.warning("Clipboard is empty, nothing to process")
        return

    if clip_type == "text":
        text = cb.read_text()
        if not text:
            _log.warning("read_text returned empty, aborting")
            return
        content_blocks = [{"type": "text", "text": text}]

    else:  # image
        image_bytes = cb.read_image()
        if not image_bytes:
            _log.warning("read_image returned None, aborting")
            return
        data_url = cb.image_to_base64(image_bytes)
        content_blocks = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Реши задачу по изображению."},
        ]

    _log.info("Sending clipboard content to geometry solver")
    result = geometry_solver.solve_content_blocks(content_blocks)
    cb.write_text(result)
    _log.info("Geometry solver response written to clipboard")
