import logging

from app import clipboard as cb
from app.api_client import APIClient
from app.config import Config

_log = logging.getLogger("mcko.scenario2")


def handle_clipboard_hotkey(config: Config, api_client: APIClient) -> None:
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
        content = text
        messages = [{"role": "user", "content": content}]

    else:  # image
        image_bytes = cb.read_image()
        if not image_bytes:
            _log.warning("read_image returned None, aborting")
            return
        data_url = cb.image_to_base64(image_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Что изображено на картинке?"},
                ],
            }
        ]

    _log.info("Sending clipboard content to API (stream=False)")
    result = api_client.send(
        system_prompt=config.system_prompt_2,
        messages=messages,
        stream=False,
    )

    if isinstance(result, str):
        _log.info("Response received: %d chars", len(result))
        cb.write_text(result)
        _log.info("Response written to clipboard")
    else:
        # Should not happen with stream=False, but guard anyway
        _log.error("Unexpected generator returned for stream=False")
