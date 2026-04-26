from __future__ import annotations

import logging
import re
import time
from typing import Dict, List

import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = (20, 180)
MAX_TOKENS = 4000
MAX_RETRIES = 3
RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504)

SYSTEM_PROMPT = (
    "Reply in plain text without any markdown formatting. "
    "Do not use **, *, _, __, `, ~~, #, ##, ###, code fences, "
    "blockquotes, or bullet/numbered list syntax. "
    "Отвечай обычным текстом без markdown-разметки."
)

FALLBACK_MESSAGE = "Не удалось определить ответ"

_log = logging.getLogger("mcko.geometry_solver")


class GeometryPhotoSolver:
    """Lite single-model solver. Sends content blocks to one configured model."""

    def __init__(self, api_key: str, model: str = ""):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = (model or "").strip()
        self._session = requests.Session()

    def solve_content_blocks(self, content_blocks: List[Dict], multi_model: bool = True) -> str:
        # multi_model retained for back-compat with existing callers; lite runtime ignores it.
        del multi_model
        if not self.model:
            _log.error("MODEL is not configured; set MODEL in .env")
            return FALLBACK_MESSAGE
        if not content_blocks:
            _log.warning("Empty content_blocks")
            return FALLBACK_MESSAGE
        try:
            text = self._call_model(content_blocks)
        except Exception as exc:
            _log.error("Model call failed: %s", exc, exc_info=True)
            return FALLBACK_MESSAGE
        cleaned = self._strip_markdown(text).strip()
        return cleaned or FALLBACK_MESSAGE

    def _call_model(self, content_blocks: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_blocks},
            ],
            "provider": {
                "allow_fallbacks": True,
                "data_collection": "allow",
                "sort": "throughput",
            },
            "plugins": [{"id": "response-healing"}],
        }
        headers = {
            "Authorization": "Bearer %s" % self.api_key,
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.post(
                    ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_error = exc
                _log.warning("Request error attempt %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
                self._sleep(attempt)
                continue
            if resp.status_code in RETRYABLE_STATUSES:
                last_error = "HTTP %d" % resp.status_code
                _log.warning(
                    "Retryable status %d attempt %d/%d", resp.status_code, attempt + 1, MAX_RETRIES
                )
                self._sleep(attempt)
                continue
            if resp.status_code >= 400:
                _log.error("Non-retryable HTTP %d: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            data = resp.json()
            text = self._extract_text(data)
            _log.info("Model %s returned %d chars", self.model, len(text))
            return text
        raise RuntimeError("All %d attempts failed: %s" % (MAX_RETRIES, last_error))

    def _extract_text(self, data: Dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text_val = chunk.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
            return "".join(parts)
        return ""

    def _sleep(self, attempt: int) -> None:
        delay = min(2.0 ** attempt, 8.0)
        time.sleep(delay)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"```[\w-]*\n?", "", text)
        text = text.replace("```", "")
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*\*\*([^\*]+)\*\*\*", r"\1", text)
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^\*\n]+)\*", r"\1", text)
        text = re.sub(r"___([^_]+)___", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
        text = re.sub(r"~~([^~]+)~~", r"\1", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
        return text
