from __future__ import annotations

import json
import logging
from typing import Generator, Union

import requests

from app.logger import logger

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_log = logging.getLogger("mcko.api_client")


class APIClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        _log.info("APIClient initialized: model=%s", model)

    def _build_headers(self) -> dict:
        auth_header = f"Bearer {self.api_key}"
        _log.info(
            "Building headers: auth_prefix=%s, auth_length=%d",
            auth_header[:15],
            len(auth_header),
        )
        return {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

    def send(
        self,
        system_prompt: str,
        messages: list[dict],
        stream: bool = True,
    ) -> Union[str, Generator[str, None, None]]:
        """Send request to OpenRouter API.

        Args:
            system_prompt: System prompt prepended to every request.
            messages: List of {role, content} dicts (chat history).
            stream: If True return a generator of text chunks; if False return full text.

        Returns:
            str (stream=False) or Generator[str] (stream=True).
            On error returns a string describing the error.
        """
        has_images = any(
            isinstance(m.get("content"), list) for m in messages
        )
        _log.info(
            "Sending request: model=%s, messages=%d, images=%s, stream=%s",
            self.model,
            len(messages),
            has_images,
            stream,
        )

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": stream,
        }

        try:
            response = requests.post(
                ENDPOINT,
                headers=self._build_headers(),
                json=payload,
                stream=stream,
                timeout=(15, 120),  # (connect, read) — read применяется к каждому чанку стрима
            )
        except requests.RequestException as exc:
            _log.error("Network error: %s", exc)
            return f"[Ошибка сети: {exc}]"

        if not response.ok:
            body = response.text[:500]
            request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "-"
            _log.error(
                "API error: model=%s status=%d request_id=%s body=%s",
                self.model,
                response.status_code,
                request_id,
                body,
            )
            if response.status_code == 401 and "User not found" in body:
                _log.error(
                    "OpenRouter returned 401 User not found. This usually means an invalid/revoked key, "
                    "a stale env var overriding .env, or a temporary OpenRouter auth-side incident."
                )
            return f"[Ошибка API {response.status_code}: {body}]"

        if stream:
            return self._stream_response(response)
        else:
            return self._parse_full_response(response)

    def _parse_full_response(self, response: requests.Response) -> str:
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            _log.info("Response received: %d chars", len(text))
            return text
        except (KeyError, IndexError, ValueError) as exc:
            _log.error("Failed to parse response: %s, body=%s", exc, response.text[:200])
            return f"[Ошибка разбора ответа: {exc}]"

    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        _log.debug("Starting SSE stream")
        total_chars = 0
        try:
            for line in response.iter_lines(chunk_size=512):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        total_chars += len(delta)
                        yield delta
                except (KeyError, IndexError, ValueError) as exc:
                    _log.debug("Skipping malformed SSE chunk: %s", exc)
        except requests.RequestException as exc:
            _log.error("Stream error: %s", exc)
            yield f"\n[Ошибка потока: {exc}]"
        finally:
            _log.info("Stream finished: %d chars total", total_chars)
