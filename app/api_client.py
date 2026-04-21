from __future__ import annotations

import json
import logging
from typing import Generator, Union

import requests

from app.logger import logger

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_log = logging.getLogger("mcko.api_client")


class APIClient:
    def __init__(self, api_keys: list[str], model: str):
        self.api_keys = api_keys
        self.model = model
        self._active_key_index = 0
        _log.info("APIClient initialized: model=%s, keys=%d", model, len(api_keys))

    def _build_headers(self, api_key: str) -> dict:
        auth_header = f"Bearer {api_key}"
        _log.info(
            "Building headers: auth_prefix=%s, auth_length=%d",
            auth_header[:15],
            len(auth_header),
        )
        return {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

    def _should_try_next_key(self, response: requests.Response) -> bool:
        return response.status_code in (401, 403, 429)

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

        last_error = ""
        key_count = len(self.api_keys)
        for offset in range(key_count):
            key_index = (self._active_key_index + offset) % key_count
            api_key = self.api_keys[key_index]
            try:
                response = requests.post(
                    ENDPOINT,
                    headers=self._build_headers(api_key),
                    json=payload,
                    stream=stream,
                    timeout=(15, 120),  # (connect, read) — read применяется к каждому чанку стрима
                )
            except requests.RequestException as exc:
                _log.error("Network error on key #%d: %s", key_index + 1, exc)
                return f"[Ошибка сети: {exc}]"

            if response.ok:
                self._active_key_index = key_index
                _log.info("API request succeeded with key #%d", key_index + 1)
                if stream:
                    return self._stream_response(response)
                else:
                    return self._parse_full_response(response)

            body = response.text[:500]
            last_error = f"[Ошибка API {response.status_code}: {body}]"
            _log.error("API error on key #%d: status=%d, body=%s", key_index + 1, response.status_code, body)
            if response.status_code == 401 and "User not found" in body:
                _log.error(
                    "OpenRouter returned 401 User not found. This usually means an invalid/revoked key, "
                    "a stale env var overriding .env, or a temporary OpenRouter auth-side incident."
                )
            if offset < key_count - 1 and self._should_try_next_key(response):
                _log.warning("Trying next API key after status %d on key #%d", response.status_code, key_index + 1)
                continue
            break

        return last_error or "[Ошибка API: неизвестная ошибка]"

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
