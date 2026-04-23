from __future__ import annotations

import json
import logging
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 2048
RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504)

_log = logging.getLogger("mcko.api_client")


class APIClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model.strip()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            }
        )
        _log.info("APIClient initialized: model=%s", self.model)

    def send(
        self,
        system_prompt: str,
        messages: List[Dict],
        stream: bool = True,
    ) -> Union[str, Generator[str, None, None]]:
        if not self.model:
            _log.error("No OpenRouter model configured")
            return "[Ошибка конфигурации: не задана модель OpenRouter]"

        has_images = any(isinstance(message.get("content"), list) for message in messages)
        full_messages = list(messages)
        if system_prompt.strip():
            full_messages = [{"role": "system", "content": system_prompt}] + full_messages
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": stream,
            "max_tokens": MAX_TOKENS,
        }

        _log.info(
            "Sending OpenRouter request: model=%s stream=%s messages=%d images=%s",
            self.model,
            stream,
            len(messages),
            has_images,
        )

        last_error = "[Ошибка API: неизвестная ошибка]"
        for attempt in range(1, 4):
            response, network_error, elapsed_ms = self._send_once(payload, stream)

            if network_error is not None:
                _log.warning(
                    "Network error: model=%s attempt=%d elapsed_ms=%d error=%s",
                    self.model,
                    attempt,
                    elapsed_ms,
                    network_error,
                )
                last_error = "[Ошибка сети: %s]" % network_error
                if attempt < 3:
                    self._sleep_before_retry(attempt)
                    continue
                return last_error

            if response is None:
                return last_error

            if response.ok:
                _log.info(
                    "OpenRouter request succeeded: model=%s attempt=%d elapsed_ms=%d",
                    self.model,
                    attempt,
                    elapsed_ms,
                )
                if stream:
                    return self._stream_response(response)
                return self._parse_full_response(response)

            body = response.text[:500]
            request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "-"
            _log.error(
                "API error: model=%s attempt=%d status=%d request_id=%s elapsed_ms=%d body=%s",
                self.model,
                attempt,
                response.status_code,
                request_id,
                elapsed_ms,
                body,
            )
            last_error = "[Ошибка API %d: %s]" % (response.status_code, body)

            if response.status_code in RETRYABLE_STATUSES and attempt < 3:
                self._sleep_before_retry(attempt)
                continue
            return last_error

        return last_error

    def _parse_full_response(self, response: requests.Response) -> str:
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            text = message.get("content")
            if text is None:
                text = message.get("reasoning") or "[Пустой ответ модели]"
            _log.info("Response received: %d chars", len(text))
            return text
        except (KeyError, IndexError, ValueError) as exc:
            _log.error("Failed to parse response: %s, body=%s", exc, response.text[:200])
            return "[Ошибка разбора ответа: %s]" % exc

    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        total_chars = 0
        saw_content = False
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
                except ValueError as exc:
                    _log.debug("Skipping malformed SSE chunk: %s", exc)
                    continue
                error = chunk.get("error")
                if error:
                    message = error.get("message", "unknown stream error")
                    _log.error("Stream error event: model=%s error=%s", self.model, message)
                    yield "\n[Ошибка потока: %s]" % message
                    break
                try:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, AttributeError):
                    delta = ""
                if delta:
                    saw_content = True
                    total_chars += len(delta)
                    yield delta
        except requests.RequestException as exc:
            _log.error("Stream error: %s", exc)
            yield "\n[Ошибка потока: %s]" % exc
        finally:
            _log.info(
                "Stream finished: model=%s chars=%d saw_content=%s",
                self.model,
                total_chars,
                saw_content,
            )

    def _send_once(self, payload: Dict, stream: bool) -> Tuple[Optional[requests.Response], Optional[str], int]:
        started_at = time.time()
        try:
            response = self._session.post(
                ENDPOINT,
                json=payload,
                stream=stream,
                timeout=(15, 180),
            )
            elapsed_ms = int((time.time() - started_at) * 1000)
            return response, None, elapsed_ms
        except requests.RequestException as exc:
            elapsed_ms = int((time.time() - started_at) * 1000)
            return None, str(exc), elapsed_ms

    def _sleep_before_retry(self, retry_count: int) -> None:
        delay = 0.5 * retry_count
        _log.info("Retrying OpenRouter request after %.1fs", delay)
        time.sleep(delay)
