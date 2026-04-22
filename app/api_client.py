from __future__ import annotations

import json
import logging
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

import requests

from app.logger import logger

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 4096
RETRYABLE_STATUSES = (408, 429, 502, 503, 504)

_log = logging.getLogger("mcko.api_client")


class APIClient:
    def __init__(self, api_key: str, models: List[str]):
        self.api_key = api_key
        self.models = [model for model in models if model]
        _log.info("APIClient initialized: models=%s", ",".join(self.models))

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
        messages: List[Dict],
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
        has_images = any(isinstance(m.get("content"), list) for m in messages)
        if not self.models:
            _log.error("No OpenRouter models configured")
            return "[Ошибка конфигурации: не задана ни одна модель OpenRouter]"

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        last_error = "[Ошибка API: неизвестная ошибка]"

        for model_index, model in enumerate(self.models, 1):
            attempt_ctx = "model=%s attempt=%d/%d stream=%s messages=%d images=%s" % (
                model,
                model_index,
                len(self.models),
                stream,
                len(messages),
                has_images,
            )
            _log.info("Trying model: %s", attempt_ctx)

            payload = {
                "model": model,
                "messages": full_messages,
                "stream": stream,
                "max_tokens": MAX_TOKENS,
                "provider": {
                    "allow_fallbacks": True,
                },
            }

            retry_count = 0
            while retry_count <= 1:
                response, network_error, elapsed_ms = self._send_once(payload, stream)
                retry_index = retry_count + 1

                if network_error is not None:
                    _log.warning(
                        "Network error before response: %s retry=%d elapsed_ms=%d error=%s",
                        attempt_ctx,
                        retry_index,
                        elapsed_ms,
                        network_error,
                    )
                    last_error = "[Ошибка сети: %s]" % network_error
                    if retry_count == 0:
                        retry_count += 1
                        self._sleep_before_retry(retry_count)
                        _log.info("Retrying same model after network error: %s retry=%d", attempt_ctx, retry_index + 1)
                        continue
                    break

                assert response is not None

                if response.ok:
                    _log.info("Model succeeded: %s retry=%d elapsed_ms=%d", attempt_ctx, retry_index, elapsed_ms)
                    if stream:
                        return self._stream_response(response, model)
                    return self._parse_full_response(response)

                body = response.text[:500]
                request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "-"
                error_class = self._classify_api_error(response.status_code, body)
                _log.error(
                    "API error: %s retry=%d class=%s status=%d request_id=%s elapsed_ms=%d body=%s",
                    attempt_ctx,
                    retry_index,
                    error_class,
                    response.status_code,
                    request_id,
                    elapsed_ms,
                    body,
                )
                if response.status_code == 401 and "User not found" in body:
                    _log.error(
                        "OpenRouter returned 401 User not found. This usually means an invalid/revoked key, "
                        "a stale env var overriding .env, or a temporary OpenRouter auth-side incident."
                    )

                last_error = "[Ошибка API %d: %s]" % (response.status_code, body)

                if error_class == "retry" and retry_count == 0:
                    retry_count += 1
                    self._sleep_before_retry(retry_count)
                    _log.info("Retrying same model after retryable API error: %s retry=%d", attempt_ctx, retry_index + 1)
                    continue

                if error_class == "fallback":
                    _log.warning("Falling back to next model: %s", attempt_ctx)
                    break

                _log.error("Stopping model chain with terminal error: %s", attempt_ctx)
                return last_error

        _log.error("Model chain exhausted without success: %s", last_error)
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
            return f"[Ошибка разбора ответа: {exc}]"

    def _stream_response(self, response: requests.Response, model: str) -> Generator[str, None, None]:
        _log.debug("Starting SSE stream: model=%s", model)
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
                    error = chunk.get("error")
                    if error:
                        message = error.get("message", "unknown stream error")
                        _log.error("Stream error event: model=%s saw_content=%s error=%s", model, saw_content, message)
                        yield "\n[Ошибка потока: %s]" % message
                        break
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        saw_content = True
                        total_chars += len(delta)
                        yield delta
                except (KeyError, IndexError, ValueError) as exc:
                    _log.debug("Skipping malformed SSE chunk: %s", exc)
        except requests.RequestException as exc:
            _log.error("Stream error: %s", exc)
            yield f"\n[Ошибка потока: {exc}]"
        finally:
            _log.info("Stream finished: model=%s chars=%d saw_content=%s", model, total_chars, saw_content)

    def _send_once(self, payload: Dict, stream: bool) -> Tuple[Optional[requests.Response], Optional[str], int]:
        started_at = time.time()
        try:
            response = requests.post(
                ENDPOINT,
                headers=self._build_headers(),
                json=payload,
                stream=stream,
                timeout=(15, 120),
            )
            elapsed_ms = int((time.time() - started_at) * 1000)
            return response, None, elapsed_ms
        except requests.RequestException as exc:
            elapsed_ms = int((time.time() - started_at) * 1000)
            return None, str(exc), elapsed_ms

    def _classify_api_error(self, status_code: int, body: str) -> str:
        lower = body.lower()

        if status_code == 400:
            return "hard_fail"
        if status_code in (401, 402):
            if "provider" in lower or "byok" in lower:
                return "fallback"
            return "hard_fail"
        if status_code == 403:
            if "terms of service" in lower:
                return "fallback"
            if "provider restriction" in lower or "routing" in lower or "geographic" in lower:
                return "fallback"
            if "moderation" in lower or "flagged_input" in lower:
                return "hard_fail"
            return "fallback"
        if status_code == 404:
            return "fallback"
        if status_code in RETRYABLE_STATUSES:
            return "retry"
        if "no endpoints found for this model" in lower:
            return "fallback"
        if "provider returned error" in lower or "provider unavailable" in lower or "model unavailable" in lower:
            return "fallback"
        if "invalid request" in lower or "moderation" in lower:
            return "hard_fail"
        return "hard_fail"

    def _sleep_before_retry(self, retry_count: int) -> None:
        delay = 0.3 if retry_count <= 1 else 0.6
        time.sleep(delay)
