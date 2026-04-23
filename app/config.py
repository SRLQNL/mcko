from __future__ import annotations

import base64
import os

from app.logger import logger

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

REQUIRED_FIELDS = ["OPENROUTER_API_KEY"]

DEFAULTS = {
    "OPENROUTER_MODEL": "moonshotai/kimi-k2.6",
    "SYSTEM_PROMPT_1": (
        "Решай задачу максимально тщательно и приходи к правильному ответу. "
        "Внутренние рассуждения наружу не выводи. В ответе не пиши объяснения, "
        "рассуждения, вступления, комментарии, markdown, заголовки, метки вроде "
        "'Ответ:' и любой лишний текст. Выводи только итоговые ответы в формате "
        "нумерованного списка. Каждый пункт должен быть строго в виде: 1) Ответ. "
        "Если ответ один, выведи только одну строку: 1) Ответ. Если ответов несколько, "
        "выведи каждый ответ с новой строки в таком же формате. Ничего не добавляй "
        "до или после списка."
    ),
    "SYSTEM_PROMPT_2": (
        "Решай задачу максимально тщательно и приходи к правильному ответу. "
        "Внутренние рассуждения наружу не выводи. В ответе не пиши объяснения, "
        "рассуждения, вступления, комментарии, markdown, заголовки, метки вроде "
        "'Ответ:' и любой лишний текст. Выводи только итоговые ответы в формате "
        "нумерованного списка. Каждый пункт должен быть строго в виде: 1) Ответ. "
        "Если ответ один, выведи только одну строку: 1) Ответ. Если ответов несколько, "
        "выведи каждый ответ с новой строки в таком же формате. Ничего не добавляй "
        "до или после списка."
    ),
    "HOTKEY_WINDOW": "<ctrl>+<space>",
    "HOTKEY_SHOW": "<ctrl>+<shift>+<space>",
    "HOTKEY_CLIPBOARD": "<ctrl>+<alt>+<space>",
    "HOTKEY_SCREENSHOT": "<ctrl>+<shift>+s",
}


class Config:
    def __init__(self):
        self.api_key = ""
        self.model = ""
        self.system_prompt_1 = ""
        self.system_prompt_2 = ""
        self.hotkey_window = ""
        self.hotkey_show = ""
        self.hotkey_clipboard = ""
        self.hotkey_screenshot = ""

    def load(self) -> None:
        logger.info("Loading configuration from .env")
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

        if os.path.exists(env_path):
            if load_dotenv is not None:
                load_dotenv(env_path, override=False)
                logger.info("Loaded .env from: %s (override=False)", env_path)
            else:
                self._load_env_fallback(env_path)
                logger.warning("python-dotenv not installed, used fallback .env loader: %s", env_path)
        else:
            logger.warning(".env file not found at %s, using environment variables only", env_path)

        if not os.environ.get("OPENROUTER_API_KEY"):
            logger.error("Missing required config fields: %s", REQUIRED_FIELDS)
            raise ValueError("Missing required environment variables: %s" % REQUIRED_FIELDS)

        self.api_key = self._normalize_secret(os.environ["OPENROUTER_API_KEY"])
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is empty after stripping whitespace")

        logger.info(
            "Primary API key loaded: starts_with=%s, ends_with=%s, length=%d",
            self.api_key[:8],
            self.api_key[-4:],
            len(self.api_key),
        )

        self.model = self._env_or_default("OPENROUTER_MODEL")
        self.system_prompt_1 = self._env_or_default("SYSTEM_PROMPT_1")
        self.system_prompt_2 = self._env_or_default("SYSTEM_PROMPT_2")
        self.hotkey_window = self._env_or_default("HOTKEY_WINDOW")
        self.hotkey_show = self._env_or_default("HOTKEY_SHOW")
        self.hotkey_clipboard = self._env_or_default("HOTKEY_CLIPBOARD")
        self.hotkey_screenshot = self._env_or_default("HOTKEY_SCREENSHOT")

        logger.info(
            "Config loaded: model=%s, hotkey_window=%s, hotkey_show=%s, hotkey_clipboard=%s, hotkey_screenshot=%s",
            self.model,
            self.hotkey_window,
            self.hotkey_show,
            self.hotkey_clipboard,
            self.hotkey_screenshot,
        )

    def _normalize_secret(self, value: str) -> str:
        logger.info("Normalizing OPENROUTER_API_KEY from environment")
        normalized = value.strip().lstrip("\ufeff").strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in ("'", '"'):
            logger.info("OPENROUTER_API_KEY had surrounding quotes, removing them")
            normalized = normalized[1:-1].strip()
        if not normalized.startswith("sk-"):
            try:
                normalized = base64.b64decode(normalized).decode("utf-8").strip()
                logger.info("OPENROUTER_API_KEY decoded from base64")
            except Exception as exc:
                logger.warning("base64 decode failed, using value as-is: %s", exc)
        return normalized

    def _env_or_default(self, key: str) -> str:
        value = os.environ.get(key, "").strip()
        return value or DEFAULTS[key]

    def _load_env_fallback(self, env_path: str) -> None:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if value and key not in os.environ:
                    os.environ[key] = value
