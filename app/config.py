from __future__ import annotations

import base64
import os
from typing import List
from app.logger import logger

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

REQUIRED_FIELDS = ["OPENROUTER_API_KEY"]

DEFAULTS = {
    "OPENROUTER_MODEL": "openai/gpt-4o",
    "SYSTEM_PROMPT_1": "Ты полезный AI-ассистент. Отвечай кратко и по делу.",
    "SYSTEM_PROMPT_2": "Обработай содержимое буфера обмена и дай краткий полезный ответ.",
    "HOTKEY_WINDOW": "<ctrl>+<space>",
    "HOTKEY_SHOW": "<ctrl>+<shift>+<space>",
    "HOTKEY_CLIPBOARD": "<ctrl>+<alt>+<space>",
    "HOTKEY_SCREENSHOT": "<ctrl>+<shift>+s",
}


class Config:
    def __init__(self):
        self.api_key: str = ""
        self.api_keys: List[str] = []
        self.model: str = ""
        self.system_prompt_1: str = ""
        self.system_prompt_2: str = ""
        self.hotkey_window: str = ""
        self.hotkey_show: str = ""
        self.hotkey_clipboard: str = ""
        self.hotkey_screenshot: str = ""

    def load(self) -> None:
        logger.info("Loading configuration from .env")
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

        if os.path.exists(env_path):
            if load_dotenv is not None:
                load_dotenv(env_path, override=True)
                logger.info("Loaded .env from: %s (override=True)", env_path)
            else:
                self._load_env_fallback(env_path)
                logger.warning("python-dotenv not installed, used fallback .env loader: %s", env_path)
        else:
            logger.warning(".env file not found at %s, using environment variables only", env_path)

        raw_keys = os.environ.get("OPENROUTER_API_KEYS", "").strip()
        if raw_keys:
            self.api_keys = self._parse_secret_list(raw_keys)
        elif os.environ.get("OPENROUTER_API_KEY"):
            self.api_keys = self._parse_secret_list(os.environ["OPENROUTER_API_KEY"])
        else:
            logger.error("Missing required config fields: %s", REQUIRED_FIELDS)
            raise ValueError(f"Missing required environment variables: {REQUIRED_FIELDS}")

        self.api_keys = [key for key in self.api_keys if key]
        if not self.api_keys:
            raise ValueError("No valid OPENROUTER_API_KEY values found after normalization")

        self.api_key = self.api_keys[0]
        logger.info(
            "Primary API key loaded: starts_with=%s, ends_with=%s, length=%d, keys_total=%d",
            self.api_key[:8],
            self.api_key[-4:],
            len(self.api_key),
            len(self.api_keys),
        )
        self.model = os.environ.get("OPENROUTER_MODEL", DEFAULTS["OPENROUTER_MODEL"])
        self.system_prompt_1 = os.environ.get("SYSTEM_PROMPT_1", DEFAULTS["SYSTEM_PROMPT_1"])
        self.system_prompt_2 = os.environ.get("SYSTEM_PROMPT_2", DEFAULTS["SYSTEM_PROMPT_2"])
        self.hotkey_window = os.environ.get("HOTKEY_WINDOW", DEFAULTS["HOTKEY_WINDOW"])
        self.hotkey_show = os.environ.get("HOTKEY_SHOW", DEFAULTS["HOTKEY_SHOW"])
        self.hotkey_clipboard = os.environ.get("HOTKEY_CLIPBOARD", DEFAULTS["HOTKEY_CLIPBOARD"])
        self.hotkey_screenshot = os.environ.get("HOTKEY_SCREENSHOT", DEFAULTS["HOTKEY_SCREENSHOT"])

        logger.info(
            "Config loaded: model=%s, hotkey_window=%s, hotkey_show=%s, hotkey_clipboard=%s, hotkey_screenshot=%s",
            self.model,
            self.hotkey_window,
            self.hotkey_show,
            self.hotkey_clipboard,
            self.hotkey_screenshot,
        )

    def _normalize_secret(self, value: str) -> str:
        """Trim whitespace/BOM, unwrap quotes, and base64-decode if needed."""
        logger.info("Normalizing OPENROUTER_API_KEY from environment")
        normalized = value.strip().lstrip("\ufeff").strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in ("'", '"'):
            logger.info("OPENROUTER_API_KEY had surrounding quotes, removing them")
            normalized = normalized[1:-1].strip()
        # Если ключ не начинается с sk- — считаем что он в base64
        if not normalized.startswith("sk-"):
            try:
                normalized = base64.b64decode(normalized).decode("utf-8").strip()
                logger.info("OPENROUTER_API_KEY decoded from base64")
            except Exception as exc:
                logger.warning("base64 decode failed, using value as-is: %s", exc)
        return normalized

    def _parse_secret_list(self, raw_value: str) -> List[str]:
        """Parse OPENROUTER_API_KEYS from newline- or comma-separated values."""
        logger.info("Parsing OPENROUTER_API_KEYS from environment")
        normalized = raw_value.replace("\r", "\n")
        parts = []
        for line in normalized.split("\n"):
            for chunk in line.split(","):
                item = chunk.strip()
                if item:
                    parts.append(self._normalize_secret(item))
        logger.info("Parsed %d API keys from OPENROUTER_API_KEYS", len(parts))
        return parts

    def _load_env_fallback(self, env_path: str) -> None:
        """Minimal .env loader used when python-dotenv is unavailable."""
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
                os.environ[key] = value
