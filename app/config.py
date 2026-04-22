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
    "HOTKEY_WINDOW": "<ctrl>+<space>",
    "HOTKEY_SHOW": "<ctrl>+<shift>+<space>",
    "HOTKEY_CLIPBOARD": "<ctrl>+<alt>+<space>",
    "HOTKEY_SCREENSHOT": "<ctrl>+<shift>+s",
    "PHOTO_SOLVER_MODE": "cheap",
    "PHOTO_SOLVER_KIMI_MODEL": "moonshotai/kimi-k2.6",
    "PHOTO_SOLVER_QWEN_MODEL": "qwen/qwen2.5-vl-72b-instruct",
    "PHOTO_SOLVER_LLAMA_MODEL": "meta-llama/llama-4-maverick",
}


class Config:
    def __init__(self):
        self.api_key: str = ""
        self.hotkey_window: str = ""
        self.hotkey_show: str = ""
        self.hotkey_clipboard: str = ""
        self.hotkey_screenshot: str = ""
        self.photo_solver_mode: str = ""
        self.photo_solver_kimi_model: str = ""
        self.photo_solver_qwen_model: str = ""
        self.photo_solver_llama_model: str = ""

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
            raise ValueError(f"Missing required environment variables: {REQUIRED_FIELDS}")

        self.api_key = self._normalize_secret(os.environ["OPENROUTER_API_KEY"])
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is empty after stripping whitespace")
        logger.info(
            "Primary API key loaded: starts_with=%s, ends_with=%s, length=%d",
            self.api_key[:8],
            self.api_key[-4:],
            len(self.api_key),
        )
        self.hotkey_window = self._env_or_default("HOTKEY_WINDOW")
        self.hotkey_show = self._env_or_default("HOTKEY_SHOW")
        self.hotkey_clipboard = self._env_or_default("HOTKEY_CLIPBOARD")
        self.hotkey_screenshot = self._env_or_default("HOTKEY_SCREENSHOT")
        self.photo_solver_mode = self._env_or_default("PHOTO_SOLVER_MODE")
        self.photo_solver_kimi_model = self._env_or_default("PHOTO_SOLVER_KIMI_MODEL")
        self.photo_solver_qwen_model = self._env_or_default("PHOTO_SOLVER_QWEN_MODEL")
        self.photo_solver_llama_model = self._env_or_default("PHOTO_SOLVER_LLAMA_MODEL")

        logger.info(
            "Config loaded: hotkey_window=%s, hotkey_show=%s, hotkey_clipboard=%s, hotkey_screenshot=%s, photo_solver_mode=%s, kimi=%s, qwen=%s, llama=%s",
            self.hotkey_window,
            self.hotkey_show,
            self.hotkey_clipboard,
            self.hotkey_screenshot,
            self.photo_solver_mode,
            self.photo_solver_kimi_model,
            self.photo_solver_qwen_model,
            self.photo_solver_llama_model,
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

    def _env_or_default(self, key: str) -> str:
        value = os.environ.get(key, "").strip()
        return value or DEFAULTS[key]

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
                if value and key not in os.environ:
                    os.environ[key] = value
