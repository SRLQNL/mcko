import base64
import os
from dotenv import load_dotenv
from app.logger import logger

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
            load_dotenv(env_path, override=True)
            logger.info("Loaded .env from: %s (override=True)", env_path)
        else:
            logger.warning(".env file not found at %s, using environment variables only", env_path)

        # Validate required fields
        missing = [key for key in REQUIRED_FIELDS if not os.environ.get(key)]
        if missing:
            logger.error("Missing required config fields: %s", missing)
            raise ValueError(f"Missing required environment variables: {missing}")

        self.api_key = self._normalize_secret(os.environ["OPENROUTER_API_KEY"])
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is empty after stripping whitespace")
        logger.info(
            "API key loaded: starts_with=%s, ends_with=%s, length=%d",
            self.api_key[:8],
            self.api_key[-4:],
            len(self.api_key),
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
