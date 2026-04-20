import logging
from typing import List

_log = logging.getLogger("mcko.session")


class Session:
    """Stores in-memory chat history for the current application session."""

    def __init__(self):
        self._messages: List[dict] = []
        _log.info("Session initialized")

    def add_user(self, content) -> None:
        """Add a user message. content can be str or list of content blocks."""
        self._messages.append({"role": "user", "content": content})
        length = len(content) if isinstance(content, str) else len(content)
        _log.info("User message added: content_length=%s", length)

    def add_assistant(self, text: str) -> None:
        """Add an assistant message."""
        self._messages.append({"role": "assistant", "content": text})
        _log.info("Assistant message added: chars=%d", len(text))

    def get_history(self) -> List[dict]:
        """Return a copy of the full message history."""
        _log.debug("History requested: %d messages", len(self._messages))
        return list(self._messages)

    def clear(self) -> None:
        """Clear the session history."""
        count = len(self._messages)
        self._messages.clear()
        _log.info("Session cleared: removed %d messages", count)
