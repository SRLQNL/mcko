import logging
import tkinter as tk
from tkinter import font as tkfont

_log = logging.getLogger("mcko.chat_view")

BG_COLOR = "#f7f7f7"
USER_FG = "#a0a7ab"
AI_FG = "#9f9f9f"
LABEL_FG = "#a2a8a0"
FONT_FAMILY = "Courier"
FONT_SIZE = 9
SCROLL_STEP = 1


class ChatView(tk.Text):
    """Read-only scrollable chat history widget."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=BG_COLOR,
            fg=AI_FG,
            insertbackground=AI_FG,
            selectbackground="#d7dde2",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            padx=5,
            pady=1,
            font=(FONT_FAMILY, FONT_SIZE),
            **kwargs,
        )

        # Scrollbar
        scrollbar = tk.Scrollbar(parent, command=self.yview, bg=BG_COLOR, troughcolor=BG_COLOR)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.configure(yscrollcommand=scrollbar.set)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel_linux_up)
        self.bind("<Button-5>", self._on_mousewheel_linux_down)

        # Tags
        self.tag_configure("user_label", foreground="#a8afb2", font=(FONT_FAMILY, FONT_SIZE, "bold"))
        self.tag_configure("user_text", foreground=USER_FG)
        self.tag_configure("ai_label", foreground=LABEL_FG, font=(FONT_FAMILY, FONT_SIZE, "bold"))
        self.tag_configure("ai_text", foreground=AI_FG)
        self.tag_configure("separator", foreground="#ececec")
        self.configure(spacing1=0, spacing3=1)

        _log.info("ChatView initialized")

    def _write(self, text: str, *tags) -> None:
        self.configure(state=tk.NORMAL)
        self.insert(tk.END, text, tags)
        self.configure(state=tk.DISABLED)
        self.see(tk.END)

    def append_user(self, text: str) -> None:
        """Append a user message block."""
        _log.info("Appending user message: %d chars", len(text))
        self._write("\n", "user_text")
        self._write(text + "\n", "user_text")

    def begin_assistant(self) -> None:
        """Start a new assistant response block (call before streaming chunks)."""
        _log.info("Beginning assistant response block")
        self._write("", "ai_text")

    def append_assistant_chunk(self, chunk: str) -> None:
        """Append a streaming chunk to the current assistant response."""
        self._write(chunk, "ai_text")

    def end_assistant(self) -> None:
        """Finalize the assistant response block."""
        self._write("\n", "ai_text")
        _log.info("Assistant response block ended")

    def clear(self) -> None:
        """Clear all chat content."""
        _log.info("ChatView cleared")
        self.configure(state=tk.NORMAL)
        self.delete("1.0", tk.END)
        self.configure(state=tk.DISABLED)

    def _scroll_lines(self, delta_lines: int) -> str:
        self.yview_scroll(delta_lines, "units")
        return "break"

    def _on_mousewheel(self, event) -> str:
        if event.delta == 0:
            return "break"
        direction = -SCROLL_STEP if event.delta > 0 else SCROLL_STEP
        return self._scroll_lines(direction)

    def _on_mousewheel_linux_up(self, event) -> str:
        return self._scroll_lines(-SCROLL_STEP)

    def _on_mousewheel_linux_down(self, event) -> str:
        return self._scroll_lines(SCROLL_STEP)
