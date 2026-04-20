import logging
import tkinter as tk
from tkinter import font as tkfont

_log = logging.getLogger("mcko.chat_view")

BG_COLOR = "#1e1e1e"
USER_FG = "#9cdcfe"
AI_FG = "#d4d4d4"
LABEL_FG = "#6a9955"
FONT_FAMILY = "Courier"
FONT_SIZE = 12


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
            selectbackground="#264f78",
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=4,
            font=(FONT_FAMILY, FONT_SIZE),
            **kwargs,
        )

        # Scrollbar
        scrollbar = tk.Scrollbar(parent, command=self.yview, bg=BG_COLOR, troughcolor=BG_COLOR)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.configure(yscrollcommand=scrollbar.set)

        # Tags
        self.tag_configure("user_label", foreground=LABEL_FG, font=(FONT_FAMILY, FONT_SIZE, "bold"))
        self.tag_configure("user_text", foreground=USER_FG)
        self.tag_configure("ai_label", foreground=LABEL_FG, font=(FONT_FAMILY, FONT_SIZE, "bold"))
        self.tag_configure("ai_text", foreground=AI_FG)
        self.tag_configure("separator", foreground="#444444")

        _log.info("ChatView initialized")

    def _write(self, text: str, *tags) -> None:
        self.configure(state=tk.NORMAL)
        self.insert(tk.END, text, tags)
        self.configure(state=tk.DISABLED)
        self.see(tk.END)

    def append_user(self, text: str) -> None:
        """Append a user message block."""
        _log.info("Appending user message: %d chars", len(text))
        self._write("\nВы: ", "user_label")
        self._write(text + "\n", "user_text")

    def begin_assistant(self) -> None:
        """Start a new assistant response block (call before streaming chunks)."""
        _log.info("Beginning assistant response block")
        self._write("\nИИ: ", "ai_label")

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
