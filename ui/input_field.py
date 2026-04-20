import base64
import io
import logging
import tkinter as tk
from typing import Callable, List

_log = logging.getLogger("mcko.input_field")

BG_COLOR = "#f5f5f5"
FG_COLOR = "#8a8a8a"
INSERT_COLOR = "#8a8a8a"
FONT_FAMILY = "Courier"
FONT_SIZE = 11
MAX_INPUT_LINES = 2
MIN_INPUT_LINES = 1


class InputField(tk.Text):
    """Multi-line input field with atomic image-label support.

    - Enter → submit; Shift+Enter → newline
    - Ctrl+V: if clipboard has image, insert [added picN] label; otherwise paste text
    - Image labels are atomic: any keystroke on a label character deletes the whole label
    - get_content() → list of API content blocks (text + image_url)
    """

    def __init__(self, parent, on_submit: Callable[[list], None], **kwargs):
        super().__init__(
            parent,
            height=MAX_INPUT_LINES,
            wrap=tk.WORD,
            bg=BG_COLOR,
            fg=FG_COLOR,
            insertbackground=INSERT_COLOR,
            insertwidth=1,
            selectbackground="#d7dde2",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            padx=6,
            pady=3,
            font=(FONT_FAMILY, FONT_SIZE),
            takefocus=True,
            **kwargs,
        )

        self._on_submit = on_submit
        self._images: dict = {}  # "pic1" → raw image bytes (PNG)
        self._pic_counter = 0

        # Bindings
        self.bind("<Return>", self._on_enter)
        self.bind("<Shift-Return>", self._on_shift_enter)
        self.bind("<Control-v>", self._on_paste)
        self.bind("<Control-V>", self._on_paste)
        self.bind("<Control-l>", self._on_clear)
        self.bind("<Control-L>", self._on_clear)
        self.bind("<Button-1>", self._on_click_focus)
        self.bind("<Key>", self._on_key)
        self.bind("<BackSpace>", self._on_backspace)
        self.bind("<Delete>", self._on_delete)

        # Tag for image labels (visual style)
        self.tag_configure(
            "image_label",
            background="#efefef",
            foreground="#a0a0a0",
            font=(FONT_FAMILY, FONT_SIZE, "bold"),
        )

        self.configure(height=MIN_INPUT_LINES)
        _log.info("InputField initialized")

    # ─── Public ──────────────────────────────────────────────────────────────

    def get_content(self) -> List[dict]:
        """Build a list of API content blocks from current field content."""
        _log.info("Building content blocks from input field")
        blocks = []
        text_buf = ""

        content = self.get("1.0", tk.END)
        # Walk character by character, detecting image label spans
        index = "1.0"
        while True:
            next_index = self.index(f"{index}+1c")
            if self.compare(next_index, ">=", tk.END):
                break
            tags = self.tag_names(index)
            # Find if this index is inside any image tag
            pic_key = self._pic_tag_at(index)
            if pic_key:
                # Flush text buffer first
                if text_buf.strip():
                    blocks.append({"type": "text", "text": text_buf})
                    text_buf = ""
                elif text_buf:
                    text_buf = ""
                # Add image block
                img_bytes = self._images.get(pic_key)
                if img_bytes:
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                    _log.debug("Added image block for %s", pic_key)
                # Skip to end of tag range
                ranges = self.tag_ranges(pic_key)
                if ranges:
                    index = str(ranges[1])
                    continue
            else:
                char = self.get(index, next_index)
                text_buf += char
            index = next_index

        # Flush remaining text
        plain = text_buf.rstrip("\n")
        if plain:
            blocks.append({"type": "text", "text": plain})

        # If only one text block, simplify to string form
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            _log.info("Content: single text block, %d chars", len(blocks[0]["text"]))
            return blocks
        _log.info("Content: %d blocks", len(blocks))
        return blocks

    def clear(self) -> None:
        """Clear input field and all stored images."""
        self.delete("1.0", tk.END)
        self._images.clear()
        self.configure(height=MIN_INPUT_LINES)
        _log.info("InputField cleared")

    def focus(self) -> None:
        self.focus_set()

    def submit_current(self) -> None:
        content = self.get_content()
        if content:
            _log.info("Submit triggered by button: %d blocks", len(content))
            self.clear()
            self._on_submit(content)

    # ─── Event handlers ──────────────────────────────────────────────────────

    def _on_enter(self, event) -> str:
        """Submit on Enter."""
        self.submit_current()
        return "break"

    def _on_shift_enter(self, event) -> str:
        """Insert newline on Shift+Enter."""
        self.insert(tk.INSERT, "\n")
        self._sync_height()
        return "break"

    def _on_paste(self, event) -> str:
        """Handle Ctrl+V: paste image as label or paste text normally."""
        try:
            from app import clipboard as cb
            clip_type = cb.detect_type()
            if clip_type == "image":
                img_bytes = cb.read_image()
                if img_bytes:
                    self._insert_image_label(img_bytes)
                    return "break"
        except Exception as exc:
            _log.error("Error during paste handling: %s", exc)
        # Fall through to default text paste
        return None

    def _on_key(self, event) -> str:
        """If keypress lands on an image label, delete the label atomically."""
        if event.keysym in ("BackSpace", "Delete", "Return", "Tab"):
            return None
        if self._is_cursor_in_image_label():
            self._delete_label_at_cursor()
            return "break"
        self.after_idle(self._sync_height)
        return None

    def _on_backspace(self, event) -> str:
        """Backspace: if prev char is in image label, delete whole label."""
        idx = self.index(tk.INSERT)
        prev_idx = self.index(f"{idx}-1c")
        pic_key = self._pic_tag_at(prev_idx)
        if pic_key:
            self._delete_label(pic_key)
            self._sync_height()
            return "break"
        return None

    def _on_delete(self, event) -> str:
        """Delete: if next char is in image label, delete whole label."""
        idx = self.index(tk.INSERT)
        pic_key = self._pic_tag_at(idx)
        if pic_key:
            self._delete_label(pic_key)
            self._sync_height()
            return "break"
        return None

    def _on_clear(self, event) -> str:
        self.clear()
        return "break"

    def _on_click_focus(self, event) -> None:
        self.focus_set()
        self.after_idle(self._sync_height)

    # ─── Image label management ───────────────────────────────────────────────

    def _insert_image_label(self, img_bytes: bytes) -> None:
        self._pic_counter += 1
        key = f"pic{self._pic_counter}"
        label_text = f"[added {key}]"
        self._images[key] = img_bytes

        start = self.index(tk.INSERT)
        self.insert(tk.INSERT, label_text)
        end = self.index(tk.INSERT)

        # Apply unique tag (using key name) + shared visual tag
        self.tag_add(key, start, end)
        self.tag_add("image_label", start, end)

        _log.info("Image label inserted: %s, size=%d bytes", label_text, len(img_bytes))
        self._sync_height()

    def _pic_tag_at(self, index: str):
        """Return the pic key (e.g. 'pic1') for the tag at index, or None."""
        for tag in self.tag_names(index):
            if tag.startswith("pic") and tag[3:].isdigit():
                return tag
        return None

    def _is_cursor_in_image_label(self) -> bool:
        return self._pic_tag_at(self.index(tk.INSERT)) is not None

    def _delete_label_at_cursor(self) -> None:
        pic_key = self._pic_tag_at(self.index(tk.INSERT))
        if pic_key:
            self._delete_label(pic_key)

    def _delete_label(self, pic_key: str) -> None:
        ranges = self.tag_ranges(pic_key)
        if ranges:
            self.delete(str(ranges[0]), str(ranges[1]))
            self.tag_delete(pic_key)
        self._images.pop(pic_key, None)
        self._sync_height()
        _log.info("Image label deleted: %s", pic_key)

    def _sync_height(self) -> None:
        line_count = int(self.index("end-1c").split(".")[0])
        target = max(MIN_INPUT_LINES, min(MAX_INPUT_LINES, line_count))
        self.configure(height=target)
