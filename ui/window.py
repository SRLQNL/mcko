import logging
import os
import tkinter as tk

_log = logging.getLogger("mcko.window")

WINDOW_WIDTH = 300
WINDOW_HEIGHT = 150
MIN_WINDOW_WIDTH = 260
MIN_WINDOW_HEIGHT = 120
BG_COLOR = "#1e1e1e"
TITLEBAR_BG = "#2a2a2a"
TITLEBAR_FG = "#888888"
BORDER_COLOR = "#3c3c3c"
CLOSE_BTN_FG = "#666666"
CLOSE_BTN_HOVER = "#e05555"
ALPHA = 0.35  # прозрачность окна (liquid glass)


class ChatWindow:
    """Main floating chat window. Hidden by default, toggled by hotkey.

    Использует overrideredirect(True) чтобы убрать WM-декорации.
    Фокус форсируется через Xlib XSetInputFocus (минует WM).
    Ресайз — через кастомный grip в правом нижнем углу.
    """

    def __init__(self, root: tk.Tk, on_send_callback):
        self._root = root
        self._on_send = on_send_callback
        self._visible = False
        self._window: tk.Toplevel = None
        self._chat_view = None
        self._input_field = None

        # Drag state
        self._drag_x = 0
        self._drag_y = 0
        self._first_show = True

        # Resize state
        self._resize_start_x = 0
        self._resize_start_y = 0
        self._resize_start_w = 0
        self._resize_start_h = 0

        self._build()
        _log.info("ChatWindow created (hidden)")

    # ─── Build ───────────────────────────────────────────────────────────────

    def _build(self):
        from ui.chat_view import ChatView
        from ui.input_field import InputField

        win = tk.Toplevel(self._root)
        win.configure(bg=BG_COLOR)
        win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        win.resizable(True, True)
        win.takefocus = True

        # Убираем WM-декорации (KDE-полоску)
        win.overrideredirect(True)
        _log.info("overrideredirect=True")

        # Stay on top
        win.attributes("-topmost", True)

        # Полупрозрачность — через _NET_WM_WINDOW_OPACITY для X11/KWin compositor
        win.attributes("-alpha", ALPHA)  # работает без overrideredirect
        self._apply_x11_opacity(win, ALPHA)  # для overrideredirect на X11

        # Thin border to distinguish from desktop
        win.configure(highlightthickness=1, highlightbackground=BORDER_COLOR)

        # Bottom-left corner
        win.update_idletasks()
        sh = win.winfo_screenheight()
        x = 0
        y = sh - WINDOW_HEIGHT
        win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        # ── Custom title bar ─────────────────────────────────────────────────
        titlebar = tk.Frame(win, bg=TITLEBAR_BG, height=24)
        titlebar.pack(fill=tk.X, side=tk.TOP)
        titlebar.pack_propagate(False)

        title_label = tk.Label(
            titlebar,
            text="MCKO",
            bg=TITLEBAR_BG,
            fg=TITLEBAR_FG,
            font=("Courier", 10),
            anchor="w",
            padx=8,
        )
        title_label.pack(side=tk.LEFT, fill=tk.Y)

        hint_label = tk.Label(
            titlebar,
            text="Enter send",
            bg=TITLEBAR_BG,
            fg="#444444",
            font=("Courier", 9),
        )
        hint_label.pack(side=tk.LEFT, fill=tk.Y)

        close_btn = tk.Label(
            titlebar,
            text="  ×  ",
            bg=TITLEBAR_BG,
            fg=CLOSE_BTN_FG,
            font=("Courier", 13, "bold"),
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT, fill=tk.Y)
        close_btn.bind("<Button-1>", lambda e: self.hide())
        close_btn.bind("<Enter>", lambda e: close_btn.configure(fg=CLOSE_BTN_HOVER))
        close_btn.bind("<Leave>", lambda e: close_btn.configure(fg=CLOSE_BTN_FG))

        restart_btn = tk.Label(
            titlebar,
            text=" ↺ ",
            bg=TITLEBAR_BG,
            fg=CLOSE_BTN_FG,
            font=("Courier", 12, "bold"),
            cursor="hand2",
        )
        restart_btn.pack(side=tk.RIGHT, fill=tk.Y)
        restart_btn.bind("<Button-1>", lambda e: self._on_restart_click())
        restart_btn.bind("<Enter>", lambda e: restart_btn.configure(fg="#55aaff"))
        restart_btn.bind("<Leave>", lambda e: restart_btn.configure(fg=CLOSE_BTN_FG))

        # Drag bindings on titlebar
        for widget in (titlebar, title_label, hint_label):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_motion)

        # Separator under titlebar
        tk.Frame(win, height=1, bg=BORDER_COLOR).pack(fill=tk.X)

        # ── Input area (пакуется снизу первой — всегда видима) ───────────────
        input_frame = tk.Frame(win, bg=BG_COLOR)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0, 4))

        self._input_field = InputField(input_frame, on_submit=self._handle_submit)
        self._input_field.pack(fill=tk.X, expand=True)

        tk.Frame(win, height=1, bg=BORDER_COLOR).pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

        # ── Chat area (заполняет оставшееся пространство) ────────────────────
        chat_frame = tk.Frame(win, bg=BG_COLOR)
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        self._chat_view = ChatView(chat_frame)
        self._chat_view.pack(fill=tk.BOTH, expand=True)

        # ── Resize grip (правый нижний угол) ─────────────────────────────────
        grip = tk.Label(win, text="◢", bg=BG_COLOR, fg="#444444",
                        font=("Courier", 8), cursor="sizing")
        grip.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)
        grip.bind("<ButtonPress-1>", self._resize_start)
        grip.bind("<B1-Motion>", self._resize_motion)

        win.bind("<Escape>", lambda e: self.hide())

        # Start hidden
        win.withdraw()
        self._window = win

    # ─── Drag ────────────────────────────────────────────────────────────────

    def _drag_start(self, event) -> None:
        self._drag_x = event.x_root - self._window.winfo_x()
        self._drag_y = event.y_root - self._window.winfo_y()

    def _drag_motion(self, event) -> None:
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self._window.geometry(f"+{x}+{y}")

    # ─── Resize ──────────────────────────────────────────────────────────────

    def _resize_start(self, event) -> None:
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._resize_start_w = self._window.winfo_width()
        self._resize_start_h = self._window.winfo_height()

    def _resize_motion(self, event) -> None:
        dx = event.x_root - self._resize_start_x
        dy = event.y_root - self._resize_start_y
        new_w = max(MIN_WINDOW_WIDTH, self._resize_start_w + dx)
        new_h = max(MIN_WINDOW_HEIGHT, self._resize_start_h + dy)
        self._window.geometry(f"{new_w}x{new_h}")

    # ─── Visibility ──────────────────────────────────────────────────────────

    def toggle(self) -> None:
        """Show if hidden, hide if visible."""
        if self._visible:
            self.hide()
        else:
            self.show()

    def show(self) -> None:
        """Show the window and focus the input field."""
        if self._visible:
            # Already visible — just raise and focus
            self._focus_input()
            return
        _log.info("ChatWindow shown")
        if self._first_show:
            self._first_show = False
            self._window.update_idletasks()
            sh = self._window.winfo_screenheight()
            x = 0
            y = sh - WINDOW_HEIGHT
            self._window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
            _log.info("ChatWindow positioned bottom-left: +%d+%d", x, y)
        self._window.deiconify()
        self._window.update_idletasks()
        self._window.lift()
        # Переустанавливаем opacity после deiconify — compositor читает свойство у mapped окна
        self._apply_x11_opacity(self._window, ALPHA)
        self._focus_input()
        self._visible = True

    def hide(self) -> None:
        """Hide the window without destroying it."""
        if not self._visible:
            return
        _log.info("ChatWindow hidden")
        try:
            if self._window.grab_current() == self._window:
                self._window.grab_release()
                _log.info("ChatWindow grab released")
        except tk.TclError as exc:
            _log.warning("Failed to release grab while hiding window: %s", exc)
        self._window.withdraw()
        self._visible = False

    # ─── Internal ────────────────────────────────────────────────────────────

    def _handle_submit(self, content_blocks: list) -> None:
        _log.info("Submit from InputField: %d content blocks", len(content_blocks))
        self._on_send(content_blocks)

    def _focus_input(self) -> None:
        """Форсируем фокус через Xlib XSetInputFocus (минует WM, работает с overrideredirect)."""
        # Сначала Tkinter-методы
        try:
            self._window.deiconify()
            self._window.lift()
            self._window.focus_force()
            _log.info("Tkinter focus_force called")
        except tk.TclError as exc:
            _log.warning("Tkinter focus failed: %s", exc)

        # Xlib XSetInputFocus — надёжно работает с overrideredirect на X11
        is_x11 = os.name == "posix" and bool(os.environ.get("DISPLAY"))
        if is_x11:
            self._window.after(30, self._xlib_set_focus)
            self._window.after(150, self._xlib_set_focus)

        # Фокус на поле ввода
        if self._input_field:
            self._window.after_idle(self._input_field.focus_set)
            self._window.after(50, self._input_field.focus_set)
            self._window.after(200, self._input_field.focus_force)
            # Переставляем курсор за конец контента — иначе после программной вставки
            # (например, скриншота пока окно скрыто) курсор может оказаться внутри
            # тега image_label, и _on_key будет блокировать весь ввод через "break"
            self._window.after(250, self._move_cursor_to_end)
            _log.info("InputField focus scheduled")

    def _apply_x11_opacity(self, win: tk.Toplevel, alpha: float) -> None:
        """Устанавливает _NET_WM_WINDOW_OPACITY — читается KWin compositor напрямую."""
        is_x11 = os.name == "posix" and bool(os.environ.get("DISPLAY"))
        if not is_x11:
            return
        try:
            from Xlib import display as xdisplay, Xatom
            d = xdisplay.Display()
            win_id = win.winfo_id()
            xwin = d.create_resource_object("window", win_id)
            OPACITY = d.intern_atom("_NET_WM_WINDOW_OPACITY")
            opacity_value = int(alpha * 0xFFFFFFFF)
            # Xatom.CARDINAL = 6 — предопределённый X-атом, не intern_atom
            xwin.change_property(OPACITY, Xatom.CARDINAL, 32, [opacity_value])
            d.sync()
            d.close()
            _log.info("_NET_WM_WINDOW_OPACITY set: %d (alpha=%.2f)", opacity_value, alpha)
        except Exception as exc:
            _log.warning("Failed to set _NET_WM_WINDOW_OPACITY: %s", exc)

    def _on_restart_click(self) -> None:
        """Запускает перезапуск приложения через kill.sh → run.sh."""
        _log.info("Restart button clicked")
        from app.restart import restart_app
        restart_app()

    def _move_cursor_to_end(self) -> None:
        """Перемещает курсор в конец поля ввода, за пределы любых тегов image_label."""
        if self._input_field:
            try:
                self._input_field.mark_set(tk.INSERT, tk.END)
                self._input_field.see(tk.INSERT)
                _log.debug("InputField cursor moved to END")
            except tk.TclError as exc:
                _log.warning("Failed to move cursor to end: %s", exc)

    def _xlib_set_focus(self) -> None:
        """Прямой XSetInputFocus через Xlib — минует WM-политику фокуса."""
        if not self._visible:
            _log.debug("_xlib_set_focus skipped: window not visible")
            return
        try:
            from Xlib import display as xdisplay, X
            d = xdisplay.Display()
            win_id = self._window.winfo_id()
            xwin = d.create_resource_object("window", win_id)
            xwin.set_input_focus(X.RevertToParent, X.CurrentTime)
            d.sync()
            d.close()
            _log.info("Xlib XSetInputFocus done for wid=%d", win_id)
        except Exception as exc:
            _log.warning("Xlib XSetInputFocus failed: %s", exc)

    @property
    def chat_view(self):
        return self._chat_view
