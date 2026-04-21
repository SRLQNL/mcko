#!/usr/bin/env bash
# MCKO launcher — installs dependencies and starts the app via watchdog.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/mcko.log"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
PKG_DIR="$SCRIPT_DIR/packages"
WATCHDOG_PID_FILE="$SCRIPT_DIR/watchdog.pid"

echo "=== MCKO Launcher ==="
echo "[i] Script dir : $SCRIPT_DIR"
echo "[i] Packages   : $PKG_DIR"
echo "[i] Log        : $LOG"

# ── Определение пакетного менеджера ──────────────────────────────────────────
pkg_install() {
    echo "[*] pkg_install: $*"
    if command -v dnf &>/dev/null; then
        sudo dnf install -y "$@" 2>&1 | sed 's/^/    /'
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y "$@" 2>&1 | sed 's/^/    /'
    else
        echo "[!] Не найден пакетный менеджер (dnf / apt-get). Установите пакеты вручную: $*"
        exit 1
    fi
}

# ── 1. Python 3 ──────────────────────────────────────────────────────────────
echo ""
echo "── [1/8] Python 3 ──────────────────────────────────────────────────────────"
if ! command -v python3 &>/dev/null; then
    echo "[*] python3 not found, installing..."
    pkg_install python3 python3-pip
fi
PYTHON_VER="$(python3 --version 2>&1)"
echo "[✓] $PYTHON_VER"

# ── 2. pip3 ──────────────────────────────────────────────────────────────────
echo ""
echo "── [2/8] pip3 ──────────────────────────────────────────────────────────────"
if ! command -v pip3 &>/dev/null; then
    echo "[*] pip3 not found, installing..."
    pkg_install python3-pip
fi
PIP_VER="$(pip3 --version 2>&1)"
echo "[✓] $PIP_VER"

# ── 3. tkinter ───────────────────────────────────────────────────────────────
echo ""
echo "── [3/8] tkinter ───────────────────────────────────────────────────────────"
if ! python3 -c "import tkinter" &>/dev/null; then
    echo "[*] tkinter not found, installing..."
    if command -v dnf &>/dev/null; then
        pkg_install python3-tkinter
    else
        pkg_install python3-tk
    fi
fi
TKINTER_VER="$(python3 -c "import tkinter; print(tkinter.TkVersion)" 2>&1)"
echo "[✓] tkinter available (Tk $TKINTER_VER)"

# ── 4. xclip ─────────────────────────────────────────────────────────────────
echo ""
echo "── [4/8] xclip ─────────────────────────────────────────────────────────────"
if [[ "$OSTYPE" == "linux"* ]]; then
    if ! command -v xclip &>/dev/null; then
        echo "[*] xclip not found, installing..."
        pkg_install xclip
    fi
    echo "[✓] xclip: $(xclip -version 2>&1 | head -1)"
fi

# ── 5. Build tools (gcc + python3-devel, нужны для компиляции evdev) ─────────
echo ""
echo "── [5/8] Build tools ───────────────────────────────────────────────────────"
if ! command -v gcc &>/dev/null; then
    echo "[*] gcc not found, installing..."
    if command -v dnf &>/dev/null; then
        pkg_install gcc python3-devel kernel-headers
    else
        pkg_install gcc python3-dev linux-headers-generic
    fi
else
    echo "[✓] gcc: $(gcc --version | head -1)"
fi
if python3 -c "import sysconfig; print(sysconfig.get_path('include'))" &>/dev/null; then
    PYINC="$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")"
    if [ -f "$PYINC/Python.h" ]; then
        echo "[✓] Python.h found: $PYINC/Python.h"
    else
        echo "[!] Python.h NOT found at $PYINC — evdev compilation may fail"
        echo "[*] Attempting to install python3-devel..."
        if command -v dnf &>/dev/null; then
            pkg_install python3-devel
        else
            pkg_install python3-dev
        fi
    fi
fi

# ── 6. Проверка папки packages/ ──────────────────────────────────────────────
echo ""
echo "── [6/8] Packages cache ────────────────────────────────────────────────────"
if [ ! -d "$PKG_DIR" ]; then
    echo "[!] Папка packages/ не найдена: $PKG_DIR"
    exit 1
fi
PKG_COUNT="$(ls "$PKG_DIR" | wc -l | tr -d ' ')"
echo "[✓] packages/ найдена ($PKG_COUNT файлов):"
ls "$PKG_DIR" | sed 's/^/    /'

# ── 7. Python dependencies (offline, из локального кэша) ─────────────────────
echo ""
echo "── [7/8] Python dependencies ───────────────────────────────────────────────"
echo "[*] Предустановка setuptools и wheel..."
pip3 install --no-index --find-links="$PKG_DIR" setuptools wheel 2>&1 | sed 's/^/    /'

echo "[*] Установка зависимостей из requirements.txt..."
pip3 install --no-index --find-links="$PKG_DIR" -r "$SCRIPT_DIR/requirements.txt" 2>&1 | sed 's/^/    /'
echo "[✓] Dependencies installed"

# ── 8. .env setup ────────────────────────────────────────────────────────────
echo ""
echo "── [8/8] Config ─────────────────────────────────────────────────────────────"
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo "[*] .env created from template: $ENV_FILE"
    else
        echo "[!] .env not found and template is missing: $ENV_FILE"
        exit 1
    fi
fi
echo "[✓] .env found: $ENV_FILE"

if ! grep -Eq '^[[:space:]]*OPENROUTER_API_KEY=[^[:space:]#]+' "$ENV_FILE"; then
    echo "[!] OPENROUTER_API_KEY is empty in $ENV_FILE"
    echo "[!] Fill the key in .env and run bash run.sh again."
    exit 1
fi

# ── Launch via watchdog ───────────────────────────────────────────────────────
echo ""
chmod +x "$SCRIPT_DIR/watchdog.sh"

if [ -f "$WATCHDOG_PID_FILE" ]; then
    EXISTING_PID="$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)"
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        EXISTING_CMD="$(ps -p "$EXISTING_PID" -o args= 2>/dev/null || true)"
        if printf '%s' "$EXISTING_CMD" | grep -Fq "$SCRIPT_DIR/watchdog.sh"; then
            echo "[✓] MCKO watchdog already running (PID: $EXISTING_PID)"
            echo "   Logs: $LOG"
            echo "   To stop: kill $EXISTING_PID"
            exit 0
        fi
    fi
    echo "[*] Removing stale watchdog PID file: $WATCHDOG_PID_FILE"
    rm -f "$WATCHDOG_PID_FILE"
fi

echo "[*] Starting MCKO in background..."
nohup bash "$SCRIPT_DIR/watchdog.sh" >> "$LOG" 2>&1 &
WATCHDOG_PID=$!

echo ""
echo "✅ MCKO started (watchdog PID: $WATCHDOG_PID)"
echo "   Logs: $LOG"
echo "   To stop: kill $WATCHDOG_PID"
echo ""
echo "   Hotkeys:"
echo "   • Ctrl+Space      — open/close chat window"
echo "   • Ctrl+Alt+Space  — process clipboard content with AI"
