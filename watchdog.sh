#!/usr/bin/env bash
# Watchdog: restarts main.py if it exits unexpectedly.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/mcko.log"
PID_FILE="$SCRIPT_DIR/watchdog.pid"

echo "$$" > "$PID_FILE"
cleanup() {
    rm -f "$PID_FILE"
}
trap cleanup EXIT

echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Watchdog started (PID $$)" >> "$LOG"

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Starting main.py" >> "$LOG"
    python3 "$SCRIPT_DIR/main.py" >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] main.py exited with code $EXIT_CODE, restarting in 3s..." >> "$LOG"
    sleep 3
done
