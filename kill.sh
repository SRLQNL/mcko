#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATTERNS=(
  "python3 $SCRIPT_DIR/main.py"
  "bash $SCRIPT_DIR/watchdog.sh"
  "$SCRIPT_DIR/watchdog.sh"
)

echo "=== MCKO Kill ==="
echo "[i] Script dir: $SCRIPT_DIR"

PIDS=()
for pattern in "${PATTERNS[@]}"; do
    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        case " ${PIDS[*]} " in
            *" $pid "*) ;;
            *) PIDS+=("$pid") ;;
        esac
    done < <(pgrep -f "$pattern" || true)
done

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "[i] No MCKO processes found"
    exit 0
fi

echo "[*] Stopping PIDs: ${PIDS[*]}"
kill "${PIDS[@]}" 2>/dev/null || true
sleep 1

REMAINING=()
for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        REMAINING+=("$pid")
    fi
done

if [ "${#REMAINING[@]}" -gt 0 ]; then
    echo "[*] Force stopping stubborn PIDs: ${REMAINING[*]}"
    kill -9 "${REMAINING[@]}" 2>/dev/null || true
fi

echo "[✓] MCKO processes stopped"
