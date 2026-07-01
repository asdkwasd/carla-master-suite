#!/usr/bin/env bash
# Autopilot supervisor — loop + multi-provider failover + backoff saat semua limit
set -uo pipefail
REPO="D:/AI/traffic_vision/master_suite"
STATE="$REPO/STATE.md"; STOP="$REPO/STOP"; export LOG="$REPO/supervisor.log"
PROMPT_FILE="$REPO/MASTER_PROMPT.txt"
SLEEP_BETWEEN=900         # 15 menit antar cycle (jaga budget request)
SLEEP_ALL_LIMITED=18000   # 5 jam: tunggu window reset kalau SEMUA provider limit
cd "$REPO"; cycle=0
while true; do
  [ -f "$STOP" ] && { echo "STOP ditemukan, keluar."; break; }
  cycle=$((cycle+1)); echo "=== Cycle $cycle  $(date -Is) ===" | tee -a "$LOG"

  PROMPT="$(cat "$PROMPT_FILE")

STATE SEKARANG:
$(cat "$STATE" 2>/dev/null || echo 'fresh start')

Ikuti RESUME RULE. Kerjakan TEPAT SATU next action, commit+push, lalu selesai."

  bash run_agent.sh "$PROMPT"
  if [ $? -eq 42 ]; then
    echo "[supervisor] semua provider limit -> tunggu reset ${SLEEP_ALL_LIMITED}s" | tee -a "$LOG"
    sleep "$SLEEP_ALL_LIMITED"
  else
    echo "[supervisor] tidur ${SLEEP_BETWEEN}s untuk jaga budget request..." | tee -a "$LOG"
    sleep "$SLEEP_BETWEEN"
  fi
done
