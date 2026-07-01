#!/usr/bin/env bash
# Universal agent dispatcher: coba provider berurutan; kena limit -> pindah.
set -uo pipefail
PROMPT="$1"
export LOG="${LOG:-./supervisor.log}"

run_claude() { claude -p "$PROMPT" --allowedTools "Bash Edit Write Read" \
  --dangerously-skip-permissions --output-format stream-json --verbose 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
run_codex()  { codex exec --full-auto "$PROMPT" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
run_gemini() { gemini -y -p "$PROMPT" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
run_aider()  { aider --yes --message "$PROMPT" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

is_rate_limited() {   # deteksi pola limit di output terakhir
  tail -n 40 "$LOG" | grep -Eiq "rate.?limit|429|quota|usage limit|temporarily unavailable|too many requests"
}

# Urutan failover — atur sesukamu. Hanya provider yang ter-install yang dicoba.
for provider in claude codex gemini aider; do
  command -v "$provider" >/dev/null 2>&1 || continue
  echo "[dispatch] coba provider: $provider" | tee -a "$LOG"
  "run_$provider"; rc=$?
  if [ "$rc" -eq 0 ] && ! is_rate_limited; then
    echo "[dispatch] SUKSES via $provider" | tee -a "$LOG"; exit 0
  fi
  echo "[dispatch] $provider gagal/limit (rc=$rc) -> pindah provider" | tee -a "$LOG"
done
echo "[dispatch] SEMUA provider limit/gagal." | tee -a "$LOG"; exit 42
