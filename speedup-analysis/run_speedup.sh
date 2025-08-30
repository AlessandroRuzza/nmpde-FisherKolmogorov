#!/usr/bin/env bash
# Usage: (from speedup-analysis folder)
#   bash run_speedup.sh [config_name] [out_csv]
# Example: (from speedup-analysis folder)
#   bash run_speedup.sh Sagittal speedup_results.csv
#   bash run_speedup.sh MNI my_results.csv
# Optional env vars:
#   THREADS="2 4 8 12 16"
#   REPEAT=1
# REPEAT=3 bash run_speedup.sh Sagittal speedup_results.csv
# REPEAT=3 THREADS="4 8 16" bash run_speedup.sh MyConfig my_results.csv

set -uo pipefail  # (don't stop at errors)

CONFIG_NAME="${1:-Sagittal}"
OUT_CSV="${2:-speedup_results.csv}"
MAIN_BIN="../build/main"
LOG_DIR="./logs"

THREADS="${THREADS:-2 4 8 12 16}"
REPEAT="${REPEAT:-1}"

to_abs() {
  if command -v readlink >/dev/null 2>&1; then readlink -f "$1"; else
    python3 - "$1" <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
  fi
}

MAIN_BIN="$(to_abs "$MAIN_BIN")"
OUT_CSV="$(to_abs "$OUT_CSV")"
LOG_DIR="$(to_abs "$LOG_DIR")"

# Move to folder where MAIN_BIN is located
cd "$(dirname "$MAIN_BIN")"
mkdir -p output
mkdir -p "$LOG_DIR"

# CSV header (overwrites)
echo "config,mode,processes,run,elapsed_s" > "$OUT_CSV"

# Timer: uses /usr/bin/time
time_cmd() {
  local out_tmp="$(mktemp)"
  # -f '%e' = only elapsed seconds; stdout of the program discarded, stderr to run log
  /usr/bin/time -f '%e' -o "$out_tmp" "$@" 1>/dev/null
  local rc=$?
  if ((rc != 0)); then
    rm -f "$out_tmp"
    return $rc
  fi
  local t
  t="$(cat "$out_tmp")"
  rm -f "$out_tmp"
  printf '%s' "$t"
}

echo "[INFO] Running with $CONFIG_NAME configuration"

# Validate configuration by running a test execution with timeout
timeout_sec=1
echo "[INFO] Validating configuration: $CONFIG_NAME (timeout: $timeout_sec s, threads: $(nproc))"
timeout $timeout_sec "$MAIN_BIN" "$CONFIG_NAME" >/dev/null 2>/dev/null
exit_code=$?

if [ $exit_code -eq 124 ]; then
  # Exit code 124 means timeout - program is running
  echo "[INFO] Configuration validated successfully (execution timed out as expected)"
elif [ $exit_code -eq 0 ]; then
  # Exit code 0 means successful completion
  echo "[INFO] Configuration validated successfully (execution completed)"
else
  # Any other exit code means program failure
  echo "[ERROR] Invalid configuration or execution failed: $CONFIG_NAME" >&2
  echo "[ERROR] Program output:" >&2
  timeout $timeout_sec "$MAIN_BIN" "$CONFIG_NAME" 2>&1 || true
  exit 1
fi

# --- SEQUENTIAL ---
for r in $(seq 1 "$REPEAT"); do
  echo "=== sequential | $CONFIG_NAME | run $r/$REPEAT ==="
  log="$LOG_DIR/logs-${CONFIG_NAME}-seq-$r.log"
  # execute sequential (single process)
  if t="$(time_cmd "$MAIN_BIN" "$CONFIG_NAME" 2> "$log")"; then
    echo "$CONFIG_NAME,sequential,1,$r,$t" >> "$OUT_CSV"
  else
    echo "Sequential run failed: $CONFIG_NAME (run $r) — see $log" >&2
    echo "$CONFIG_NAME,sequential,1,$r,NA" >> "$OUT_CSV"
  fi
done

# --- MPI ---
for th in $THREADS; do
  for r in $(seq 1 "$REPEAT"); do
    echo "=== mpi | $CONFIG_NAME | ${th}p | run $r/$REPEAT ==="
    log="$LOG_DIR/logs-${CONFIG_NAME}-mpi-${th}-r${r}.log"
    # run with mpirun using n processes
    if t="$(time_cmd mpirun -host localhost:$(nproc) -np "$th" "$MAIN_BIN" "$CONFIG_NAME" 2> "$log")"; then
      echo "$CONFIG_NAME,mpi,$th,$r,$t" >> "$OUT_CSV"
    else
      echo "MPI run failed: $CONFIG_NAME with $th processes (run $r) — see $log" >&2
      echo "$CONFIG_NAME,mpi,$th,$r,NA" >> "$OUT_CSV"
    fi
  done
done

echo "All runs completed. Results saved to: $OUT_CSV"
