#!/usr/bin/env bash
# Usage: (from speedup-analysis folder)
#   bash run_speedup.sh [out_csv]
# Example: (from speedup-analysis folder)
#   bash run_speedup.sh speedup_results.csv
# Optional env vars:
#   MESH_PRESETS="Sagittal MNI"
#   THREADS="2 4 8 12 16"
#   REPEAT=1
# REPEAT=3 bash run_speedup.sh speedup_results.csv
# REPEAT=3 THREADS="4 8 16" MESH_PRESETS="Sagittal Cube40" bash run_speedup.sh my_results.csv

set -uo pipefail  # (don't stop at errors)

OUT_CSV="${1:-speedup_results.csv}"
MAIN_BIN="../build/main"
LOG_DIR="./logs"

MESH_PRESETS="${MESH_PRESETS:-Sagittal}"
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
echo "mesh_preset,mode,processes,run,elapsed_s" > "$OUT_CSV"

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

echo "[INFO] Running with mesh presets: $MESH_PRESETS"

# Validate mesh presets by running test executions with timeout
timeout_sec=5
for mesh_preset in $MESH_PRESETS; do
  echo "[INFO] Validating mesh preset: $mesh_preset (timeout: ${timeout_sec}s)"
  timeout $timeout_sec "$MAIN_BIN" "$mesh_preset" >/dev/null 2>/dev/null
  exit_code=$?

  if [ $exit_code -eq 124 ]; then
    # Exit code 124 means timeout - program is running
    echo "[INFO] Mesh preset validated successfully: $mesh_preset (execution timed out as expected)"
  elif [ $exit_code -eq 0 ]; then
    # Exit code 0 means successful completion
    echo "[INFO] Mesh preset validated successfully: $mesh_preset (execution completed)"
  else
    # Any other exit code means program failure
    echo "[ERROR] Invalid mesh preset or execution failed: $mesh_preset" >&2
    echo "[ERROR] Program output:" >&2
    timeout $timeout_sec "$MAIN_BIN" "$mesh_preset" 2>&1 || true
    exit 1
  fi
done

# --- MAIN BENCHMARK LOOPS ---
for mesh_preset in $MESH_PRESETS; do
  echo "[INFO] Running benchmarks for mesh preset: $mesh_preset"
  
  # --- SEQUENTIAL ---
  for r in $(seq 1 "$REPEAT"); do
    echo "=== sequential | $mesh_preset | run $r/$REPEAT ==="
    log="$LOG_DIR/logs-${mesh_preset}-seq-$r.log"
    # execute sequential (single process)
    if t="$(time_cmd "$MAIN_BIN" "$mesh_preset" 2> "$log")"; then
      echo "$mesh_preset,sequential,1,$r,$t" >> "$OUT_CSV"
    else
      echo "Sequential run failed: $mesh_preset (run $r) — see $log" >&2
      echo "$mesh_preset,sequential,1,$r,NA" >> "$OUT_CSV"
    fi
  done

  # --- MPI ---
  for th in $THREADS; do
    for r in $(seq 1 "$REPEAT"); do
      echo "=== mpi | $mesh_preset | ${th}p | run $r/$REPEAT ==="
      log="$LOG_DIR/logs-${mesh_preset}-mpi-${th}-r${r}.log"
      # run with mpirun using n processes
      if t="$(time_cmd mpirun -host localhost:$(nproc) -np "$th" "$MAIN_BIN" "$mesh_preset" 2> "$log")"; then
        echo "$mesh_preset,mpi,$th,$r,$t" >> "$OUT_CSV"
      else
        echo "MPI run failed: $mesh_preset with $th processes (run $r) — see $log" >&2
        echo "$mesh_preset,mpi,$th,$r,NA" >> "$OUT_CSV"
      fi
    done
  done
done

echo "All runs completed. Results saved to: $OUT_CSV"
