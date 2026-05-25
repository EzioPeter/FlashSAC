#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET_PATTERN="${TARGET_PATTERN:-flat_50m_gpu2_dr_from_no_dr_step190000}"
GO2_SCRIPT="${GO2_SCRIPT:-$REPO_ROOT/scripts/run_mjp_go2_hrlg_flat_gpu0.sh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/queue_logs}"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

find_g1_pids() {
  ps -eo pid=,cmd= \
    | awk -v pat="$TARGET_PATTERN" '
      $0 ~ pat && $0 ~ /train.py/ && $0 !~ /queue_mjp_go2_after_g1_hrlg/ {print $1}
    '
}

go2_already_running() {
  ps -eo pid=,cmd= \
    | awk '
      ($0 ~ /run_mjp_go2_hrlg_flat_gpu0/ || $0 ~ /mujoco_playground_go2_hrlg/) &&
      $0 !~ /queue_mjp_go2_after_g1_hrlg/ {print $1}
    ' \
    | grep -q .
}

echo "[$(timestamp)] queue watcher started"
echo "[$(timestamp)] repo: $REPO_ROOT"
echo "[$(timestamp)] waiting for G1 pattern: $TARGET_PATTERN"
echo "[$(timestamp)] Go2 script: $GO2_SCRIPT"

while true; do
  mapfile -t pids < <(find_g1_pids)
  if (( ${#pids[@]} == 0 )); then
    echo "[$(timestamp)] no matching G1 training process remains"
    break
  fi
  echo "[$(timestamp)] G1 still running; pids: ${pids[*]}"
  sleep "$POLL_SECONDS"
done

if go2_already_running; then
  echo "[$(timestamp)] Go2 training already appears to be running; not launching a duplicate"
  exit 0
fi

if [[ ! -x "$GO2_SCRIPT" ]]; then
  echo "[$(timestamp)] ERROR: Go2 script is not executable: $GO2_SCRIPT" >&2
  exit 1
fi

echo "[$(timestamp)] launching Go2 training"
bash "$GO2_SCRIPT"
status=$?
echo "[$(timestamp)] Go2 training finished with exit code $status"
exit "$status"
