#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MAX_JOBS="${MAX_JOBS:-2}"
GPU_ID="${GPU_ID:-0}"
DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000.npz"
LOG_ROOT="outputs/mjp_g1_hrlg_bc_100k_queue_step120000"
mkdir -p "$LOG_ROOT"

if [[ ! -f "$DATASET" ]]; then
  echo "Missing dataset: $DATASET" >&2
  exit 1
fi

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

running_jobs() {
  jobs -rp | wc -l
}

wait_for_slot() {
  while [[ "$(running_jobs)" -ge "$MAX_JOBS" ]]; do
    wait -n || true
  done
}

launch_bc() {
  local ntraj="$1"
  local out_dir="models/mjp_g1_hrlg_bc/seed0_0523_202023_step120000_dr_warmstart_${ntraj}traj_100ksteps_noeval"
  mkdir -p "$out_dir"

  echo "[$(timestamp)] LAUNCH ${ntraj}traj -> $out_dir"
  (
    set +e
    echo "[$(timestamp)] START ${ntraj}traj"
    echo "dataset=$DATASET"
    echo "output_dir=$out_dir"
    echo "max_train_steps=100000"
    echo "gpu=$GPU_ID"

    CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/train_mjp_g1_hrlg_bc.py \
      --dataset "$DATASET" \
      --output-dir "$out_dir" \
      --limit-trajectories "$ntraj" \
      --epochs 20000 \
      --max-train-steps 100000 \
      --batch-size 1024 \
      --eval-batch-size 4096 \
      --lr 3e-4 \
      --weight-decay 0.0 \
      --seed 0 \
      --device cuda:0 \
      --eval-interval 100 \
      --save-interval 1000 \
      --online-eval-interval 0

    status=$?
    echo "$status" > "$out_dir/exit_code.txt"
    echo "[$(timestamp)] END ${ntraj}traj status=$status"
    exit "$status"
  ) > "$out_dir/train_stdout.log" 2>&1 &

  local pid=$!
  echo "$pid" > "$out_dir/pid"
  echo "[$(timestamp)] PID ${ntraj}traj $pid" | tee -a "$LOG_ROOT/queue.log"
}

echo "[$(timestamp)] Queue start MAX_JOBS=$MAX_JOBS GPU_ID=$GPU_ID DATASET=$DATASET" | tee -a "$LOG_ROOT/queue.log"

for ntraj in 10 25 50 100; do
  wait_for_slot
  launch_bc "$ntraj"
done

status=0
while [[ "$(running_jobs)" -gt 0 ]]; do
  wait -n || status=1
done

echo "[$(timestamp)] Queue done status=$status" | tee -a "$LOG_ROOT/queue.log"
exit "$status"
