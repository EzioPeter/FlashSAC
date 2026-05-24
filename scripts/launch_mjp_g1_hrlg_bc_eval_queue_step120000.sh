#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MAX_JOBS="${MAX_JOBS:-1}"
GPU_ID="${GPU_ID:-0}"
LOG_ROOT="outputs/mjp_g1_hrlg_bc_eval_queue_step120000"
mkdir -p "$LOG_ROOT"

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

run_eval() {
  local ntraj="$1"
  local seed="$2"
  local checkpoint="models/mjp_g1_hrlg_bc/seed0_0523_202023_step120000_dr_warmstart_${ntraj}traj_100ksteps_noeval/final"
  local output="outputs/eval_bc_step120000_dr_warmstart_${ntraj}traj_100ksteps_final_gpu_seed${seed}.json"
  local job_log="$LOG_ROOT/${ntraj}traj_seed${seed}.log"

  if [[ ! -f "$checkpoint/actor.pt" ]]; then
    echo "[$(timestamp)] WAIT missing checkpoint $checkpoint/actor.pt" | tee -a "$LOG_ROOT/queue.log"
    while [[ ! -f "$checkpoint/actor.pt" ]]; do
      sleep 60
    done
  fi

  if [[ -f "$output" ]]; then
    echo "[$(timestamp)] SKIP existing ${ntraj}traj seed=${seed}: $output" | tee -a "$LOG_ROOT/queue.log"
    return
  fi

  echo "[$(timestamp)] LAUNCH eval ${ntraj}traj seed=${seed}" | tee -a "$LOG_ROOT/queue.log"
  (
    set +e
    echo "[$(timestamp)] START eval ${ntraj}traj seed=${seed}"
    echo "checkpoint=$checkpoint"
    echo "output=$output"
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/eval_mjp_g1_hrlg_bc.py \
      --checkpoint "$checkpoint" \
      --num-eval-episodes 50 \
      --num-envs 50 \
      --max-episode-steps 1000 \
      --seed "$seed" \
      --device cuda:0 \
      --noise-level 1.0 \
      --output "$output" \
      --quiet
    status=$?
    echo "$status" > "$LOG_ROOT/${ntraj}traj_seed${seed}.exit_code"
    echo "[$(timestamp)] END eval ${ntraj}traj seed=${seed} status=$status"
    exit "$status"
  ) > "$job_log" 2>&1 &

  echo "$!" > "$LOG_ROOT/${ntraj}traj_seed${seed}.pid"
}

echo "[$(timestamp)] Eval queue start MAX_JOBS=$MAX_JOBS GPU_ID=$GPU_ID" | tee -a "$LOG_ROOT/queue.log"

for ntraj in 50 100; do
  for seed in 0 1 2; do
    wait_for_slot
    run_eval "$ntraj" "$seed"
  done
done

status=0
while [[ "$(running_jobs)" -gt 0 ]]; do
  wait -n || status=1
done

echo "[$(timestamp)] Eval queue done status=$status" | tee -a "$LOG_ROOT/queue.log"
exit "$status"
