#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "[%s] launcher failed at line %s with exit_code=%s\n" "$(date "+%F %T")" "$LINENO" "$code" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MAX_JOBS="${MAX_JOBS:-2}"
FORCE="${FORCE:-0}"
GPU_IDS="${GPU_IDS:-0}"
RUN_TAG="${RUN_TAG:-a00_step050000bc_seedrep50}"

TRAIN_SCRIPT="scripts/train_mjp_g1_hrlg_ages_mopo.py"
SUMMARY_SCRIPT="scripts/summarize_mjp_g1_hrlg_ages_tune.py"
ROOT_DIR="models/mjp_g1_hrlg_ages_mopo"
SUMMARY_CSV="outputs/mjp_g1_hrlg_${RUN_TAG}_summary.csv"

DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000_observable_reward.npz"
DYN_CKPT="models/mjp_g1_hrlg_bc_mopo/root_world_model_seed0_0523_202023_step120000_dr_warmstart_50traj_observable/best_model"
BC_CKPT="models/mjp_g1_hrlg_bc/seed0_0524_155708_step190000_no_dr_50traj_200ksteps_noeval/step_050000"

mkdir -p outputs "$ROOT_DIR"

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
RUNNING_PIDS=()
LAUNCH_COUNT=0

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "missing required path: $1" >&2
    exit 1
  fi
}

gpu_python_count() {
  local out
  if out="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"; then
    printf '%s\n' "$out" | awk 'BEGIN{n=0} /python/{n++} END{print n}'
  else
    pgrep -af 'train_mjp_g1_hrlg_ages_mopo.py|eval_mjp_g1_hrlg' | wc -l
  fi
}

prune_running() {
  local kept=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kept+=("$pid")
    fi
  done
  RUNNING_PIDS=("${kept[@]}")
}

wait_for_internal_slot() {
  while true; do
    prune_running
    if (( ${#RUNNING_PIDS[@]} < MAX_JOBS )); then
      return
    fi
    wait -n || true
  done
}

wait_for_gpu_slot() {
  while true; do
    local count
    count="$(gpu_python_count)"
    if (( count < MAX_JOBS )); then
      return
    fi
    log "GPU already has $count python processes, waiting for a free slot"
    sleep 60
  done
}

wait_for_all() {
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" || true
  done
  RUNNING_PIDS=()
}

device_for_next_job() {
  local n=${#GPU_ID_ARRAY[@]}
  local idx=$(( LAUNCH_COUNT % n ))
  printf 'cuda:%s' "${GPU_ID_ARRAY[$idx]}"
}

run_summary() {
  uv run python "$SUMMARY_SCRIPT" --root "$ROOT_DIR" --output "$SUMMARY_CSV"
}

launch_job() {
  local output_dir="$1"
  local cmd="$2"
  mkdir -p "$output_dir"
  if [[ "$FORCE" != "1" && -f "$output_dir/exit_code" && "$(cat "$output_dir/exit_code")" == "0" ]]; then
    log "skip completed $output_dir"
    return
  fi
  wait_for_internal_slot
  wait_for_gpu_slot
  log "launch $output_dir"
  printf '%s\n' "$cmd" > "$output_dir/command.txt"
  (
    date '+%F %T' > "$output_dir/started_at.txt"
    set +e
    bash -lc "$cmd" > "$output_dir/train_stdout.log" 2>&1
    code=$?
    echo "$code" > "$output_dir/exit_code"
    date '+%F %T' > "$output_dir/finished_at.txt"
    exit "$code"
  ) &
  local pid=$!
  echo "$pid" > "$output_dir/job.pid"
  RUNNING_PIDS+=("$pid")
  LAUNCH_COUNT=$(( LAUNCH_COUNT + 1 ))
}

base_args() {
  local seed="$1"
  local device="$2"
  cat <<ARGS | tr '\n' ' '
--dataset $DATASET
--reward-source observable
--limit-trajectories 50
--dynamics-checkpoint $DYN_CKPT
--bc-checkpoint $BC_CKPT
--skip-dynamics-training
--predict-root
--formula-reward
--formula-done
--no-predict-reward-head
--no-predict-done-head
--rollout-horizon 1
--actor-lr 1e-5
--critic-lr 3e-4
--temperature-lr 3e-4
--critic-min-v -5
--critic-max-v 20
--critic-warmup-epochs 5
--actor-update-period 4
--no-actor-weight-normalization
--device $device
--online-eval-device $device
--seed $seed
--data-split-seed 0
--online-eval-max-episode-steps 1000
--save-interval 20
--epochs 200
--online-eval-interval 20
--online-eval-episodes 50
--online-eval-num-envs 50
--no-use-ages
--actor-rl-coef 0.003
--bc-alpha 100
--bc-ref-alpha 30
--bc-ref-on synthetic
--real-ratio 0.50
--mopo-penalty-coef 10
--reward-scale 5
ARGS
}

launch_a00() {
  local seed="$1"
  local device
  device="$(device_for_next_job)"
  local out="$ROOT_DIR/${RUN_TAG}_seed${seed}"
  local args
  args="$(base_args "$seed" "$device")"
  local cmd="uv run python $TRAIN_SCRIPT $args --output-dir $out"
  launch_job "$out" "$cmd"
}

require_path "$TRAIN_SCRIPT"
require_path "$SUMMARY_SCRIPT"
require_path "$DATASET"
require_path "$DYN_CKPT/dynamics_ensemble.pt"
require_path "$BC_CKPT/actor.pt"

log "starting A00 BC-MOPO seed replication with step_050000 BC checkpoint"
log "GPU_IDS=$GPU_IDS MAX_JOBS=$MAX_JOBS RUN_TAG=$RUN_TAG"
log "dataset=$DATASET"
log "dynamics=$DYN_CKPT"
log "bc_checkpoint=$BC_CKPT"

launch_a00 0
launch_a00 1
launch_a00 2

wait_for_all
run_summary
log "A00 step_050000 BC seed replication queue finished"
