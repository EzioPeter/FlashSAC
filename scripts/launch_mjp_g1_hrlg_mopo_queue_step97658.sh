#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_JOBS="${MAX_JOBS:-2}"
DATASET="outputs/mjp_g1_hrlg_seed0_0522_232057_step97658_buffer_100x1000_observable_reward.npz"

BASE_OUT="models/mjp_g1_hrlg_bc_mopo"

BC_10="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_10traj_200ep_noeval/final"
BC_25="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_25traj_200ep_noeval/final"
BC_50="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_50traj_200ep_noeval/final"

COMMON_ARGS=(
  --dataset "$DATASET"
  --reward-source observable
  --seed 0
  --device cuda:0
  --batch-size 1024
  --model-batch-size 1024
  --dynamics-ensemble-size 5
  --dynamics-hidden-dim 256
  --dynamics-layers 3
  --dynamics-epochs 50
  --rollout-horizon 1
  --rollout-batch-size 4096
  --mopo-penalty-coef 1.0
  --real-ratio 0.9
  --lr 3e-4
  --actor-lr 3e-5
  --critic-lr 3e-4
  --temperature-lr 3e-4
  --dynamics-lr 3e-4
  --reward-scale 50.0
  --critic-min-v -5
  --critic-max-v 50
  --no-actor-weight-normalization
  --online-eval-interval 10
  --online-eval-episodes 50
  --online-eval-num-envs 50
  --online-eval-max-episode-steps 1000
  --online-eval-device cuda:0
  --save-interval 10
)

QUEUE_LOG_DIR="$BASE_OUT/queue_logs"
mkdir -p "$QUEUE_LOG_DIR"
QUEUE_LOG="$QUEUE_LOG_DIR/queue_step97658_$(date +%Y%m%d_%H%M%S).log"

SEM="$(mktemp -u)"
mkfifo "$SEM"
exec 9<>"$SEM"
rm "$SEM"
for _ in $(seq 1 "$MAX_JOBS"); do
  printf "." >&9
done

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

run_limited() {
  local name="$1"
  local out_dir="$2"
  shift 2
  read -r -n 1 _ <&9
  mkdir -p "$out_dir"
  (
    set +e
    log "START name=$name out=$out_dir"
    "$@" >"$out_dir/train_stdout.log" 2>&1 &
    local cmd_pid=$!
    echo "$cmd_pid" >"$out_dir/pid"
    echo "$(date '+%F %T')" >"$out_dir/start_time.txt"
    wait "$cmd_pid"
    local status=$?
    echo "$status" >"$out_dir/exit_code"
    echo "$(date '+%F %T')" >"$out_dir/end_time.txt"
    if [[ "$status" -eq 0 ]]; then
      log "DONE name=$name status=$status out=$out_dir"
    else
      log "FAIL name=$name status=$status out=$out_dir log=$out_dir/train_stdout.log"
    fi
    printf "." >&9
    exit "$status"
  )
}

run_group() {
  local ntraj="$1"
  local bc_checkpoint="$2"

  local wm_dir="$BASE_OUT/world_model_seed0_0522_232057_step97658_${ntraj}traj_observable"
  local bc_mopo_dir="$BASE_OUT/seed0_0522_232057_step97658_${ntraj}traj_bc_mopo_200ep_eval10"
  local pure_mopo_dir="$BASE_OUT/seed0_0522_232057_step97658_${ntraj}traj_pure_mopo_200ep_eval10"
  local wm_checkpoint="$wm_dir/best_model"

  if ! run_limited "world_model_${ntraj}traj" "$wm_dir" \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      "${COMMON_ARGS[@]}" \
      --limit-trajectories "$ntraj" \
      --no-bc-checkpoint \
      --dynamics-only \
      --output-dir "$wm_dir"; then
    log "SKIP downstream jobs for ${ntraj}traj because world model failed."
    return 1
  fi

  if [[ ! -f "$wm_checkpoint/dynamics_ensemble.pt" ]]; then
    log "SKIP downstream jobs for ${ntraj}traj because missing $wm_checkpoint/dynamics_ensemble.pt"
    return 1
  fi

  run_limited "bc_mopo_${ntraj}traj" "$bc_mopo_dir" \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      "${COMMON_ARGS[@]}" \
      --limit-trajectories "$ntraj" \
      --epochs 200 \
      --skip-dynamics-training \
      --dynamics-checkpoint "$wm_checkpoint" \
      --bc-checkpoint "$bc_checkpoint" \
      --bc-alpha 10.0 \
      --output-dir "$bc_mopo_dir" &
  local bc_job_pid=$!

  run_limited "pure_mopo_${ntraj}traj" "$pure_mopo_dir" \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      "${COMMON_ARGS[@]}" \
      --limit-trajectories "$ntraj" \
      --epochs 200 \
      --skip-dynamics-training \
      --dynamics-checkpoint "$wm_checkpoint" \
      --no-bc-checkpoint \
      --bc-alpha 0.0 \
      --output-dir "$pure_mopo_dir" &
  local pure_job_pid=$!

  local status=0
  wait "$bc_job_pid" || status=1
  wait "$pure_job_pid" || status=1
  return "$status"
}

log "Queue started with MAX_JOBS=$MAX_JOBS"
log "Dataset=$DATASET"
log "10traj BC checkpoint=$BC_10"
log "25traj BC checkpoint=$BC_25"
log "50traj BC checkpoint=$BC_50"

run_group 10 "$BC_10" &
pid10=$!
run_group 25 "$BC_25" &
pid25=$!
run_group 50 "$BC_50" &
pid50=$!

status=0
wait "$pid10" || status=1
wait "$pid25" || status=1
wait "$pid50" || status=1

log "Queue finished status=$status"
exit "$status"
