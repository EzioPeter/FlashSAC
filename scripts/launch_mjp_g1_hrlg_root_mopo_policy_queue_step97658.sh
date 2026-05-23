#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_JOBS="${MAX_JOBS:-2}"
DATASET="outputs/mjp_g1_hrlg_seed0_0522_232057_step97658_buffer_100x1000_observable_reward.npz"
BASE_OUT="models/mjp_g1_hrlg_bc_mopo"

BC_10="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_10traj_200ep_noeval/final"
BC_25="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_25traj_200ep_noeval/final"
BC_50="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_50traj_200ep_noeval/final"

WM_10="$BASE_OUT/root_world_model_seed0_0522_232057_step97658_10traj_observable/best_model"
WM_25="$BASE_OUT/root_world_model_seed0_0522_232057_step97658_25traj_observable/best_model"
WM_50="$BASE_OUT/root_world_model_seed0_0522_232057_step97658_50traj_observable/best_model"

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
  --predict-root
  --formula-reward
  --formula-done
  --no-predict-reward-head
  --no-predict-done-head
  --root-loss-coef 1.0
  --qpos-root-loss-coef 1.0
  --qvel-root-loss-coef 1.0
  --quat-loss-coef 1.0
  --rollout-horizon 1
  --rollout-batch-size 4096
  --mopo-penalty-coef 1.0
  --real-ratio 0.9
  --lr 3e-4
  --actor-lr 3e-5
  --critic-lr 3e-4
  --temperature-lr 3e-4
  --reward-scale 50.0
  --critic-min-v -5
  --critic-max-v 50
  --no-actor-weight-normalization
  --epochs 200
  --skip-dynamics-training
  --online-eval-interval 10
  --online-eval-episodes 50
  --online-eval-num-envs 50
  --online-eval-max-episode-steps 1000
  --online-eval-device cuda:0
  --save-interval 10
)

QUEUE_LOG_DIR="$BASE_OUT/queue_logs"
mkdir -p "$QUEUE_LOG_DIR"
QUEUE_LOG="$QUEUE_LOG_DIR/root_policy_queue_step97658_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "Missing required path: $path"
    exit 1
  fi
}

require_file "$DATASET"
require_file "$BC_10/actor.pt"
require_file "$BC_25/actor.pt"
require_file "$BC_50/actor.pt"
require_file "$WM_10/dynamics_ensemble.pt"
require_file "$WM_25/dynamics_ensemble.pt"
require_file "$WM_50/dynamics_ensemble.pt"

SEM="$(mktemp -u)"
mkfifo "$SEM"
exec 9<>"$SEM"
rm "$SEM"
for _ in $(seq 1 "$MAX_JOBS"); do
  printf "." >&9
done

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

launch_group() {
  local ntraj="$1"
  local bc_checkpoint="$2"
  local wm_checkpoint="$3"

  local bc_mopo_dir="$BASE_OUT/root_seed0_0522_232057_step97658_${ntraj}traj_bc_mopo_200ep_eval10"
  local pure_mopo_dir="$BASE_OUT/root_seed0_0522_232057_step97658_${ntraj}traj_pure_mopo_200ep_eval10"

  run_limited "root_bc_mopo_${ntraj}traj" "$bc_mopo_dir" \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      "${COMMON_ARGS[@]}" \
      --limit-trajectories "$ntraj" \
      --dynamics-checkpoint "$wm_checkpoint" \
      --bc-checkpoint "$bc_checkpoint" \
      --bc-alpha 10.0 \
      --output-dir "$bc_mopo_dir" &

  run_limited "root_pure_mopo_${ntraj}traj" "$pure_mopo_dir" \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      "${COMMON_ARGS[@]}" \
      --limit-trajectories "$ntraj" \
      --dynamics-checkpoint "$wm_checkpoint" \
      --no-bc-checkpoint \
      --bc-alpha 0.0 \
      --output-dir "$pure_mopo_dir" &
}

log "Root-aware policy-only queue started with MAX_JOBS=$MAX_JOBS"
log "Dataset=$DATASET"
log "10traj world model=$WM_10"
log "25traj world model=$WM_25"
log "50traj world model=$WM_50"
log "10traj BC checkpoint=$BC_10"
log "25traj BC checkpoint=$BC_25"
log "50traj BC checkpoint=$BC_50"
log "No 100traj BC checkpoint is referenced by this queue."

launch_group 10 "$BC_10" "$WM_10"
launch_group 25 "$BC_25" "$WM_25"
launch_group 50 "$BC_50" "$WM_50"

status=0
wait || status=1
log "Root-aware policy-only queue finished status=$status"
exit "$status"
