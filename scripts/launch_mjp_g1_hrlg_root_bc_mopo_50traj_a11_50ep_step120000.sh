#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
ONLINE_EVAL_DEVICE="${ONLINE_EVAL_DEVICE:-cuda:0}"

DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000_observable_reward.npz"
BC_CHECKPOINT="models/mjp_g1_hrlg_bc/seed0_0523_202023_step120000_dr_warmstart_50traj_100ksteps_noeval/final"
WORLD_MODEL_DIR="models/mjp_g1_hrlg_bc_mopo/root_world_model_seed0_0523_202023_step120000_dr_warmstart_50traj_observable"
WORLD_MODEL_CKPT="${WORLD_MODEL_DIR}/best_model"
OUTPUT_DIR="models/mjp_g1_hrlg_bc_mopo/root_50traj_a11_bc_mopo_50ep_eval10_step120000"
LOG_ROOT="outputs/mjp_g1_hrlg_bc_mopo_50traj_a11_50ep_step120000"

mkdir -p "${WORLD_MODEL_DIR}" "${OUTPUT_DIR}" "${LOG_ROOT}"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "JAX_PLATFORMS=cuda"
  "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "OMP_NUM_THREADS=2"
  "MKL_NUM_THREADS=2"
  "NUMEXPR_NUM_THREADS=2"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  log "start ${name}"
  set +e
  env "${COMMON_ENV[@]}" "$@" >"${log_file}" 2>&1
  local code=$?
  set -e
  echo "${code}" >"${LOG_ROOT}/${name}.exit_code"
  if [[ "${code}" -ne 0 ]]; then
    log "${name} failed with exit code ${code}; see ${log_file}"
    exit "${code}"
  fi
  log "done ${name}"
}

if [[ ! -f "${DATASET}" ]]; then
  log "missing relabeled dataset: ${DATASET}"
  exit 1
fi

if [[ ! -f "${BC_CHECKPOINT}/actor.pt" ]]; then
  log "missing 50traj BC checkpoint: ${BC_CHECKPOINT}/actor.pt"
  exit 1
fi

if [[ ! -f "${WORLD_MODEL_CKPT}/dynamics_ensemble.pt" ]]; then
  run_step world_model_50traj_train \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      --dataset "${DATASET}" \
      --reward-source observable \
      --limit-trajectories 50 \
      --output-dir "${WORLD_MODEL_DIR}" \
      --seed 0 \
      --device "${DEVICE}" \
      --dynamics-only \
      --predict-root \
      --formula-reward \
      --formula-done \
      --no-predict-reward-head \
      --no-predict-done-head \
      --dynamics-ensemble-size 5 \
      --dynamics-hidden-dim 256 \
      --dynamics-layers 3 \
      --dynamics-epochs 50 \
      --model-batch-size 1024 \
      --dynamics-lr 3e-4
else
  log "world model exists: ${WORLD_MODEL_CKPT}"
fi

cat >"${OUTPUT_DIR}/queue_config.json" <<JSON
{
  "source_stageA_config": "A11",
  "limit_trajectories": 50,
  "epochs": 50,
  "online_eval_interval": 10,
  "online_eval_episodes": 50,
  "actor_rl_coef": 0.003,
  "bc_alpha": 100,
  "bc_ref_alpha": 30,
  "bc_ref_on": "synthetic",
  "reward_scale": 5,
  "real_ratio": 0.95,
  "mopo_penalty_coef": 10,
  "dataset": "${DATASET}",
  "bc_checkpoint": "${BC_CHECKPOINT}",
  "world_model_checkpoint": "${WORLD_MODEL_CKPT}"
}
JSON

log "start 50traj A11 BC-MOPO policy training"
date '+%Y-%m-%d %H:%M:%S' >"${OUTPUT_DIR}/start_time"
set +e
env "${COMMON_ENV[@]}" uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
  --dataset "${DATASET}" \
  --reward-source observable \
  --limit-trajectories 50 \
  --dynamics-checkpoint "${WORLD_MODEL_CKPT}" \
  --bc-checkpoint "${BC_CHECKPOINT}" \
  --skip-dynamics-training \
  --predict-root \
  --formula-reward \
  --formula-done \
  --no-predict-reward-head \
  --no-predict-done-head \
  --rollout-horizon 1 \
  --actor-lr 1e-5 \
  --critic-lr 3e-4 \
  --temperature-lr 3e-4 \
  --critic-min-v -5 \
  --critic-max-v 20 \
  --critic-warmup-epochs 5 \
  --actor-update-period 4 \
  --no-actor-weight-normalization \
  --epochs 50 \
  --online-eval-interval 10 \
  --online-eval-episodes 50 \
  --online-eval-num-envs 50 \
  --online-eval-max-episode-steps 1000 \
  --save-interval 10 \
  --device "${DEVICE}" \
  --online-eval-device "${ONLINE_EVAL_DEVICE}" \
  --seed 0 \
  --batch-size 1024 \
  --model-batch-size 1024 \
  --rollout-batch-size 4096 \
  --actor-rl-coef 0.003 \
  --bc-alpha 100 \
  --bc-ref-alpha 30 \
  --bc-ref-on synthetic \
  --reward-scale 5 \
  --real-ratio 0.95 \
  --mopo-penalty-coef 10 \
  --output-dir "${OUTPUT_DIR}" \
  >"${OUTPUT_DIR}/train_stdout.log" 2>&1
code=$?
set -e
echo "${code}" >"${OUTPUT_DIR}/exit_code"
date '+%Y-%m-%d %H:%M:%S' >"${OUTPUT_DIR}/end_time"
log "50traj A11 BC-MOPO finished with exit code ${code}"
exit "${code}"
