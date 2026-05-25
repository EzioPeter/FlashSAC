#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_STEP200="models/mjp_go2_hrlg/flat_lowcmd_soft_antihop/Go2JoystickFlatTerrainHRLG/seed0-0525-140507/step200000"
DEFAULT_STEP190="models/mjp_go2_hrlg/flat_lowcmd_soft_antihop/Go2JoystickFlatTerrainHRLG/seed0-0525-140507/step190000"

if [[ -n "${GO2_LOAD_PATH:-}" ]]; then
  CURRENT_CHECKPOINT="$GO2_LOAD_PATH"
elif [[ -f "${DEFAULT_STEP200}/actor.pt" ]]; then
  CURRENT_CHECKPOINT="$DEFAULT_STEP200"
else
  CURRENT_CHECKPOINT="$DEFAULT_STEP190"
fi

if [[ ! -f "${CURRENT_CHECKPOINT}/actor.pt" ]]; then
  echo "Missing actor checkpoint: ${CURRENT_CHECKPOINT}/actor.pt" >&2
  exit 1
fi

MIN_CLEAN_LENGTH="${MIN_CLEAN_LENGTH:-800}"
COMMON_NUM_ENVS="${COMMON_NUM_ENVS:-1024}"
COMMON_EVAL_ENVS="${COMMON_EVAL_ENVS:-50}"
COMMON_BATCH_SIZE="${COMMON_BATCH_SIZE:-2048}"
COMMON_BUFFER_MIN_LENGTH="${COMMON_BUFFER_MIN_LENGTH:-1024}"
COMMON_UPDATES_PER_STEP="${COMMON_UPDATES_PER_STEP:-0.5}"
COMMON_LR="${COMMON_LR:-1e-4}"
COMMON_LR_END="${COMMON_LR_END:-5e-5}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

latest_checkpoint_for_exp() {
  local exp_name="$1"
  find "models/mjp_go2_hrlg/${exp_name}/Go2JoystickFlatTerrainHRLG" \
    -mindepth 2 -maxdepth 2 -type d -name 'step*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

latest_eval_length_for_exp() {
  local exp_name="$1"
  uv run python - "$exp_name" <<'PY'
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

exp_name = sys.argv[1]
paths = sorted(
    Path("runs/mjp_go2_hrlg").glob(f"{exp_name}/Go2JoystickFlatTerrainHRLG_seed*/events.out.tfevents*"),
    key=lambda path: path.stat().st_mtime,
)
if not paths:
    print("nan")
    raise SystemExit

ea = EventAccumulator(str(paths[-1]), size_guidance={"scalars": 0})
ea.Reload()
values = ea.Scalars("avg_length")
print("nan" if not values else f"{values[-1].value:.6f}")
PY
}

assert_length_or_stop() {
  local exp_name="$1"
  local length
  length="$(latest_eval_length_for_exp "$exp_name")"
  echo "[$(timestamp)] ${exp_name} latest clean avg_length=${length}"
  uv run python - "$length" "$MIN_CLEAN_LENGTH" <<'PY'
import math
import sys

length = float(sys.argv[1])
threshold = float(sys.argv[2])
if math.isnan(length) or length < threshold:
    raise SystemExit(1)
PY
}

run_stage() {
  local stage_name="$1"
  local exp_name="$2"
  local num_env_steps="$3"
  local event_dr="$4"
  local zero_enable="$5"
  local zero_range="$6"
  local delay_enable="$7"
  local delay_range="$8"
  local push_enable="$9"
  local push_linear="${10}"
  local push_yaw="${11}"

  shift 11
  local stage_env=("$@")

  echo "[$(timestamp)] starting ${stage_name}"
  echo "[$(timestamp)] load checkpoint: ${CURRENT_CHECKPOINT}"
  echo "[$(timestamp)] exp_name: ${exp_name}"

  env "${stage_env[@]}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python train.py \
    --config_path ./configs \
    --config_name flashSAC_base \
    --overrides env=mujoco_playground_go2_hrlg \
    --overrides agent=flashSAC \
    --overrides agent.asymmetric_observation=true \
    --overrides agent_load_path="${CURRENT_CHECKPOINT}" \
    --overrides agent.load_optimizer=false \
    --overrides agent.load_reward_normalizer=false \
    --overrides env.use_domain_randomization=true \
    --overrides env.event_domain_randomization="${event_dr}" \
    --overrides env.domain_randomization_config.motor_zero_offset_enable="${zero_enable}" \
    --overrides "env.domain_randomization_config.motor_zero_offset_range=[${zero_range}]" \
    --overrides env.domain_randomization_config.action_delay_enable="${delay_enable}" \
    --overrides "env.domain_randomization_config.action_delay_step_range=[${delay_range}]" \
    --overrides env.domain_randomization_config.push_enable="${push_enable}" \
    --overrides "env.domain_randomization_config.push_interval_range=[5.0,10.0]" \
    --overrides "env.domain_randomization_config.push_linear_velocity_range=[${push_linear}]" \
    --overrides "env.domain_randomization_config.push_vertical_velocity_range=[0.0,0.0]" \
    --overrides "env.domain_randomization_config.push_roll_pitch_velocity_range=[0.0,0.0]" \
    --overrides "env.domain_randomization_config.push_yaw_velocity_range=[${push_yaw}]" \
    --overrides num_train_envs="${COMMON_NUM_ENVS}" \
    --overrides num_eval_envs="${COMMON_EVAL_ENVS}" \
    --overrides num_record_envs=0 \
    --overrides num_env_steps="${num_env_steps}" \
    --overrides evaluation_per_interaction_step=2000 \
    --overrides metrics_per_interaction_step=2000 \
    --overrides save_checkpoint_per_interaction_step=5000 \
    --overrides num_eval_episodes=50 \
    --overrides num_record_episodes=0 \
    --overrides updates_per_interaction_step="${COMMON_UPDATES_PER_STEP}" \
    --overrides agent.buffer_max_length=10000000 \
    --overrides agent.buffer_device_type=cpu \
    --overrides agent.buffer_min_length="${COMMON_BUFFER_MIN_LENGTH}" \
    --overrides agent.sample_batch_size="${COMMON_BATCH_SIZE}" \
    --overrides agent.learning_rate_init="${COMMON_LR}" \
    --overrides agent.learning_rate_peak="${COMMON_LR}" \
    --overrides agent.learning_rate_end="${COMMON_LR_END}" \
    --overrides agent.use_amp=true \
    --overrides gamma=0.97 \
    --overrides n_step=1 \
    --overrides logger_type=tensorboard \
    --overrides group_name=mjp_go2_hrlg \
    --overrides exp_name="${exp_name}"

  assert_length_or_stop "$exp_name"
  CURRENT_CHECKPOINT="$(latest_checkpoint_for_exp "$exp_name")"
  if [[ -z "${CURRENT_CHECKPOINT}" || ! -f "${CURRENT_CHECKPOINT}/actor.pt" ]]; then
    echo "[$(timestamp)] ERROR: could not find checkpoint for ${exp_name}" >&2
    exit 1
  fi
  echo "[$(timestamp)] finished ${stage_name}; next checkpoint: ${CURRENT_CHECKPOINT}"
}

run_stage \
  "DR-1 weak model randomization" \
  "flat_dr_curriculum_d1_weak_model" \
  "${DR1_NUM_ENV_STEPS:-20480000}" \
  "false" \
  "false" "0.0,0.0" \
  "false" "0,0" \
  "false" "-0.0,0.0" "-0.0,0.0" \
  GO2_DR_STATIC_FRICTION_RANGE=0.6,1.2 \
  GO2_DR_DYNAMIC_FRICTION_RANGE=0.6,1.0 \
  GO2_DR_JOINT_DEFAULT_POS_RANGE=-0.005,0.005 \
  GO2_DR_BASE_MASS_ADDED_RANGE=-0.5,0.5 \
  GO2_DR_LINK_MASS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_BASE_COM_RANGE=-0.02,0.02 \
  GO2_DR_PD_STIFFNESS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_PD_DAMPING_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_MOTOR_STRENGTH_RANGE=0.9,1.1 \
  GO2_DR_ARMATURE_MULTIPLIER_RANGE=1.0,1.02 \
  GO2_DR_FRICTIONLOSS_MULTIPLIER_RANGE=0.95,1.05

run_stage \
  "DR-2 weak model + small motor zero offset" \
  "flat_dr_curriculum_d2_zero_offset" \
  "${DR2_NUM_ENV_STEPS:-10240000}" \
  "true" \
  "true" "-0.01,0.01" \
  "false" "0,0" \
  "false" "-0.0,0.0" "-0.0,0.0" \
  GO2_DR_STATIC_FRICTION_RANGE=0.6,1.2 \
  GO2_DR_DYNAMIC_FRICTION_RANGE=0.6,1.0 \
  GO2_DR_JOINT_DEFAULT_POS_RANGE=-0.005,0.005 \
  GO2_DR_BASE_MASS_ADDED_RANGE=-0.5,0.5 \
  GO2_DR_LINK_MASS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_BASE_COM_RANGE=-0.02,0.02 \
  GO2_DR_PD_STIFFNESS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_PD_DAMPING_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_MOTOR_STRENGTH_RANGE=0.9,1.1 \
  GO2_DR_ARMATURE_MULTIPLIER_RANGE=1.0,1.02 \
  GO2_DR_FRICTIONLOSS_MULTIPLIER_RANGE=0.95,1.05

run_stage \
  "DR-3 weak model + zero offset + one-step action delay" \
  "flat_dr_curriculum_d3_delay" \
  "${DR3_NUM_ENV_STEPS:-10240000}" \
  "true" \
  "true" "-0.01,0.01" \
  "true" "0,1" \
  "false" "-0.0,0.0" "-0.0,0.0" \
  GO2_DR_STATIC_FRICTION_RANGE=0.6,1.2 \
  GO2_DR_DYNAMIC_FRICTION_RANGE=0.6,1.0 \
  GO2_DR_JOINT_DEFAULT_POS_RANGE=-0.005,0.005 \
  GO2_DR_BASE_MASS_ADDED_RANGE=-0.5,0.5 \
  GO2_DR_LINK_MASS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_BASE_COM_RANGE=-0.02,0.02 \
  GO2_DR_PD_STIFFNESS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_PD_DAMPING_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_MOTOR_STRENGTH_RANGE=0.9,1.1 \
  GO2_DR_ARMATURE_MULTIPLIER_RANGE=1.0,1.02 \
  GO2_DR_FRICTIONLOSS_MULTIPLIER_RANGE=0.95,1.05

run_stage \
  "DR-4 weak model + zero offset + delay + weak push" \
  "flat_dr_curriculum_d4_push" \
  "${DR4_NUM_ENV_STEPS:-10240000}" \
  "true" \
  "true" "-0.01,0.01" \
  "true" "0,1" \
  "true" "-0.2,0.2" "-0.3,0.3" \
  GO2_DR_STATIC_FRICTION_RANGE=0.6,1.2 \
  GO2_DR_DYNAMIC_FRICTION_RANGE=0.6,1.0 \
  GO2_DR_JOINT_DEFAULT_POS_RANGE=-0.005,0.005 \
  GO2_DR_BASE_MASS_ADDED_RANGE=-0.5,0.5 \
  GO2_DR_LINK_MASS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_BASE_COM_RANGE=-0.02,0.02 \
  GO2_DR_PD_STIFFNESS_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_PD_DAMPING_MULTIPLIER_RANGE=0.95,1.05 \
  GO2_DR_MOTOR_STRENGTH_RANGE=0.9,1.1 \
  GO2_DR_ARMATURE_MULTIPLIER_RANGE=1.0,1.02 \
  GO2_DR_FRICTIONLOSS_MULTIPLIER_RANGE=0.95,1.05

echo "[$(timestamp)] DR curriculum finished. Final checkpoint: ${CURRENT_CHECKPOINT}"
