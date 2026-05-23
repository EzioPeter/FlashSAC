#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  --config_path ./configs \
  --config_name flashSAC_base \
  --overrides env=mujoco_playground_g1_hrlg \
  --overrides agent=flashSAC \
  --overrides agent.asymmetric_observation=true \
  --overrides env.use_domain_randomization=true \
  --overrides num_train_envs=1024 \
  --overrides num_eval_envs=50 \
  --overrides num_record_envs=0 \
  --overrides num_env_steps=200003584 \
  --overrides save_checkpoint_per_interaction_step=48829 \
  --overrides num_eval_episodes=50 \
  --overrides num_record_episodes=0 \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.buffer_max_length=10000000 \
  --overrides agent.buffer_device_type=cpu \
  --overrides agent.buffer_min_length=100000 \
  --overrides agent.sample_batch_size=2048 \
  --overrides agent.use_amp=true \
  --overrides gamma=0.97 \
  --overrides n_step=1 \
  --overrides logger_type=tensorboard \
  --overrides group_name=mjp_g1_hrlg \
  --overrides exp_name=flat_50m_gpu2
