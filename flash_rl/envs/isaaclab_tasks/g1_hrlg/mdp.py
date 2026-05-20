"""MDP terms for the humanoid_rl_gym-compatible 29-DoF G1 task."""

from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def scaled_base_ang_vel(
    env: ManagerBasedEnv,
    scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return base_mdp.base_ang_vel(env, asset_cfg=asset_cfg) * scale


def scaled_generated_commands(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: tuple[float, float, float],
) -> torch.Tensor:
    command = base_mdp.generated_commands(env, command_name=command_name)
    scale_tensor = command.new_tensor(scale)
    return command * scale_tensor


def scaled_joint_pos_rel(
    env: ManagerBasedEnv,
    scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return base_mdp.joint_pos_rel(env, asset_cfg=asset_cfg) * scale


def scaled_joint_vel_rel(
    env: ManagerBasedEnv,
    scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return base_mdp.joint_vel_rel(env, asset_cfg=asset_cfg) * scale


def gait_phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    phase = env.episode_length_buf.to(torch.float32) * env.step_dt / cycle_time
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

