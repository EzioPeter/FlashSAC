#!/usr/bin/env python3
"""AGES-augmented BC-MOPO training for MuJoCo Playground G1 HRLG.

This file intentionally copies and extends ``train_mjp_g1_hrlg_bc_mopo.py`` so
the original BC-MOPO script and training flow remain untouched.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic, FlashSACTemperature
from flash_rl.agents.flashSAC.update import update_critic, update_target_network, update_temperature
from flash_rl.agents.utils.network import Network
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import constants as hrlg


DEFAULT_DATASET = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"
DEFAULT_BC_CHECKPOINT = REPO_ROOT / "models/mjp_g1_hrlg_bc/seed0_100traj/epoch_0150"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models/mjp_g1_hrlg_ages_mopo/seed0_100traj_h3_pen1"

SLICES = {
    "gyro": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 9),
    "joint_pos": slice(9, 38),
    "joint_vel": slice(38, 67),
    "last_act": slice(67, 96),
    "gait": slice(96, 98),
}


@dataclass(frozen=True)
class TransitionSplit:
    train_trajectories: list[int]
    val_trajectories: list[int]
    train_transitions: int
    val_transitions: int
    obs_dim: int
    action_dim: int
    reward_available: bool
    reward_strict: bool
    done_available: bool
    strict_mopo: bool


@dataclass
class TransitionArrays:
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    trajectory: np.ndarray
    step: np.ndarray
    qpos_root: np.ndarray | None = None
    qvel_root: np.ndarray | None = None
    next_qpos_root: np.ndarray | None = None
    next_qvel_root: np.ndarray | None = None

    def select(self, indices: np.ndarray) -> "TransitionArrays":
        return TransitionArrays(
            observation=self.observation[indices],
            action=self.action[indices],
            next_observation=self.next_observation[indices],
            reward=self.reward[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            trajectory=self.trajectory[indices],
            step=self.step[indices],
            qpos_root=None if self.qpos_root is None else self.qpos_root[indices],
            qvel_root=None if self.qvel_root is None else self.qvel_root[indices],
            next_qpos_root=None if self.next_qpos_root is None else self.next_qpos_root[indices],
            next_qvel_root=None if self.next_qvel_root is None else self.next_qvel_root[indices],
        )

    @property
    def size(self) -> int:
        return int(self.observation.shape[0])


class DynamicsMember(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        modules: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, layers)):
            modules.append(nn.Linear(dim, hidden_dim))
            modules.append(nn.SiLU())
            dim = hidden_dim
        modules.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsEnsemble(nn.Module):
    def __init__(
        self,
        *,
        ensemble_size: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        layers: int,
        predict_root: bool,
        predict_reward: bool,
        predict_done: bool,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.root_dim = 13 if predict_root else 0
        self.predict_root = predict_root
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        root_output_dim = self.root_dim
        extra_dim = int(predict_reward) + int(predict_done)
        output_dim = obs_dim + root_output_dim + extra_dim
        self.members = nn.ModuleList(
            DynamicsMember(obs_dim + action_dim + self.root_dim, output_dim, hidden_dim, layers)
            for _ in range(ensemble_size)
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        qpos_root: torch.Tensor | None = None,
        qvel_root: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [obs, action]
        if self.predict_root:
            if qpos_root is None or qvel_root is None:
                raise ValueError("Root-aware dynamics requires qpos_root and qvel_root inputs.")
            parts.extend([qpos_root, qvel_root])
        x = torch.cat(parts, dim=-1)
        return torch.stack([member(x) for member in self.members], dim=0)

    def split_prediction(
        self,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        delta = pred[..., : self.obs_dim]
        offset = self.obs_dim
        delta_qpos_root = None
        delta_qvel_root = None
        if self.predict_root:
            delta_qpos_root = pred[..., offset : offset + 7]
            offset += 7
            delta_qvel_root = pred[..., offset : offset + 6]
            offset += 6
        reward = None
        done_logit = None
        if self.predict_reward:
            reward = pred[..., offset]
            offset += 1
        if self.predict_done:
            done_logit = pred[..., offset]
        return delta, delta_qpos_root, delta_qvel_root, reward, done_logit


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _public_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if isinstance(value, Path):
            config[key] = str(value.expanduser().resolve())
        else:
            config[key] = value
    return config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "last": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "last": float(arr[-1]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _fit_obs_stats(observations: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = observations.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, eps).astype(np.float32)


def _fit_root_stats(
    qpos_root: np.ndarray | None,
    qvel_root: np.ndarray | None,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    if qpos_root is None or qvel_root is None:
        return {
            "qpos_root_mean": np.zeros(7, dtype=np.float32),
            "qpos_root_std": np.ones(7, dtype=np.float32),
            "qvel_root_mean": np.zeros(6, dtype=np.float32),
            "qvel_root_std": np.ones(6, dtype=np.float32),
            "enabled": np.asarray(False),
        }
    return {
        "qpos_root_mean": qpos_root.mean(axis=0, dtype=np.float64).astype(np.float32),
        "qpos_root_std": np.maximum(qpos_root.std(axis=0, dtype=np.float64), eps).astype(np.float32),
        "qvel_root_mean": qvel_root.mean(axis=0, dtype=np.float64).astype(np.float32),
        "qvel_root_std": np.maximum(qvel_root.std(axis=0, dtype=np.float64), eps).astype(np.float32),
        "enabled": np.asarray(True),
    }


def _load_root_stats_from_dynamics_checkpoint(checkpoint: Path) -> dict[str, np.ndarray]:
    stats_path = checkpoint / "root_stats.npz"
    if not stats_path.exists():
        return _fit_root_stats(None, None)
    with np.load(stats_path) as stats:
        return {
            "qpos_root_mean": stats["qpos_root_mean"].astype(np.float32),
            "qpos_root_std": np.maximum(stats["qpos_root_std"].astype(np.float32), 1e-6),
            "qvel_root_mean": stats["qvel_root_mean"].astype(np.float32),
            "qvel_root_std": np.maximum(stats["qvel_root_std"].astype(np.float32), 1e-6),
            "enabled": np.asarray(bool(np.asarray(stats["enabled"]).item())),
        }


def _quat_conj_torch(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, dim=-1)
    bw, bx, by, bz = torch.unbind(b, dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _rotate_world_to_body_torch(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(vec.shape[:-1] + (1,), dtype=vec.dtype, device=vec.device)
    vec_quat = torch.cat([zeros, vec], dim=-1)
    return _quat_mul_torch(_quat_mul_torch(_quat_conj_torch(quat_wxyz), vec_quat), quat_wxyz)[..., 1:]


def _normalize_root_quat(qpos_root: torch.Tensor) -> torch.Tensor:
    qpos_root = qpos_root.clone()
    quat = qpos_root[..., 3:7]
    qpos_root[..., 3:7] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return qpos_root


def _make_observable_reward_context(device: torch.device) -> dict[str, Any]:
    from scripts.relabel_mjp_g1_hrlg_observable_reward import (
        ObservableRewardConfig,
        _make_env_metadata,
    )

    reward_scales, env_info = _make_env_metadata()
    config = ObservableRewardConfig(
        dt=float(env_info["dt"]),
        tracking_sigma=float(env_info["tracking_sigma"]),
        bad_orientation_limit_rad=1.0,
        base_height_done_threshold=0.0,
        use_projected_gravity_orientation=True,
        linear_velocity_frame="root_world_velocity_rotated_to_base_frame",
        angular_velocity_source="unscaled_policy_gyro",
    )
    selected_terms = (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "orientation",
        "ang_vel_xy",
        "joint_deviation_hip",
        "joint_deviation_knee",
        "pose",
        "dof_pos_limits",
        "stand_still",
        "action_rate",
        "base_height",
        "dof_acc",
        "termination",
    )
    return {
        "dt": float(config.dt),
        "tracking_sigma": float(config.tracking_sigma),
        "bad_orientation_limit_rad": float(config.bad_orientation_limit_rad),
        "base_height_done_threshold": float(config.base_height_done_threshold),
        "base_height_target": float(env_info["base_height_target"]),
        "default_pose": torch.as_tensor(
            np.array(env_info["default_pose"], copy=True), dtype=torch.float32, device=device
        ),
        "hip_indices": torch.as_tensor(np.array(env_info["hip_indices"], copy=True), dtype=torch.long, device=device),
        "knee_indices": torch.as_tensor(np.array(env_info["knee_indices"], copy=True), dtype=torch.long, device=device),
        "soft_lowers": torch.as_tensor(
            np.array(env_info["soft_lowers"], copy=True), dtype=torch.float32, device=device
        ),
        "soft_uppers": torch.as_tensor(
            np.array(env_info["soft_uppers"], copy=True), dtype=torch.float32, device=device
        ),
        "reward_scales": {name: float(reward_scales.get(name, 0.0)) for name in selected_terms},
        "selected_terms": selected_terms,
        "metadata": {
            "reward_scales": {name: float(reward_scales.get(name, 0.0)) for name in selected_terms},
            "selected_terms": list(selected_terms),
            "formula_reward": "torch_observable_reward_from_predicted_next_obs_root",
            "base_height_done_threshold": float(config.base_height_done_threshold),
            "bad_orientation_limit_rad": float(config.bad_orientation_limit_rad),
        },
    }


def _compute_observable_reward_torch(
    *,
    obs_t_norm: torch.Tensor,
    act_t: torch.Tensor,
    obs_tp1_norm: torch.Tensor,
    qpos_root_tp1: torch.Tensor,
    qvel_root_tp1: torch.Tensor,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    context: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    obs_t = obs_t_norm * obs_std + obs_mean
    obs_tp1 = obs_tp1_norm * obs_std + obs_mean
    cmd_scale = torch.as_tensor(hrlg.CMD_SCALE, dtype=obs_t.dtype, device=obs_t.device)
    command = obs_t[..., SLICES["command"]] / cmd_scale
    gyro = obs_tp1[..., SLICES["gyro"]] / float(hrlg.ANG_VEL_SCALE)
    projected_gravity = obs_tp1[..., SLICES["projected_gravity"]]
    joint_pos_rel = obs_tp1[..., SLICES["joint_pos"]] / float(hrlg.DOF_POS_SCALE)
    joint_vel = obs_tp1[..., SLICES["joint_vel"]] / float(hrlg.DOF_VEL_SCALE)
    prev_joint_vel = obs_t[..., SLICES["joint_vel"]] / float(hrlg.DOF_VEL_SCALE)
    last_act = obs_t[..., SLICES["last_act"]]

    qpos_root_tp1 = _normalize_root_quat(qpos_root_tp1)
    quat = qpos_root_tp1[..., 3:7]
    root_lin_world = qvel_root_tp1[..., 0:3]
    base_lin_vel = _rotate_world_to_body_torch(quat, root_lin_world)
    base_height = qpos_root_tp1[..., 2]

    default_pose = context["default_pose"].to(dtype=obs_t.dtype, device=obs_t.device)
    hip_indices = context["hip_indices"].to(device=obs_t.device)
    knee_indices = context["knee_indices"].to(device=obs_t.device)
    soft_lowers = context["soft_lowers"].to(dtype=obs_t.dtype, device=obs_t.device)
    soft_uppers = context["soft_uppers"].to(dtype=obs_t.dtype, device=obs_t.device)
    joint_pos = joint_pos_rel + default_pose

    lin_vel_error = torch.sum(torch.square(command[..., :2] - base_lin_vel[..., :2]), dim=-1)
    ang_vel_error = torch.square(command[..., 2] - gyro[..., 2])
    hip_error = joint_pos[..., hip_indices] - default_pose[hip_indices]
    hip_weight_default = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=obs_t.dtype, device=obs_t.device)
    hip_weight_lateral = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=obs_t.dtype, device=obs_t.device)
    hip_weight = torch.where(command[..., 1:2] > 0.1, hip_weight_lateral, hip_weight_default)
    out_of_limits = -torch.clamp(joint_pos - soft_lowers, max=0.0)
    out_of_limits = out_of_limits + torch.clamp(joint_pos - soft_uppers, min=0.0)
    gravity_z = torch.clamp(-projected_gravity[..., 2], -1.0, 1.0)
    tilt = torch.abs(torch.acos(gravity_z))
    bad_orientation = tilt > float(context["bad_orientation_limit_rad"])
    if float(context["base_height_done_threshold"]) <= 0:
        bad_height = torch.zeros_like(bad_orientation)
    else:
        bad_height = base_height < float(context["base_height_done_threshold"])
    finite = (
        torch.isfinite(obs_t).all(dim=-1)
        & torch.isfinite(obs_tp1).all(dim=-1)
        & torch.isfinite(qpos_root_tp1).all(dim=-1)
        & torch.isfinite(qvel_root_tp1).all(dim=-1)
        & torch.isfinite(act_t).all(dim=-1)
    )
    done = (bad_orientation | bad_height | (~finite)).to(dtype=obs_t.dtype)

    components = {
        "tracking_lin_vel": torch.exp(-lin_vel_error / float(context["tracking_sigma"])),
        "tracking_ang_vel": torch.exp(-ang_vel_error / float(context["tracking_sigma"])),
        "orientation": torch.sum(torch.square(projected_gravity[..., :2]), dim=-1),
        "ang_vel_xy": torch.sum(torch.square(gyro[..., :2]), dim=-1),
        "joint_deviation_hip": torch.sum(torch.abs(hip_error) * hip_weight, dim=-1),
        "joint_deviation_knee": torch.sum(
            torch.abs(joint_pos[..., knee_indices] - default_pose[knee_indices]),
            dim=-1,
        ),
        "pose": torch.sum(torch.square(joint_pos - default_pose), dim=-1),
        "dof_pos_limits": torch.sum(out_of_limits, dim=-1),
        "stand_still": torch.sum(torch.abs(joint_pos - default_pose), dim=-1)
        * (torch.linalg.norm(command, dim=-1) < 0.01).to(dtype=obs_t.dtype),
        "action_rate": torch.sum(torch.square(act_t - last_act), dim=-1),
        "base_height": torch.square(base_height - float(context["base_height_target"])),
        "dof_acc": torch.sum(torch.square((joint_vel - prev_joint_vel) / float(context["dt"])), dim=-1),
        "termination": done,
    }
    reward = torch.zeros_like(done)
    for name, value in components.items():
        reward = reward + float(context["reward_scales"].get(name, 0.0)) * value * float(context["dt"])
    return reward, done, components


def _load_obs_stats_from_checkpoint(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, bool] | None:
    stats_path = checkpoint / "obs_stats.npz"
    if stats_path.exists():
        with np.load(stats_path) as stats:
            return (
                stats["mean"].astype(np.float32),
                np.maximum(stats["std"].astype(np.float32), 1e-6),
                bool(np.asarray(stats["enabled"]).item()),
            )

    actor_path = checkpoint / "actor.pt"
    if not actor_path.exists():
        return None
    ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
    if "obs_mean" not in ckpt or "obs_std" not in ckpt:
        return None
    obs_mean = ckpt["obs_mean"]
    obs_std = ckpt["obs_std"]
    if isinstance(obs_mean, torch.Tensor):
        obs_mean = obs_mean.detach().cpu().numpy()
    if isinstance(obs_std, torch.Tensor):
        obs_std = obs_std.detach().cpu().numpy()
    return (
        np.asarray(obs_mean, dtype=np.float32),
        np.maximum(np.asarray(obs_std, dtype=np.float32), 1e-6),
        bool(ckpt.get("obs_normalization", True)),
    )


def _resolve_dynamics_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "dynamics_ensemble.pt").exists():
        return path
    for name in ("best_model", "final"):
        candidate = path / name
        if (candidate / "dynamics_ensemble.pt").exists():
            return candidate
    candidates = [child for child in path.iterdir() if child.is_dir() and (child / "dynamics_ensemble.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"Could not find dynamics_ensemble.pt under dynamics checkpoint path: {path}")
    return max(candidates, key=lambda child: child.stat().st_mtime)


def _load_obs_stats_from_dynamics_checkpoint(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    stats_path = checkpoint / "obs_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing obs_stats.npz in dynamics checkpoint: {checkpoint}")
    with np.load(stats_path) as stats:
        return (
            stats["mean"].astype(np.float32),
            np.maximum(stats["std"].astype(np.float32), 1e-6),
            bool(np.asarray(stats["enabled"]).item()),
        )


def _load_dynamics_checkpoint(
    dynamics: DynamicsEnsemble,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    ckpt_path = checkpoint / "dynamics_ensemble.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing dynamics checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if bool(ckpt.get("predict_root", dynamics.predict_root)) != bool(dynamics.predict_root):
        raise ValueError(
            f"Dynamics root-head mismatch: checkpoint={ckpt.get('predict_root')} current={dynamics.predict_root}"
        )
    if bool(ckpt.get("predict_reward", dynamics.predict_reward)) != bool(dynamics.predict_reward):
        raise ValueError(
            f"Dynamics reward-head mismatch: checkpoint={ckpt.get('predict_reward')} current={dynamics.predict_reward}"
        )
    if bool(ckpt.get("predict_done", dynamics.predict_done)) != bool(dynamics.predict_done):
        raise ValueError(
            f"Dynamics done-head mismatch: checkpoint={ckpt.get('predict_done')} current={dynamics.predict_done}"
        )
    dynamics.load_state_dict(ckpt["network_state_dict"])
    return ckpt


def _validate_dynamics_checkpoint_split(checkpoint: Path, split_info: dict[str, Any]) -> None:
    split_path = checkpoint / "split_info.json"
    if not split_path.exists():
        return
    with split_path.open("r", encoding="utf-8") as f:
        checkpoint_split = json.load(f)
    keys = ("train_trajectories", "val_trajectories", "train_transitions", "val_transitions")
    mismatches = {
        key: {"checkpoint": checkpoint_split.get(key), "current": split_info.get(key)}
        for key in keys
        if checkpoint_split.get(key) != split_info.get(key)
    }
    if mismatches:
        raise ValueError(
            "Dynamics checkpoint split does not match the current dataset split. "
            f"checkpoint={checkpoint} mismatches={mismatches}"
        )


def _make_actor(device: torch.device) -> FlashSACActor:
    return FlashSACActor(
        num_blocks=2,
        input_dim=hrlg.POLICY_OBS_SIZE,
        hidden_dim=128,
        action_dim=hrlg.ACTION_SIZE,
    ).to(device)


def _load_actor_checkpoint(actor: FlashSACActor, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    actor_path = checkpoint / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Missing BC actor checkpoint: {actor_path}")
    ckpt = torch.load(actor_path, map_location=device, weights_only=False)
    state_dict = ckpt["network_state_dict"]
    if state_dict and next(iter(state_dict)).startswith("_orig_mod."):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    actor.load_state_dict(state_dict)
    return ckpt


def _make_network_bundle(
    module: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    use_weight_normalization: bool,
    ema_source: Network | None = None,
    tau: float | None = None,
) -> Network:
    optimizer = None
    if ema_source is None:
        optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    return Network(
        network=module,
        optimizer=optimizer,
        scheduler=None,
        compile_network=False,
        use_weight_normalization=use_weight_normalization,
        ema_source=ema_source,
        ema_tau=tau,
    )


def _load_transitions(
    path: Path,
    *,
    val_ratio: float,
    seed: int,
    limit_trajectories: int | None,
    limit_transitions: int | None,
    reward_source: str,
    reward_relabel_batch_size: int,
) -> tuple[TransitionArrays, TransitionArrays, TransitionSplit, dict[str, Any]]:
    with np.load(path) as data:
        states = data["state"].astype(np.float32, copy=False)
        actions = data["action"].astype(np.float32, copy=False)
        qpos_root = data["qpos_root"].astype(np.float32, copy=False) if "qpos_root" in data else None
        qvel_root = data["qvel_root"].astype(np.float32, copy=False) if "qvel_root" in data else None
        dataset_reward = data["reward"].astype(np.float32, copy=False) if "reward" in data else None
        dataset_done = data["done"].astype(np.float32, copy=False) if "done" in data else None
        metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data else None

    if states.ndim != 3 or actions.ndim != 3:
        raise ValueError(f"Expected trajectory arrays, got state={states.shape}, action={actions.shape}.")
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError(f"Mismatched trajectory/time dimensions: state={states.shape}, action={actions.shape}.")
    if states.shape[-1] != hrlg.POLICY_OBS_SIZE or actions.shape[-1] != hrlg.ACTION_SIZE:
        raise ValueError(
            f"Expected state/action dims {(hrlg.POLICY_OBS_SIZE, hrlg.ACTION_SIZE)}, "
            f"got {(states.shape[-1], actions.shape[-1])}."
        )

    rng = np.random.default_rng(seed)
    trajectory_indices = np.arange(states.shape[0])
    if limit_trajectories is not None:
        if limit_trajectories < 2:
            raise ValueError("--limit-trajectories must leave at least train and val trajectories.")
        trajectory_indices = trajectory_indices[: min(limit_trajectories, len(trajectory_indices))]
    shuffled = trajectory_indices.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_traj = np.sort(shuffled[:val_count])
    train_traj = np.sort(shuffled[val_count:])

    if dataset_reward is not None:
        if dataset_reward.shape == states.shape[:2]:
            dataset_reward = dataset_reward[:, :-1]
        if dataset_reward.shape != (states.shape[0], states.shape[1] - 1):
            raise ValueError(
                "Dataset reward must have shape (N,T-1) or padded shape (N,T), "
                f"got reward={dataset_reward.shape}, expected {(states.shape[0], states.shape[1] - 1)}."
            )
    if dataset_done is not None:
        if dataset_done.shape == states.shape[:2]:
            dataset_done = dataset_done[:, :-1]
        if dataset_done.shape != (states.shape[0], states.shape[1] - 1):
            raise ValueError(
                "Dataset done must have shape (N,T-1) or padded shape (N,T), "
                f"got done={dataset_done.shape}, expected {(states.shape[0], states.shape[1] - 1)}."
            )

    def build(traj: np.ndarray) -> TransitionArrays:
        obs = states[traj, :-1].reshape(-1, states.shape[-1]).copy()
        act = actions[traj, :-1].reshape(-1, actions.shape[-1]).copy()
        next_obs = states[traj, 1:].reshape(-1, states.shape[-1]).copy()
        steps = np.broadcast_to(np.arange(states.shape[1] - 1, dtype=np.int32), (len(traj), states.shape[1] - 1))
        traj_ids = np.broadcast_to(traj[:, None].astype(np.int32), (len(traj), states.shape[1] - 1))
        qpos = None if qpos_root is None else qpos_root[traj, :-1].reshape(-1, qpos_root.shape[-1]).copy()
        qvel = None if qvel_root is None else qvel_root[traj, :-1].reshape(-1, qvel_root.shape[-1]).copy()
        next_qpos = None if qpos_root is None else qpos_root[traj, 1:].reshape(-1, qpos_root.shape[-1]).copy()
        next_qvel = None if qvel_root is None else qvel_root[traj, 1:].reshape(-1, qvel_root.shape[-1]).copy()
        reward = (
            np.zeros(obs.shape[0], dtype=np.float32)
            if dataset_reward is None
            else dataset_reward[traj].reshape(-1).copy()
        )
        terminated = (
            np.zeros(obs.shape[0], dtype=np.float32) if dataset_done is None else dataset_done[traj].reshape(-1).copy()
        )
        return TransitionArrays(
            observation=obs,
            action=act,
            next_observation=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=np.zeros(obs.shape[0], dtype=np.float32),
            trajectory=traj_ids.reshape(-1).copy(),
            step=steps.reshape(-1).copy(),
            qpos_root=qpos,
            qvel_root=qvel,
            next_qpos_root=next_qpos,
            next_qvel_root=next_qvel,
        )

    train = build(train_traj)
    val = build(val_traj)

    if limit_transitions is not None:
        if limit_transitions < 2:
            raise ValueError("--limit-transitions must be >= 2.")
        train_limit = min(limit_transitions, train.size)
        val_limit = min(max(1, limit_transitions // 5), val.size)
        train = train.select(np.sort(rng.choice(train.size, size=train_limit, replace=False)))
        val = val.select(np.sort(rng.choice(val.size, size=val_limit, replace=False)))

    relabel_info: dict[str, Any] = {
        "reward_source": reward_source,
        "reward_available": False,
        "reward_strict": False,
        "done_available": False,
        "strict_mopo": False,
        "missing_reward_state": [],
    }

    if reward_source == "observable":
        if dataset_reward is None or dataset_done is None:
            raise ValueError("--reward-source observable requires reward and done keys in the dataset.")
        source_metadata: dict[str, Any] | None = None
        if metadata_json is not None:
            try:
                source_metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                source_metadata = {"raw_metadata_json": metadata_json}
        relabel_info.update(
            {
                "reward_source": "observable_dataset_keys",
                "reward_available": True,
                "reward_strict": False,
                "done_available": True,
                "strict_mopo": False,
                "missing_reward_state": [
                    "torques",
                    "energy",
                    "contact_force",
                    "feet_air_time",
                    "feet_slip",
                    "feet_phase",
                    "feet_height",
                    "feet_clearance",
                    "collision",
                ],
                "reward_mean": float(np.mean(np.concatenate([train.reward, val.reward]))),
                "reward_std": float(np.std(np.concatenate([train.reward, val.reward]))),
                "reward_min": float(np.min(np.concatenate([train.reward, val.reward]))),
                "reward_max": float(np.max(np.concatenate([train.reward, val.reward]))),
                "done_count": int(np.sum(np.concatenate([train.terminated, val.terminated]) > 0.5)),
                "source_metadata": source_metadata,
            }
        )
    elif reward_source == "relabel_env":
        if train.qpos_root is None or train.qvel_root is None or val.qpos_root is None or val.qvel_root is None:
            raise ValueError("--reward-source relabel_env requires qpos_root and qvel_root in the dataset.")
        for arrays in (train, val):
            reward, done, info = _relabel_rewards_with_env(
                arrays,
                batch_size=reward_relabel_batch_size,
                seed=seed,
            )
            arrays.reward = reward
            arrays.terminated = done
        relabel_info.update(info)
    elif reward_source == "zero":
        relabel_info.update(
            {
                "reward_source": "zero",
                "reward_available": False,
                "reward_strict": False,
                "done_available": False,
                "strict_mopo": False,
                "missing_reward_state": ["reward", "done"],
            }
        )
    else:
        raise ValueError(f"Unsupported reward source: {reward_source}")

    split = TransitionSplit(
        train_trajectories=[int(x) for x in train_traj.tolist()],
        val_trajectories=[int(x) for x in val_traj.tolist()],
        train_transitions=train.size,
        val_transitions=val.size,
        obs_dim=int(states.shape[-1]),
        action_dim=int(actions.shape[-1]),
        reward_available=bool(relabel_info["reward_available"]),
        reward_strict=bool(relabel_info["reward_strict"]),
        done_available=bool(relabel_info["done_available"]),
        strict_mopo=bool(relabel_info["strict_mopo"]),
    )
    return train, val, split, relabel_info


def _relabel_rewards_with_env(
    arrays: TransitionArrays,
    *,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import jax
    import jax.numpy as jp
    from mujoco_playground._src import mjx_env

    from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import G1JoystickFlatTerrainHRLG, default_config

    if arrays.qpos_root is None or arrays.qvel_root is None:
        raise ValueError("qpos_root/qvel_root are required for env reward relabeling.")

    cfg = default_config()
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.push_config.magnitude_range = [0.0, 0.0]
    cfg.event_domain_randomization = False
    env = G1JoystickFlatTerrainHRLG(config=cfg)
    joint_default_pose = np.asarray(env._joint_default_pose(), dtype=np.float32)
    cmd_scale = np.asarray(hrlg.CMD_SCALE, dtype=np.float32)

    def init_data(qpos, qvel, ctrl):
        return mjx_env.init(env.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

    init_data_fn = jax.jit(jax.vmap(init_data))
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    for batch_idx, start in enumerate(range(0, arrays.size, batch_size), start=1):
        end = min(start + batch_size, arrays.size)
        obs_t = arrays.observation[start:end]
        action_t = arrays.action[start:end]
        batch = end - start

        qpos = np.zeros((batch, env.mjx_model.nq), dtype=np.float32)
        qvel = np.zeros((batch, env.mjx_model.nv), dtype=np.float32)
        qpos[:, :7] = arrays.qpos_root[start:end]
        qpos[:, 7:] = obs_t[:, SLICES["joint_pos"]] / hrlg.DOF_POS_SCALE + joint_default_pose
        qvel[:, :6] = arrays.qvel_root[start:end]
        qvel[:, 6:] = obs_t[:, SLICES["joint_vel"]] / hrlg.DOF_VEL_SCALE

        command = obs_t[:, SLICES["command"]] / cmd_scale
        last_act = obs_t[:, SLICES["last_act"]]
        ctrl = joint_default_pose[None, :] + last_act * hrlg.ACTION_SCALE

        gait = obs_t[:, SLICES["gait"]]
        gait_angle = np.arctan2(gait[:, 0], gait[:, 1]).astype(np.float32)
        phase = np.stack([gait_angle, gait_angle + np.pi], axis=-1).astype(np.float32)
        phase_dt = np.full((batch, 1), 2.0 * np.pi * env.dt / hrlg.GAIT_PHASE_CYCLE, dtype=np.float32)

        template = reset_fn(jax.random.split(jax.random.PRNGKey(seed + batch_idx), batch))
        data = init_data_fn(jp.asarray(qpos), jp.asarray(qvel), jp.asarray(ctrl))
        info = dict(template.info)
        info["command"] = jp.asarray(command, dtype=jp.float32)
        info["last_act"] = jp.asarray(last_act, dtype=jp.float32)
        info["last_last_act"] = jp.zeros_like(info["last_act"])
        info["motor_targets"] = jp.asarray(ctrl, dtype=jp.float32)
        info["joint_default_pose"] = jp.asarray(np.broadcast_to(joint_default_pose, (batch, hrlg.ACTION_SIZE)))
        info["policy_step"] = jp.asarray(arrays.step[start:end], dtype=jp.int32)
        info["step"] = jp.asarray(arrays.step[start:end] % 501, dtype=jp.int32)
        info["phase"] = jp.asarray(phase, dtype=jp.float32)
        info["phase_dt"] = jp.asarray(phase_dt, dtype=jp.float32)
        info["feet_air_time"] = jp.zeros((batch, 2), dtype=jp.float32)
        info["last_contact"] = jp.zeros((batch, 2), dtype=bool)
        info["swing_peak"] = jp.zeros((batch, 2), dtype=jp.float32)
        reconstructed = template.replace(data=data, info=info)

        pred = step_fn(reconstructed, jp.asarray(action_t, dtype=jp.float32))
        rewards.append(np.asarray(pred.reward, dtype=np.float32))
        dones.append(np.asarray(pred.done, dtype=np.float32))

    reward = np.concatenate(rewards, axis=0).astype(np.float32)
    done = np.concatenate(dones, axis=0).astype(np.float32)
    info = {
        "reward_source": "env_relabel_from_state_qpos_root_qvel_root",
        "reward_available": True,
        "reward_strict": False,
        "done_available": True,
        "strict_mopo": False,
        "missing_reward_state": [
            "full_qpos",
            "full_qvel",
            "phase_dt_from_reset_gait_frequency",
            "feet_air_time_history",
            "last_contact_history",
            "swing_peak_history",
            "last_last_action",
            "domain_randomization_state",
        ],
        "reward_mean": float(np.mean(reward)),
        "reward_std": float(np.std(reward)),
        "reward_min": float(np.min(reward)),
        "reward_max": float(np.max(reward)),
        "done_count": int(np.sum(done > 0.5)),
    }
    return reward, done, info


def _normalize(arrays: TransitionArrays, obs_mean: np.ndarray, obs_std: np.ndarray) -> TransitionArrays:
    return TransitionArrays(
        observation=((arrays.observation - obs_mean) / obs_std).astype(np.float32),
        action=arrays.action,
        next_observation=((arrays.next_observation - obs_mean) / obs_std).astype(np.float32),
        reward=arrays.reward,
        terminated=arrays.terminated,
        truncated=arrays.truncated,
        trajectory=arrays.trajectory,
        step=arrays.step,
        qpos_root=arrays.qpos_root,
        qvel_root=arrays.qvel_root,
        next_qpos_root=arrays.next_qpos_root,
        next_qvel_root=arrays.next_qvel_root,
    )


def _sample_indices(size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, size, (batch_size,), device=device)


def _batch_from_arrays(
    arrays: TransitionArrays,
    indices: torch.Tensor,
    device: torch.device,
    *,
    reward_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    idx = indices.detach().cpu().numpy()
    obs = torch.as_tensor(arrays.observation[idx], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(arrays.next_observation[idx], dtype=torch.float32, device=device)
    action = torch.as_tensor(arrays.action[idx], dtype=torch.float32, device=device)
    reward = torch.as_tensor(arrays.reward[idx], dtype=torch.float32, device=device) * float(reward_scale)
    terminated = torch.as_tensor(arrays.terminated[idx], dtype=torch.float32, device=device)
    truncated = torch.as_tensor(arrays.truncated[idx], dtype=torch.float32, device=device)
    return {
        "observation": obs,
        "actor_observation": obs,
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "next_observation": next_obs,
        "actor_next_observation": next_obs,
    }


def _concat_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batches[0].keys()
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def _make_real_batch(
    arrays: TransitionArrays,
    batch_size: int,
    device: torch.device,
    *,
    reward_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    return _batch_from_arrays(
        arrays,
        _sample_indices(arrays.size, batch_size, device),
        device,
        reward_scale=reward_scale,
    )


def _force_last_action_slice(
    next_obs_norm: torch.Tensor,
    action: torch.Tensor,
    *,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return next_obs_norm
    next_obs_norm = next_obs_norm.clone()
    slc = SLICES["last_act"]
    next_obs_norm[:, slc] = (action - obs_mean[slc]) / obs_std[slc]
    return next_obs_norm


@torch.no_grad()
def _generate_model_rollouts(
    *,
    actor: Network,
    dynamics: DynamicsEnsemble,
    start_obs: torch.Tensor,
    start_qpos_root: torch.Tensor | None,
    start_qvel_root: torch.Tensor | None,
    rollout_horizon: int,
    mopo_penalty_coef: float,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    force_last_action: bool,
    done_threshold: float,
    reward_scale: float,
    formula_reward: bool,
    formula_done: bool,
    observable_reward_context: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    device = start_obs.device
    observations = []
    actions = []
    next_observations = []
    rewards = []
    terminateds = []
    truncateds = []
    uncertainties = []
    raw_rewards = []
    obs = start_obs
    qpos_root = None if start_qpos_root is None else start_qpos_root
    qvel_root = None if start_qvel_root is None else start_qvel_root
    active = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
    ensemble_size = len(dynamics.members)
    root_uncertainties = []
    if dynamics.predict_root and (qpos_root is None or qvel_root is None):
        raise ValueError("Root-aware rollout requires start_qpos_root and start_qvel_root.")
    if (formula_reward or formula_done) and (not dynamics.predict_root or observable_reward_context is None):
        raise ValueError("Formula reward/done rollout requires root-aware dynamics and reward context.")

    for horizon_step in range(rollout_horizon):
        if not bool(torch.any(active)):
            break
        obs_active = obs[active]
        qpos_active = None if qpos_root is None else qpos_root[active]
        qvel_active = None if qvel_root is None else qvel_root[active]
        mean, _ = actor.apply("get_mean_and_std", observations=obs_active, training=False)
        action = torch.tanh(mean)
        pred = dynamics(obs_active, action, qpos_active, qvel_active)
        delta, delta_qpos_root, delta_qvel_root, reward_pred, done_logit = dynamics.split_prediction(pred)
        next_per_model = obs_active.unsqueeze(0) + delta
        next_per_model = torch.stack(
            [
                _force_last_action_slice(
                    next_per_model[i],
                    action,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    enabled=force_last_action,
                )
                for i in range(ensemble_size)
            ],
            dim=0,
        )
        obs_uncertainty = next_per_model.var(dim=0).mean(dim=-1)
        root_uncertainty = torch.zeros_like(obs_uncertainty)
        next_qpos_per_model = None
        next_qvel_per_model = None
        if dynamics.predict_root:
            assert qpos_active is not None and qvel_active is not None
            assert delta_qpos_root is not None and delta_qvel_root is not None
            next_qpos_per_model = _normalize_root_quat(qpos_active.unsqueeze(0) + delta_qpos_root)
            next_qvel_per_model = qvel_active.unsqueeze(0) + delta_qvel_root
            root_uncertainty = next_qpos_per_model.var(dim=0).mean(dim=-1)
            root_uncertainty = root_uncertainty + next_qvel_per_model.var(dim=0).mean(dim=-1)
        uncertainty = obs_uncertainty + root_uncertainty

        member_idx = torch.randint(0, ensemble_size, (obs_active.shape[0],), device=device)
        row_idx = torch.arange(obs_active.shape[0], device=device)
        next_sample = next_per_model[member_idx, row_idx]
        next_qpos_sample = None
        next_qvel_sample = None
        if next_qpos_per_model is not None and next_qvel_per_model is not None:
            next_qpos_sample = next_qpos_per_model[member_idx, row_idx]
            next_qvel_sample = next_qvel_per_model[member_idx, row_idx]

        formula_reward_unscaled = None
        formula_done_tensor = None
        if formula_reward or formula_done:
            assert next_qpos_sample is not None and next_qvel_sample is not None
            assert observable_reward_context is not None
            formula_reward_unscaled, formula_done_tensor, _ = _compute_observable_reward_torch(
                obs_t_norm=obs_active,
                act_t=action,
                obs_tp1_norm=next_sample,
                qpos_root_tp1=next_qpos_sample,
                qvel_root_tp1=next_qvel_sample,
                obs_mean=obs_mean,
                obs_std=obs_std,
                context=observable_reward_context,
            )

        if formula_reward and formula_reward_unscaled is not None:
            reward_mean = formula_reward_unscaled
        elif reward_pred is None:
            reward_mean = torch.zeros_like(uncertainty)
        else:
            reward_mean = reward_pred.mean(dim=0)
        penalized_reward_unscaled = reward_mean - mopo_penalty_coef * uncertainty
        penalized_reward = penalized_reward_unscaled * float(reward_scale)

        if formula_done and formula_done_tensor is not None:
            done = formula_done_tensor
        elif done_logit is None:
            done = torch.zeros_like(penalized_reward)
        else:
            done = (torch.sigmoid(done_logit.mean(dim=0)) > done_threshold).float()
        truncated = torch.zeros_like(done)
        if horizon_step == rollout_horizon - 1:
            truncated = 1.0 - done

        observations.append(obs_active)
        actions.append(action)
        next_observations.append(next_sample)
        rewards.append(penalized_reward)
        terminateds.append(done)
        truncateds.append(truncated)
        uncertainties.append(uncertainty)
        root_uncertainties.append(root_uncertainty)
        raw_rewards.append(reward_mean)

        obs_next = obs.clone()
        active_indices = active.nonzero(as_tuple=False).squeeze(-1)
        obs_next[active_indices] = next_sample
        obs = obs_next
        if qpos_root is not None and next_qpos_sample is not None:
            qpos_next = qpos_root.clone()
            qpos_next[active_indices] = next_qpos_sample
            qpos_root = qpos_next
        if qvel_root is not None and next_qvel_sample is not None:
            qvel_next = qvel_root.clone()
            qvel_next[active_indices] = next_qvel_sample
            qvel_root = qvel_next
        still_active = done <= 0.5
        active_next = active.clone()
        active_next[active_indices] = still_active
        active = active_next

    if not observations:
        raise RuntimeError("Model rollout produced no synthetic transitions.")

    synthetic = {
        "observation": torch.cat(observations, dim=0),
        "actor_observation": torch.cat(observations, dim=0),
        "action": torch.cat(actions, dim=0),
        "reward": torch.cat(rewards, dim=0),
        "terminated": torch.cat(terminateds, dim=0),
        "truncated": torch.cat(truncateds, dim=0),
        "next_observation": torch.cat(next_observations, dim=0),
        "actor_next_observation": torch.cat(next_observations, dim=0),
    }
    uncertainty_tensor = torch.cat(uncertainties, dim=0)
    root_uncertainty_tensor = (
        torch.cat(root_uncertainties, dim=0) if root_uncertainties else torch.zeros_like(uncertainty_tensor)
    )
    raw_reward_tensor = torch.cat(raw_rewards, dim=0)
    metrics = {
        "synthetic_transition_count": float(synthetic["observation"].shape[0]),
        "uncertainty_mean": float(uncertainty_tensor.mean().item()),
        "uncertainty_max": float(uncertainty_tensor.max().item()),
        "root_uncertainty_mean": float(root_uncertainty_tensor.mean().item()),
        "synthetic_reward_mean": float(synthetic["reward"].mean().item()),
        "synthetic_raw_reward_mean": float(raw_reward_tensor.mean().item()),
        "synthetic_penalized_unscaled_reward_mean": float((synthetic["reward"] / float(reward_scale)).mean().item()),
        "synthetic_done_fraction": float(synthetic["terminated"].mean().item()),
        "formula_reward": float(bool(formula_reward)),
        "formula_done": float(bool(formula_done)),
    }
    return synthetic, metrics


def _empty_ages_metrics() -> dict[str, float]:
    return {
        "ages_candidate_return_mean": float("nan"),
        "ages_candidate_return_std": float("nan"),
        "ages_selected_return_mean": float("nan"),
        "ages_selected_adv_mean": float("nan"),
        "ages_selected_adv_min": float("nan"),
        "ages_selected_adv_max": float("nan"),
        "ages_accept_rate": 0.0,
        "ages_buffer_size": 0.0,
        "ages_uncertainty_mean": float("nan"),
        "ages_generation_wall_time": 0.0,
        "ages_sim_ratio": float("nan"),
        "ages_select_ratio": float("nan"),
        "ages_action_temperature": float("nan"),
    }


def _batch_size(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["observation"].shape[0])


def _slice_torch_batch(batch: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[mask] for key, value in batch.items()}


def _select_torch_batch_indices(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in batch.items()}


def _ages_buffer_size(buffer: list[tuple[int, dict[str, torch.Tensor]]]) -> int:
    return sum(_batch_size(batch) for _, batch in buffer)


def _trim_ages_buffer(
    buffer: list[tuple[int, dict[str, torch.Tensor]]],
    *,
    current_epoch: int,
    retain_epochs: int,
    max_size: int,
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    if retain_epochs > 0:
        min_epoch = current_epoch - retain_epochs + 1
        buffer = [(epoch, batch) for epoch, batch in buffer if epoch >= min_epoch]
    while len(buffer) > 1 and _ages_buffer_size(buffer) > max_size:
        buffer.pop(0)
    if buffer and _ages_buffer_size(buffer) > max_size > 0:
        epoch, batch = buffer[-1]
        size = _batch_size(batch)
        if size > max_size:
            idx = torch.randperm(size, device=batch["observation"].device)[:max_size]
            buffer[-1] = (epoch, _select_torch_batch_indices(batch, idx))
    return buffer


def _sample_from_ages_buffer(
    buffer: list[tuple[int, dict[str, torch.Tensor]]],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if not buffer:
        raise RuntimeError("Cannot sample from an empty AGES buffer.")
    batch = _concat_batches([item for _, item in buffer]) if len(buffer) > 1 else buffer[0][1]
    return _sample_from_torch_batch(batch, batch_size)


@torch.no_grad()
def _generate_ages_rollouts(
    *,
    actor: Network,
    dynamics: DynamicsEnsemble,
    arrays: TransitionArrays,
    device: torch.device,
    num_dataset_states: int,
    trajectories_per_state: int,
    sim_rollout_length: int,
    inject_rollout_length: int,
    action_temperature: float,
    select_ratio: float,
    select_mode: str,
    mopo_penalty_coef: float,
    reward_scale: float,
    discount: float,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    force_last_action: bool,
    done_threshold: float,
    formula_reward: bool,
    formula_done: bool,
    observable_reward_context: dict[str, Any] | None,
    score_use_uncertainty: bool,
    score_use_mopo_penalty: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    started = time.perf_counter()
    if arrays.qpos_root is None or arrays.qvel_root is None:
        raise ValueError("AGES root-aware rollout requires qpos_root and qvel_root in the dataset.")
    if sim_rollout_length <= 0:
        raise ValueError("--ages-sim-rollout-length must be positive.")
    if trajectories_per_state <= 0:
        raise ValueError("--ages-trajectories-per-state must be positive.")
    if not (0.0 < select_ratio <= 1.0):
        raise ValueError("--ages-select-ratio must be in (0, 1].")
    if (formula_reward or formula_done) and (not dynamics.predict_root or observable_reward_context is None):
        raise ValueError("AGES formula reward/done requires root-aware dynamics and reward context.")

    state_count = min(int(num_dataset_states), arrays.size)
    base_idx = torch.randint(0, arrays.size, (state_count,), device=device)
    base_np = base_idx.detach().cpu().numpy()
    repeat = int(trajectories_per_state)
    start_obs = torch.as_tensor(arrays.observation[base_np], dtype=torch.float32, device=device).repeat_interleave(
        repeat, dim=0
    )
    start_qpos = torch.as_tensor(arrays.qpos_root[base_np], dtype=torch.float32, device=device).repeat_interleave(
        repeat, dim=0
    )
    start_qvel = torch.as_tensor(arrays.qvel_root[base_np], dtype=torch.float32, device=device).repeat_interleave(
        repeat, dim=0
    )
    total_candidates = int(start_obs.shape[0])
    state_ids = torch.arange(state_count, device=device).repeat_interleave(repeat)
    candidate_ids = torch.arange(total_candidates, device=device)

    obs = start_obs
    qpos_root = start_qpos
    qvel_root = start_qvel
    active = torch.ones(total_candidates, dtype=torch.bool, device=device)
    returns = torch.zeros(total_candidates, dtype=torch.float32, device=device)
    score_uncertainties = torch.zeros(total_candidates, dtype=torch.float32, device=device)

    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    next_observations: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminateds: list[torch.Tensor] = []
    truncateds: list[torch.Tensor] = []
    transition_candidate_ids: list[torch.Tensor] = []
    transition_steps: list[torch.Tensor] = []
    transition_uncertainties: list[torch.Tensor] = []
    raw_rewards: list[torch.Tensor] = []

    ensemble_size = len(dynamics.members)
    inject_horizon = int(inject_rollout_length) if inject_rollout_length and inject_rollout_length > 0 else sim_rollout_length
    inject_horizon = min(inject_horizon, sim_rollout_length)

    for horizon_step in range(sim_rollout_length):
        if not bool(torch.any(active)):
            break
        obs_active = obs[active]
        qpos_active = qpos_root[active]
        qvel_active = qvel_root[active]
        active_candidate_ids = candidate_ids[active]

        mean, std = actor.apply("get_mean_and_std", observations=obs_active, training=False)
        if action_temperature > 0:
            action = torch.tanh(mean + float(action_temperature) * std * torch.randn_like(std))
        else:
            action = torch.tanh(mean)

        pred = dynamics(obs_active, action, qpos_active, qvel_active)
        delta, delta_qpos_root, delta_qvel_root, reward_pred, done_logit = dynamics.split_prediction(pred)
        next_per_model = obs_active.unsqueeze(0) + delta
        next_per_model = torch.stack(
            [
                _force_last_action_slice(
                    next_per_model[i],
                    action,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    enabled=force_last_action,
                )
                for i in range(ensemble_size)
            ],
            dim=0,
        )
        obs_uncertainty = next_per_model.var(dim=0).mean(dim=-1)
        assert delta_qpos_root is not None and delta_qvel_root is not None
        next_qpos_per_model = _normalize_root_quat(qpos_active.unsqueeze(0) + delta_qpos_root)
        next_qvel_per_model = qvel_active.unsqueeze(0) + delta_qvel_root
        root_uncertainty = next_qpos_per_model.var(dim=0).mean(dim=-1)
        root_uncertainty = root_uncertainty + next_qvel_per_model.var(dim=0).mean(dim=-1)
        uncertainty = obs_uncertainty + root_uncertainty

        member_idx = torch.randint(0, ensemble_size, (obs_active.shape[0],), device=device)
        row_idx = torch.arange(obs_active.shape[0], device=device)
        next_sample = next_per_model[member_idx, row_idx]
        next_qpos_sample = next_qpos_per_model[member_idx, row_idx]
        next_qvel_sample = next_qvel_per_model[member_idx, row_idx]

        formula_reward_unscaled = None
        formula_done_tensor = None
        if formula_reward or formula_done:
            assert observable_reward_context is not None
            formula_reward_unscaled, formula_done_tensor, _ = _compute_observable_reward_torch(
                obs_t_norm=obs_active,
                act_t=action,
                obs_tp1_norm=next_sample,
                qpos_root_tp1=next_qpos_sample,
                qvel_root_tp1=next_qvel_sample,
                obs_mean=obs_mean,
                obs_std=obs_std,
                context=observable_reward_context,
            )

        if formula_reward and formula_reward_unscaled is not None:
            reward_mean = formula_reward_unscaled
        elif reward_pred is None:
            reward_mean = torch.zeros_like(uncertainty)
        else:
            reward_mean = reward_pred.mean(dim=0)

        score_reward = reward_mean
        if score_use_uncertainty or score_use_mopo_penalty:
            score_reward = score_reward - float(mopo_penalty_coef) * uncertainty
        replay_reward = reward_mean - float(mopo_penalty_coef) * uncertainty if score_use_mopo_penalty else reward_mean
        replay_reward_scaled = replay_reward * float(reward_scale)
        returns[active_candidate_ids] += (float(discount) ** horizon_step) * score_reward * float(reward_scale)
        score_uncertainties[active_candidate_ids] += uncertainty

        if formula_done and formula_done_tensor is not None:
            done = formula_done_tensor
        elif done_logit is None:
            done = torch.zeros_like(replay_reward_scaled)
        else:
            done = (torch.sigmoid(done_logit.mean(dim=0)) > done_threshold).float()
        truncated = torch.zeros_like(done)
        if horizon_step == sim_rollout_length - 1:
            truncated = 1.0 - done

        observations.append(obs_active)
        actions.append(action)
        next_observations.append(next_sample)
        rewards.append(replay_reward_scaled)
        terminateds.append(done)
        truncateds.append(truncated)
        transition_candidate_ids.append(active_candidate_ids)
        transition_steps.append(torch.full_like(active_candidate_ids, horizon_step))
        transition_uncertainties.append(uncertainty)
        raw_rewards.append(reward_mean)

        active_indices = active.nonzero(as_tuple=False).squeeze(-1)
        obs_next = obs.clone()
        obs_next[active_indices] = next_sample
        obs = obs_next
        qpos_next = qpos_root.clone()
        qpos_next[active_indices] = next_qpos_sample
        qpos_root = qpos_next
        qvel_next = qvel_root.clone()
        qvel_next[active_indices] = next_qvel_sample
        qvel_root = qvel_next
        active_next = active.clone()
        active_next[active_indices] = done <= 0.5
        active = active_next

    if not observations:
        raise RuntimeError("AGES rollout produced no candidate transitions.")

    candidate_returns = returns
    return_matrix = candidate_returns.view(state_count, repeat)
    baseline = return_matrix.mean(dim=1, keepdim=True)
    advantages = (return_matrix - baseline).reshape(-1)
    if select_mode == "global_top_ratio":
        selected_count = max(1, int(math.ceil(total_candidates * float(select_ratio))))
        selected_count = min(total_candidates, selected_count)
        selected_candidate_ids = torch.topk(advantages, k=selected_count, largest=True).indices
    elif select_mode == "per_state_top1":
        selected_candidate_ids = torch.argmax(return_matrix - baseline, dim=1) + torch.arange(
            state_count, device=device
        ) * repeat
    else:
        raise ValueError(f"Unsupported AGES select mode: {select_mode}")

    selected_candidate_mask = torch.zeros(total_candidates, dtype=torch.bool, device=device)
    selected_candidate_mask[selected_candidate_ids] = True

    all_transition_candidate_ids = torch.cat(transition_candidate_ids, dim=0)
    all_transition_steps = torch.cat(transition_steps, dim=0)
    selected_transition_mask = selected_candidate_mask[all_transition_candidate_ids] & (all_transition_steps < inject_horizon)
    if not bool(torch.any(selected_transition_mask)):
        best_id = torch.argmax(advantages)
        selected_candidate_mask[best_id] = True
        selected_transition_mask = selected_candidate_mask[all_transition_candidate_ids] & (
            all_transition_steps < inject_horizon
        )

    full_batch = {
        "observation": torch.cat(observations, dim=0),
        "actor_observation": torch.cat(observations, dim=0),
        "action": torch.cat(actions, dim=0),
        "reward": torch.cat(rewards, dim=0),
        "terminated": torch.cat(terminateds, dim=0),
        "truncated": torch.cat(truncateds, dim=0),
        "next_observation": torch.cat(next_observations, dim=0),
        "actor_next_observation": torch.cat(next_observations, dim=0),
    }
    selected_batch = _slice_torch_batch(full_batch, selected_transition_mask)
    selected_returns = candidate_returns[selected_candidate_mask]
    selected_advantages = advantages[selected_candidate_mask]
    uncertainty_tensor = torch.cat(transition_uncertainties, dim=0)
    selected_uncertainties = uncertainty_tensor[selected_transition_mask]

    metrics = {
        "ages_candidate_return_mean": float(candidate_returns.mean().item()),
        "ages_candidate_return_std": float(candidate_returns.std(unbiased=False).item()),
        "ages_selected_return_mean": float(selected_returns.mean().item()),
        "ages_selected_adv_mean": float(selected_advantages.mean().item()),
        "ages_selected_adv_min": float(selected_advantages.min().item()),
        "ages_selected_adv_max": float(selected_advantages.max().item()),
        "ages_accept_rate": float(selected_candidate_mask.float().mean().item()),
        "ages_buffer_size": float(selected_batch["observation"].shape[0]),
        "ages_uncertainty_mean": float(selected_uncertainties.mean().item()),
        "ages_generation_wall_time": float(time.perf_counter() - started),
        "ages_sim_ratio": float("nan"),
        "ages_select_ratio": float(select_ratio),
        "ages_action_temperature": float(action_temperature),
        "ages_raw_reward_mean": float(torch.cat(raw_rewards, dim=0).mean().item()),
        "ages_selected_transition_count": float(selected_batch["observation"].shape[0]),
        "ages_total_candidates": float(total_candidates),
        "ages_sim_rollout_length": float(sim_rollout_length),
        "ages_inject_rollout_length": float(inject_horizon),
    }
    return selected_batch, metrics


def _sample_from_torch_batch(batch: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    size = int(batch["observation"].shape[0])
    idx = torch.randint(0, size, (batch_size,), device=batch["observation"].device)
    return {key: value[idx] for key, value in batch.items()}


def _update_actor_bc_mopo(
    *,
    actor: Network,
    critic: Network,
    temperature: Network,
    reference_actor: FlashSACActor | None,
    mixed_batch: dict[str, torch.Tensor],
    real_batch: dict[str, torch.Tensor] | None,
    synthetic_batch: dict[str, torch.Tensor] | None,
    bc_alpha: float,
    bc_ref_alpha: float,
    bc_ref_on: str,
    actor_rl_coef: float,
    grad_clip_norm: float,
) -> dict[str, torch.Tensor]:
    mixed_obs = mixed_batch["actor_observation"]
    actions, info = actor(observations=mixed_obs, training=True)
    log_probs = info["log_prob"]

    critic.network.requires_grad_(False)
    qs, _ = critic(
        observations=mixed_batch["observation"],
        actions=actions,
        training=False,
    )
    q = torch.minimum(qs[0], qs[1])
    critic.network.requires_grad_(True)

    temp_value = temperature().detach()
    actor_rl_loss_raw = (log_probs * temp_value - q).mean()
    actor_rl_loss_scaled = float(actor_rl_coef) * actor_rl_loss_raw
    actor_bc_loss = torch.zeros((), dtype=actor_rl_loss_raw.dtype, device=actor_rl_loss_raw.device)
    actor_bc_ref_loss = torch.zeros((), dtype=actor_rl_loss_raw.dtype, device=actor_rl_loss_raw.device)
    deterministic_actions = actions
    if bc_alpha > 0:
        if real_batch is None:
            raise ValueError("bc_alpha > 0 requires a non-empty real expert batch.")
        mean, _ = actor.apply(
            "get_mean_and_std",
            observations=real_batch["actor_observation"],
            training=True,
        )
        deterministic_actions = torch.tanh(mean)
        actor_bc_loss = F.mse_loss(deterministic_actions, real_batch["action"])

    if bc_ref_alpha > 0:
        if reference_actor is None:
            raise ValueError("bc_ref_alpha > 0 requires a frozen reference actor.")
        if bc_ref_on == "mixed":
            ref_batch = mixed_batch
        elif bc_ref_on == "synthetic":
            if synthetic_batch is None:
                raise ValueError("bc_ref_on=synthetic requires synthetic samples in the update batch.")
            ref_batch = synthetic_batch
        elif bc_ref_on == "real":
            if real_batch is None:
                raise ValueError("bc_ref_on=real requires real expert samples in the update batch.")
            ref_batch = real_batch
        else:
            raise ValueError(f"Unsupported bc_ref_on: {bc_ref_on}")

        ref_obs = ref_batch["actor_observation"]
        cur_mean, _ = actor.apply("get_mean_and_std", observations=ref_obs, training=True)
        cur_action = torch.tanh(cur_mean)
        with torch.no_grad():
            ref_mean, _ = reference_actor.get_mean_and_std(ref_obs, training=False)
            ref_action = torch.tanh(ref_mean)
        actor_bc_ref_loss = F.mse_loss(cur_action, ref_action)

    actor_total_loss = actor_rl_loss_scaled + float(bc_alpha) * actor_bc_loss + float(bc_ref_alpha) * actor_bc_ref_loss
    actor_entropy = -log_probs.mean()
    mean_action = deterministic_actions.mean()

    assert actor.optimizer is not None
    actor.optimizer.zero_grad(set_to_none=True)
    actor_total_loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(actor.network.parameters(), grad_clip_norm)
    actor.optimizer.step()
    if actor.scheduler is not None:
        actor.scheduler.step()
    actor.normalize_parameters()

    return {
        "actor/loss": actor_total_loss.detach(),
        "actor/rl_loss": actor_rl_loss_raw.detach(),
        "actor/rl_loss_raw": actor_rl_loss_raw.detach(),
        "actor/rl_loss_scaled": actor_rl_loss_scaled.detach(),
        "actor/bc_loss": actor_bc_loss.detach(),
        "actor/bc_ref_loss": actor_bc_ref_loss.detach(),
        "actor/entropy": actor_entropy.detach(),
        "actor/mean_action": mean_action.detach(),
    }


@torch.no_grad()
def _evaluate_actor_bc_real(
    *,
    actor: Network,
    arrays: TransitionArrays,
    device: torch.device,
    max_transitions: int,
    seed: int,
) -> dict[str, float]:
    size = arrays.size
    if size == 0:
        return {"mse": float("nan"), "mae": float("nan")}
    count = min(size, max(1, int(max_transitions)))
    rng = np.random.default_rng(seed)
    if count < size:
        idx_np = np.sort(rng.choice(size, size=count, replace=False))
    else:
        idx_np = np.arange(size)
    obs = torch.as_tensor(arrays.observation[idx_np], dtype=torch.float32, device=device)
    action = torch.as_tensor(arrays.action[idx_np], dtype=torch.float32, device=device)
    mean, _ = actor.apply("get_mean_and_std", observations=obs, training=False)
    pred_action = torch.tanh(mean)
    error = pred_action - action
    return {
        "mse": float(torch.mean(error.square()).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
    }


@torch.no_grad()
def _evaluate_actor_reference_mse(
    *,
    actor: Network,
    reference_actor: FlashSACActor | None,
    batch: dict[str, torch.Tensor] | None,
) -> float:
    if reference_actor is None or batch is None:
        return float("nan")
    obs = batch["actor_observation"]
    if obs.numel() == 0:
        return float("nan")
    mean, _ = actor.apply("get_mean_and_std", observations=obs, training=False)
    action = torch.tanh(mean)
    ref_mean, _ = reference_actor.get_mean_and_std(obs, training=False)
    ref_action = torch.tanh(ref_mean)
    return float(F.mse_loss(action, ref_action).item())


def _train_dynamics(
    *,
    dynamics: DynamicsEnsemble,
    train: TransitionArrays,
    val: TransitionArrays,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=args.dynamics_lr, weight_decay=args.weight_decay)
    train_obs = torch.as_tensor(train.observation, dtype=torch.float32, device=device)
    train_action = torch.as_tensor(train.action, dtype=torch.float32, device=device)
    train_delta = torch.as_tensor(train.next_observation - train.observation, dtype=torch.float32, device=device)
    train_reward = torch.as_tensor(train.reward, dtype=torch.float32, device=device)
    train_done = torch.as_tensor(train.terminated, dtype=torch.float32, device=device)
    train_qpos_root = (
        None if train.qpos_root is None else torch.as_tensor(train.qpos_root, dtype=torch.float32, device=device)
    )
    train_qvel_root = (
        None if train.qvel_root is None else torch.as_tensor(train.qvel_root, dtype=torch.float32, device=device)
    )
    train_delta_qpos_root = None
    train_delta_qvel_root = None
    if train.qpos_root is not None and train.next_qpos_root is not None:
        train_delta_qpos_root = torch.as_tensor(
            train.next_qpos_root - train.qpos_root, dtype=torch.float32, device=device
        )
    if train.qvel_root is not None and train.next_qvel_root is not None:
        train_delta_qvel_root = torch.as_tensor(
            train.next_qvel_root - train.qvel_root, dtype=torch.float32, device=device
        )
    val_obs = torch.as_tensor(val.observation, dtype=torch.float32, device=device)
    val_action = torch.as_tensor(val.action, dtype=torch.float32, device=device)
    val_delta = torch.as_tensor(val.next_observation - val.observation, dtype=torch.float32, device=device)
    val_reward = torch.as_tensor(val.reward, dtype=torch.float32, device=device)
    val_done = torch.as_tensor(val.terminated, dtype=torch.float32, device=device)

    steps_per_epoch = max(1, math.ceil(train.size / args.model_batch_size))
    if args.dynamics_steps is not None:
        total_steps = int(args.dynamics_steps)
    else:
        total_steps = max(1, int(args.dynamics_epochs) * steps_per_epoch)

    losses: list[float] = []
    delta_losses: list[float] = []
    qpos_root_losses: list[float] = []
    qvel_root_losses: list[float] = []
    reward_losses: list[float] = []
    done_losses: list[float] = []
    dynamics.train()
    for _ in range(total_steps):
        idx = torch.randint(0, train.size, (min(args.model_batch_size, train.size),), device=device)
        pred = dynamics(
            train_obs[idx],
            train_action[idx],
            None if train_qpos_root is None else train_qpos_root[idx],
            None if train_qvel_root is None else train_qvel_root[idx],
        )
        delta, delta_qpos_root, delta_qvel_root, reward, done_logit = dynamics.split_prediction(pred)
        delta_target = train_delta[idx].unsqueeze(0).expand_as(delta)
        delta_loss = F.mse_loss(delta, delta_target)
        loss = delta_loss
        qpos_root_loss = torch.zeros((), device=device)
        qvel_root_loss = torch.zeros((), device=device)
        if dynamics.predict_root:
            if train_delta_qpos_root is None or train_delta_qvel_root is None:
                raise ValueError("Root-aware dynamics training requires next_qpos_root and next_qvel_root.")
            assert delta_qpos_root is not None and delta_qvel_root is not None
            qpos_target = train_delta_qpos_root[idx].unsqueeze(0).expand_as(delta_qpos_root)
            qvel_target = train_delta_qvel_root[idx].unsqueeze(0).expand_as(delta_qvel_root)
            qpos_xyz_loss = F.mse_loss(delta_qpos_root[..., :3], qpos_target[..., :3])
            qpos_quat_loss = F.mse_loss(delta_qpos_root[..., 3:7], qpos_target[..., 3:7])
            qpos_root_loss = qpos_xyz_loss + float(args.quat_loss_coef) * qpos_quat_loss
            qvel_root_loss = F.mse_loss(delta_qvel_root, qvel_target)
            loss = loss + float(args.root_loss_coef) * float(args.qpos_root_loss_coef) * qpos_root_loss
            loss = loss + float(args.root_loss_coef) * float(args.qvel_root_loss_coef) * qvel_root_loss
        reward_loss = torch.zeros((), device=device)
        done_loss = torch.zeros((), device=device)
        if reward is not None:
            reward_target = train_reward[idx].unsqueeze(0).expand_as(reward)
            reward_loss = F.mse_loss(reward, reward_target)
            loss = loss + args.reward_loss_coef * reward_loss
        if done_logit is not None:
            done_target = train_done[idx].unsqueeze(0).expand_as(done_logit)
            done_loss = F.binary_cross_entropy_with_logits(done_logit, done_target)
            loss = loss + args.done_loss_coef * done_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite dynamics loss: {float(loss.item())}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), args.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
        delta_losses.append(float(delta_loss.item()))
        qpos_root_losses.append(float(qpos_root_loss.item()))
        qvel_root_losses.append(float(qvel_root_loss.item()))
        reward_losses.append(float(reward_loss.item()))
        done_losses.append(float(done_loss.item()))

    val_metrics = _evaluate_dynamics(
        dynamics=dynamics,
        obs=val_obs,
        action=val_action,
        delta=val_delta,
        reward=val_reward,
        done=val_done,
        qpos_root=None if val.qpos_root is None else torch.as_tensor(val.qpos_root, dtype=torch.float32, device=device),
        qvel_root=None if val.qvel_root is None else torch.as_tensor(val.qvel_root, dtype=torch.float32, device=device),
        next_qpos_root=None
        if val.next_qpos_root is None
        else torch.as_tensor(val.next_qpos_root, dtype=torch.float32, device=device),
        next_qvel_root=None
        if val.next_qvel_root is None
        else torch.as_tensor(val.next_qvel_root, dtype=torch.float32, device=device),
        obs_mean=torch.as_tensor(args._obs_mean, dtype=torch.float32, device=device),
        obs_std=torch.as_tensor(args._obs_std, dtype=torch.float32, device=device),
        force_last_action=args.force_last_action,
        formula_reward=bool(args.formula_reward),
        formula_done=bool(args.formula_done),
        observable_reward_context=getattr(args, "_observable_reward_context", None),
    )
    train_metrics = {
        "dynamics_loss": float(np.mean(losses)),
        "dynamics_delta_loss": float(np.mean(delta_losses)),
        "dynamics_qpos_root_loss": float(np.mean(qpos_root_losses)),
        "dynamics_qvel_root_loss": float(np.mean(qvel_root_losses)),
        "dynamics_reward_loss": float(np.mean(reward_losses)),
        "dynamics_done_loss": float(np.mean(done_losses)),
        "dynamics_steps": int(total_steps),
    }
    return train_metrics, val_metrics


@torch.no_grad()
def _evaluate_dynamics(
    *,
    dynamics: DynamicsEnsemble,
    obs: torch.Tensor,
    action: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    qpos_root: torch.Tensor | None,
    qvel_root: torch.Tensor | None,
    next_qpos_root: torch.Tensor | None,
    next_qvel_root: torch.Tensor | None,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    force_last_action: bool,
    formula_reward: bool,
    formula_done: bool,
    observable_reward_context: dict[str, Any] | None,
) -> dict[str, Any]:
    dynamics.eval()
    pred = dynamics(obs, action, qpos_root, qvel_root)
    pred_delta, pred_delta_qpos_root, pred_delta_qvel_root, pred_reward, pred_done_logit = dynamics.split_prediction(
        pred
    )
    next_pred = obs.unsqueeze(0) + pred_delta
    next_pred = torch.stack(
        [
            _force_last_action_slice(
                next_pred[i],
                action,
                obs_mean=obs_mean,
                obs_std=obs_std,
                enabled=force_last_action,
            )
            for i in range(next_pred.shape[0])
        ],
        dim=0,
    )
    next_target = obs + delta
    error = next_pred.mean(dim=0) - next_target
    metrics: dict[str, Any] = {
        "delta_mse": float(F.mse_loss(pred_delta.mean(dim=0), delta).item()),
        "next_obs_mse": float(torch.mean(error.square()).item()),
        "next_obs_mae": float(torch.mean(torch.abs(error)).item()),
        "uncertainty_mean": float(next_pred.var(dim=0).mean(dim=-1).mean().item()),
        "chunk_mse": {name: float(torch.mean(error[:, slc].square()).item()) for name, slc in SLICES.items()},
    }
    if dynamics.predict_root:
        if qpos_root is None or qvel_root is None or next_qpos_root is None or next_qvel_root is None:
            raise ValueError("Root-aware dynamics evaluation requires root current and next arrays.")
        assert pred_delta_qpos_root is not None and pred_delta_qvel_root is not None
        next_qpos_pred = _normalize_root_quat(qpos_root.unsqueeze(0) + pred_delta_qpos_root)
        next_qvel_pred = qvel_root.unsqueeze(0) + pred_delta_qvel_root
        qpos_error = next_qpos_pred.mean(dim=0) - next_qpos_root
        qvel_error = next_qvel_pred.mean(dim=0) - next_qvel_root
        metrics.update(
            {
                "qpos_root_delta_mse": float(
                    F.mse_loss(pred_delta_qpos_root.mean(dim=0), next_qpos_root - qpos_root).item()
                ),
                "qvel_root_delta_mse": float(
                    F.mse_loss(pred_delta_qvel_root.mean(dim=0), next_qvel_root - qvel_root).item()
                ),
                "qpos_root_next_mse": float(torch.mean(qpos_error.square()).item()),
                "qvel_root_next_mse": float(torch.mean(qvel_error.square()).item()),
                "qpos_xyz_mse": float(torch.mean(qpos_error[:, :3].square()).item()),
                "qpos_quat_mse": float(torch.mean(qpos_error[:, 3:7].square()).item()),
                "qvel_root_mse": float(torch.mean(qvel_error.square()).item()),
                "root_uncertainty_mean": float(
                    (next_qpos_pred.var(dim=0).mean(dim=-1) + next_qvel_pred.var(dim=0).mean(dim=-1)).mean().item()
                ),
            }
        )
        if formula_reward or formula_done:
            if observable_reward_context is None:
                raise ValueError("Formula reward/done evaluation requires observable reward context.")
            member_idx = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
            row_idx = torch.arange(obs.shape[0], device=obs.device)
            reward_formula, done_formula, _ = _compute_observable_reward_torch(
                obs_t_norm=obs,
                act_t=action,
                obs_tp1_norm=next_pred[member_idx, row_idx],
                qpos_root_tp1=next_qpos_pred[member_idx, row_idx],
                qvel_root_tp1=next_qvel_pred[member_idx, row_idx],
                obs_mean=obs_mean,
                obs_std=obs_std,
                context=observable_reward_context,
            )
            metrics["formula_reward_mse"] = float(F.mse_loss(reward_formula, reward).item())
            metrics["formula_reward_mae"] = float(F.l1_loss(reward_formula, reward).item())
            metrics["formula_done_accuracy"] = float(((done_formula > 0.5) == (done > 0.5)).float().mean().item())
            metrics["formula_done_pred_rate"] = float((done_formula > 0.5).float().mean().item())
            metrics["formula_done_label_rate"] = float((done > 0.5).float().mean().item())
    if pred_reward is not None:
        metrics["reward_mse"] = float(F.mse_loss(pred_reward.mean(dim=0), reward).item())
    if pred_done_logit is not None:
        done_prob = torch.sigmoid(pred_done_logit.mean(dim=0))
        metrics["done_bce"] = float(F.binary_cross_entropy(done_prob, done).item())
        metrics["done_pred_rate"] = float((done_prob > 0.5).float().mean().item())
        metrics["done_label_rate"] = float(done.mean().item())
    dynamics.train()
    return metrics


def _evaluate_dynamics_arrays(
    *,
    dynamics: DynamicsEnsemble,
    arrays: TransitionArrays,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return _evaluate_dynamics(
        dynamics=dynamics,
        obs=torch.as_tensor(arrays.observation, dtype=torch.float32, device=device),
        action=torch.as_tensor(arrays.action, dtype=torch.float32, device=device),
        delta=torch.as_tensor(arrays.next_observation - arrays.observation, dtype=torch.float32, device=device),
        reward=torch.as_tensor(arrays.reward, dtype=torch.float32, device=device),
        done=torch.as_tensor(arrays.terminated, dtype=torch.float32, device=device),
        qpos_root=None
        if arrays.qpos_root is None
        else torch.as_tensor(arrays.qpos_root, dtype=torch.float32, device=device),
        qvel_root=None
        if arrays.qvel_root is None
        else torch.as_tensor(arrays.qvel_root, dtype=torch.float32, device=device),
        next_qpos_root=None
        if arrays.next_qpos_root is None
        else torch.as_tensor(arrays.next_qpos_root, dtype=torch.float32, device=device),
        next_qvel_root=None
        if arrays.next_qvel_root is None
        else torch.as_tensor(arrays.next_qvel_root, dtype=torch.float32, device=device),
        obs_mean=torch.as_tensor(args._obs_mean, dtype=torch.float32, device=device),
        obs_std=torch.as_tensor(args._obs_std, dtype=torch.float32, device=device),
        force_last_action=args.force_last_action,
        formula_reward=bool(args.formula_reward),
        formula_done=bool(args.formula_done),
        observable_reward_context=getattr(args, "_observable_reward_context", None),
    )


def _make_flashsac_networks(
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Network, Network, Network, Network]:
    actor_lr = args.lr if args.actor_lr is None else args.actor_lr
    critic_lr = args.lr if args.critic_lr is None else args.critic_lr
    temperature_lr = args.lr if args.temperature_lr is None else args.temperature_lr
    actor_net = _make_actor(device)
    if args.bc_checkpoint is not None:
        _load_actor_checkpoint(actor_net, args.bc_checkpoint.expanduser().resolve(), device)
    actor = _make_network_bundle(
        actor_net,
        lr=actor_lr,
        weight_decay=args.weight_decay,
        use_weight_normalization=args.actor_weight_normalization,
    )
    if args.bc_checkpoint is None and args.actor_weight_normalization:
        actor.normalize_parameters()

    critic_net = FlashSACDoubleCritic(
        num_blocks=args.critic_num_blocks,
        input_dim=hrlg.POLICY_OBS_SIZE + hrlg.ACTION_SIZE,
        hidden_dim=args.critic_hidden_dim,
        num_bins=args.critic_num_bins,
        min_v=args.critic_min_v,
        max_v=args.critic_max_v,
    ).to(device)
    critic = _make_network_bundle(
        critic_net,
        lr=critic_lr,
        weight_decay=args.weight_decay,
        use_weight_normalization=True,
    )
    critic.normalize_parameters()

    target_net = FlashSACDoubleCritic(
        num_blocks=args.critic_num_blocks,
        input_dim=hrlg.POLICY_OBS_SIZE + hrlg.ACTION_SIZE,
        hidden_dim=args.critic_hidden_dim,
        num_bins=args.critic_num_bins,
        min_v=args.critic_min_v,
        max_v=args.critic_max_v,
    ).to(device)
    target_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_net,
        optimizer=None,
        scheduler=None,
        use_weight_normalization=True,
        ema_source=critic,
        ema_tau=args.tau,
    )
    target_critic.normalize_parameters()

    temperature = _make_network_bundle(
        FlashSACTemperature(args.alpha).to(device),
        lr=temperature_lr,
        weight_decay=0.0,
        use_weight_normalization=False,
    )
    return actor, critic, target_critic, temperature


def _save_checkpoint(
    path: Path,
    *,
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    dynamics: DynamicsEnsemble,
    update_step: int,
    epoch: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    root_stats: dict[str, np.ndarray],
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = _public_config(args)
    actor.save(str(path / "actor.pt"))
    critic.save(str(path / "critic.pt"))
    target_critic.save(str(path / "target_critic.pt"))
    temperature.save(str(path / "temperature.pt"))
    torch.save(
        {
            "network_state_dict": dynamics.state_dict(),
            "update_step": int(update_step),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "obs_dim": hrlg.POLICY_OBS_SIZE,
            "action_dim": hrlg.ACTION_SIZE,
            "predict_root": bool(dynamics.predict_root),
            "predict_reward": bool(dynamics.predict_reward),
            "predict_done": bool(dynamics.predict_done),
        },
        path / "dynamics_ensemble.pt",
    )
    np.savez(path / "obs_stats.npz", mean=obs_mean, std=obs_std, enabled=np.array(bool(args.normalize_obs)))
    np.savez(path / "root_stats.npz", **root_stats)
    _write_json(path / "checkpoint_metrics.json", metrics)
    _write_json(path / "resolved_config.json", config)
    _write_json(path / "dynamics_config.json", _dynamics_config(args, dynamics))
    if hasattr(args, "_offline_provenance"):
        _write_json(path / "offline_provenance.json", args._offline_provenance)
    if hasattr(args, "_split_info"):
        _write_json(path / "split_info.json", args._split_info)
    if hasattr(args, "_relabel_info"):
        _write_json(path / "reward_relabel.json", args._relabel_info)


def _dynamics_config(args: argparse.Namespace, dynamics: DynamicsEnsemble) -> dict[str, Any]:
    return {
        "dynamics_ensemble_size": int(len(dynamics.members)),
        "dynamics_hidden_dim": int(args.dynamics_hidden_dim),
        "dynamics_layers": int(args.dynamics_layers),
        "obs_dim": int(dynamics.obs_dim),
        "action_dim": int(dynamics.action_dim),
        "root_dim": int(dynamics.root_dim),
        "predict_root": bool(dynamics.predict_root),
        "predict_reward": bool(dynamics.predict_reward),
        "predict_done": bool(dynamics.predict_done),
        "predict_reward_head": bool(args.predict_reward_head),
        "predict_done_head": bool(args.predict_done_head),
        "formula_reward": bool(args.formula_reward),
        "formula_done": bool(args.formula_done),
        "normalize_obs": bool(args.normalize_obs),
        "root_stats_strategy": "saved_raw_root_stats_root_inputs_are_raw",
    }


def _save_dynamics_checkpoint(
    path: Path,
    *,
    dynamics: DynamicsEnsemble,
    update_step: int,
    epoch: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    root_stats: dict[str, np.ndarray],
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = _public_config(args)
    torch.save(
        {
            "network_state_dict": dynamics.state_dict(),
            "update_step": int(update_step),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "obs_dim": hrlg.POLICY_OBS_SIZE,
            "action_dim": hrlg.ACTION_SIZE,
            "predict_root": bool(dynamics.predict_root),
            "predict_reward": bool(dynamics.predict_reward),
            "predict_done": bool(dynamics.predict_done),
        },
        path / "dynamics_ensemble.pt",
    )
    np.savez(path / "obs_stats.npz", mean=obs_mean, std=obs_std, enabled=np.array(bool(args.normalize_obs)))
    np.savez(path / "root_stats.npz", **root_stats)
    _write_json(path / "checkpoint_metrics.json", metrics)
    _write_json(path / "resolved_config.json", config)
    _write_json(path / "dynamics_config.json", _dynamics_config(args, dynamics))
    if hasattr(args, "_offline_provenance"):
        _write_json(path / "offline_provenance.json", args._offline_provenance)
    if hasattr(args, "_split_info"):
        _write_json(path / "split_info.json", args._split_info)
    if hasattr(args, "_relabel_info"):
        _write_json(path / "reward_relabel.json", args._relabel_info)


def _append_online_eval(output_dir: Path, metrics: dict[str, Any]) -> None:
    csv_path = output_dir / "online_eval.csv"
    jsonl_path = output_dir / "online_eval.jsonl"
    fieldnames = [
        "epoch",
        "update",
        "avg_return",
        "std_return",
        "avg_length",
        "min_length",
        "max_length",
        "survival_rate",
        "survived_full",
        "done_count",
        "num_eval_episodes",
        "eval_json",
        "wall_time_sec",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")


def _run_online_eval(
    checkpoint: Path,
    *,
    args: argparse.Namespace,
    epoch: int,
    update_step: int,
    output_dir: Path,
) -> dict[str, Any]:
    from scripts.eval_mjp_g1_hrlg_bc import evaluate as evaluate_bc

    eval_dir = output_dir / "online_eval"
    eval_output = eval_dir / f"epoch_{epoch:04d}.json"
    eval_args = argparse.Namespace(
        checkpoint=checkpoint,
        num_eval_episodes=args.online_eval_episodes,
        num_envs=args.online_eval_num_envs,
        max_episode_steps=args.online_eval_max_episode_steps,
        seed=args.online_eval_seed if args.online_eval_seed is not None else args.seed + 20_000 + epoch,
        device=args.online_eval_device,
        noise_level=args.online_eval_noise_level,
        output=eval_output,
        quiet=True,
    )
    metrics = evaluate_bc(eval_args)
    metrics["epoch"] = int(epoch)
    metrics["update"] = int(update_step)
    metrics["eval_json"] = str(eval_output.resolve())
    _append_online_eval(output_dir, metrics)
    print(
        json.dumps(
            {
                "online_eval_epoch": epoch,
                "update": update_step,
                "avg_return": metrics["avg_return"],
                "avg_length": metrics["avg_length"],
                "survived_full": metrics["survived_full"],
                "num_eval_episodes": metrics["num_eval_episodes"],
                "eval_json": metrics["eval_json"],
            },
            sort_keys=True,
        )
    )
    return metrics


def _online_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    return (
        int(candidate["survived_full"]),
        float(candidate["avg_length"]),
        float(candidate["avg_return"]),
    ) > (
        int(best["survived_full"]),
        float(best["avg_length"]),
        float(best["avg_return"]),
    )


def _run_training(args: argparse.Namespace) -> dict[str, Any]:
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive.")
    if args.bc_alpha > 0 and args.real_ratio <= 0:
        raise ValueError("--bc-alpha > 0 requires --real-ratio > 0 so BC loss can use real expert samples.")
    if args.skip_dynamics_training and args.dynamics_checkpoint is None:
        raise ValueError("--skip-dynamics-training requires --dynamics-checkpoint.")
    if args.dynamics_only and args.skip_dynamics_training:
        raise ValueError("--dynamics-only cannot be combined with --skip-dynamics-training.")
    dynamics_checkpoint = (
        _resolve_dynamics_checkpoint(args.dynamics_checkpoint) if args.dynamics_checkpoint is not None else None
    )

    data_split_seed = args.seed if args.data_split_seed is None else args.data_split_seed
    train_raw, val_raw, split, relabel_info = _load_transitions(
        args.dataset.expanduser().resolve(),
        val_ratio=args.val_ratio,
        seed=data_split_seed,
        limit_trajectories=args.limit_trajectories,
        limit_transitions=args.limit_transitions,
        reward_source=args.reward_source,
        reward_relabel_batch_size=args.reward_relabel_batch_size,
    )
    if dynamics_checkpoint is not None:
        obs_mean, obs_std, normalize_obs = _load_obs_stats_from_dynamics_checkpoint(dynamics_checkpoint)
        args.normalize_obs = bool(normalize_obs)
        stats = (obs_mean, obs_std, normalize_obs)
    elif args.bc_checkpoint is not None:
        stats = _load_obs_stats_from_checkpoint(args.bc_checkpoint.expanduser().resolve())
    else:
        stats = None
    if stats is None:
        obs_mean, obs_std = _fit_obs_stats(train_raw.observation)
        normalize_obs = bool(args.normalize_obs)
    else:
        obs_mean, obs_std, normalize_obs = stats
        args.normalize_obs = bool(normalize_obs)
    if not args.normalize_obs:
        obs_mean = np.zeros(hrlg.POLICY_OBS_SIZE, dtype=np.float32)
        obs_std = np.ones(hrlg.POLICY_OBS_SIZE, dtype=np.float32)
    args._obs_mean = obs_mean
    args._obs_std = obs_std
    root_stats = (
        _load_root_stats_from_dynamics_checkpoint(dynamics_checkpoint)
        if dynamics_checkpoint is not None
        else _fit_root_stats(train_raw.qpos_root, train_raw.qvel_root)
    )
    args._root_stats = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in root_stats.items()
    }
    if args.predict_root and (train_raw.qpos_root is None or train_raw.qvel_root is None):
        raise ValueError("--predict-root requires qpos_root and qvel_root in the dataset.")
    if (args.formula_reward or args.formula_done) and not args.predict_root:
        raise ValueError("--formula-reward/--formula-done require --predict-root.")

    train = _normalize(train_raw, obs_mean, obs_std)
    val = _normalize(val_raw, obs_mean, obs_std)
    obs_mean_t = torch.as_tensor(obs_mean, dtype=torch.float32, device=device)
    obs_std_t = torch.as_tensor(obs_std, dtype=torch.float32, device=device)
    if args.formula_reward or args.formula_done:
        args._observable_reward_context = _make_observable_reward_context(device)
    else:
        args._observable_reward_context = None

    split_info = asdict(split)
    args._split_info = split_info
    args._relabel_info = relabel_info
    if dynamics_checkpoint is not None and not getattr(args, "allow_dynamics_split_mismatch", False):
        _validate_dynamics_checkpoint_split(dynamics_checkpoint, split_info)
    config = _public_config(args)
    _write_json(output_dir / "resolved_config.json", config)
    ages_config = {
        "use_ages": bool(args.use_ages),
        "ages_action_temperature": float(args.ages_action_temperature),
        "ages_sim_ratio": float(args.ages_sim_ratio),
        "ages_select_ratio": float(args.ages_select_ratio),
        "ages_num_dataset_states": int(args.ages_num_dataset_states),
        "ages_trajectories_per_state": int(args.ages_trajectories_per_state),
        "ages_baseline": args.ages_baseline,
        "ages_select_mode": args.ages_select_mode,
        "ages_buffer_retain_epochs": int(args.ages_buffer_retain_epochs),
        "ages_sim_rollout_length": int(args.ages_sim_rollout_length),
        "ages_inject_rollout_length": int(args.ages_inject_rollout_length),
        "ages_generation_interval": int(args.ages_generation_interval),
        "ages_min_buffer_size": int(args.ages_min_buffer_size),
        "ages_max_buffer_size": int(args.ages_max_buffer_size),
        "ages_score_use_uncertainty": bool(args.ages_score_use_uncertainty),
        "ages_score_use_mopo_penalty": bool(args.ages_score_use_mopo_penalty),
    }
    _write_json(output_dir / "ages_config.json", ages_config)
    offline_provenance = {
        "status": "offline_only_verified",
        "training_script": str(Path(__file__).resolve()),
        "dataset_path": str(args.dataset.expanduser().resolve()),
        "bc_checkpoint": None if args.bc_checkpoint is None else str(args.bc_checkpoint.expanduser().resolve()),
        "dynamics_checkpoint": None if dynamics_checkpoint is None else str(dynamics_checkpoint),
        "training_data_sources": [
            "mujoco_playground_hrlg_npz_dataset",
            "learned_dynamics_rollouts",
            "ages_selected_learned_dynamics_rollouts" if args.use_ages else "ages_disabled",
        ],
        "ages": ages_config,
        "limit_trajectories_expected": 50,
        "no_100traj_bc_checkpoint_used": "100traj" not in str(args.bc_checkpoint or ""),
        "no_cross_group_checkpoint_used": bool(args.limit_trajectories == 50),
        "uses_online_env_transitions_for_training": False,
        "uses_env_for_reward_relabel": args.reward_source == "relabel_env",
        "uses_observable_dataset_reward": args.reward_source == "observable",
        "uses_pretrained_dynamics": dynamics_checkpoint is not None,
        "skip_dynamics_training": bool(args.skip_dynamics_training),
        "dynamics_only": bool(args.dynamics_only),
        "predict_root": bool(args.predict_root),
        "predict_reward_head": bool(args.predict_reward_head),
        "predict_done_head": bool(args.predict_done_head),
        "formula_reward": bool(args.formula_reward),
        "formula_done": bool(args.formula_done),
        "split_by": "trajectory",
        "reward_relabel": relabel_info,
        "current_dataset_limitation": (
            "The dataset has state/action/qpos_root/qvel_root but not exact reward/done, full qpos/qvel, "
            "phase_dt, foot air-time/contact history, swing peak, actuator force, or full env info."
        ),
        **split_info,
    }
    args._offline_provenance = offline_provenance
    _write_json(output_dir / "offline_provenance.json", offline_provenance)
    _write_json(output_dir / "split_info.json", split_info)
    _write_json(output_dir / "reward_relabel.json", relabel_info)

    dynamics = DynamicsEnsemble(
        ensemble_size=args.dynamics_ensemble_size,
        obs_dim=hrlg.POLICY_OBS_SIZE,
        action_dim=hrlg.ACTION_SIZE,
        hidden_dim=args.dynamics_hidden_dim,
        layers=args.dynamics_layers,
        predict_root=bool(args.predict_root),
        predict_reward=bool(split.reward_available and args.predict_reward_head),
        predict_done=bool(split.done_available and args.predict_done_head),
    ).to(device)

    loaded_dynamics_metrics: dict[str, Any] | None = None
    if dynamics_checkpoint is not None:
        loaded_ckpt = _load_dynamics_checkpoint(dynamics, dynamics_checkpoint, device)
        loaded_dynamics_metrics = loaded_ckpt.get("metrics", {})
    if args.skip_dynamics_training:
        dynamics_train_metrics = {
            "dynamics_loss": float("nan"),
            "dynamics_delta_loss": float("nan"),
            "dynamics_qpos_root_loss": float("nan"),
            "dynamics_qvel_root_loss": float("nan"),
            "dynamics_reward_loss": float("nan"),
            "dynamics_done_loss": float("nan"),
            "dynamics_steps": 0,
            "loaded_from": None if dynamics_checkpoint is None else str(dynamics_checkpoint),
            "loaded_metrics": loaded_dynamics_metrics,
        }
        dynamics_val_metrics = _evaluate_dynamics_arrays(
            dynamics=dynamics,
            arrays=val,
            device=device,
            args=args,
        )
    else:
        dynamics_train_metrics, dynamics_val_metrics = _train_dynamics(
            dynamics=dynamics,
            train=train,
            val=val,
            device=device,
            args=args,
        )
    _write_json(output_dir / "dynamics_train_metrics.json", dynamics_train_metrics)
    _write_json(output_dir / "dynamics_val_metrics.json", dynamics_val_metrics)
    _write_json(output_dir / "dynamics_config.json", _dynamics_config(args, dynamics))

    if args.dynamics_only:
        metrics = {
            "status": "passed",
            "mode": "dynamics_only",
            "seed": args.seed,
            "device": str(device),
            "dataset_path": str(args.dataset.expanduser().resolve()),
            "output_dir": str(output_dir),
            **split_info,
            "normalize_obs": args.normalize_obs,
            "reward_relabel": relabel_info,
            "dynamics_train_metrics": dynamics_train_metrics,
            "dynamics_val_metrics": dynamics_val_metrics,
            "best_model_checkpoint": str((output_dir / "best_model").resolve()),
            "final_checkpoint": str((output_dir / "final").resolve()),
        }
        for checkpoint_name in ("best_model", "final"):
            _save_dynamics_checkpoint(
                output_dir / checkpoint_name,
                dynamics=dynamics,
                update_step=0,
                epoch=0,
                obs_mean=obs_mean,
                obs_std=obs_std,
                root_stats=root_stats,
                args=args,
                metrics=metrics,
            )
        _write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics

    actor, critic, target_critic, temperature = _make_flashsac_networks(device=device, args=args)
    reference_actor: FlashSACActor | None = None
    if args.bc_ref_alpha > 0:
        if args.bc_checkpoint is None:
            raise ValueError("--bc-ref-alpha > 0 requires --bc-checkpoint.")
        reference_actor = _make_actor(device)
        _load_actor_checkpoint(reference_actor, args.bc_checkpoint.expanduser().resolve(), device)
        reference_actor.eval()
        for param in reference_actor.parameters():
            param.requires_grad_(False)

    progress_path = output_dir / "progress.csv"
    log_path = output_dir / "train.log"
    ages_stats_path = output_dir / "ages_generation_stats.jsonl"
    ages_buffer: list[tuple[int, dict[str, torch.Tensor]]] = []
    start_time = time.perf_counter()
    update_step = 0
    stop_training = False
    best_model_loss = float(dynamics_val_metrics["next_obs_mse"])
    best_online: dict[str, Any] | None = None
    online_eval_metrics: list[dict[str, Any]] = []
    metric_lists: dict[str, list[float]] = {
        "train_actor_loss": [],
        "train_actor_rl_loss": [],
        "train_actor_rl_loss_scaled": [],
        "train_critic_loss": [],
        "train_bc_loss": [],
        "train_bc_ref_loss": [],
        "temperature_loss": [],
        "uncertainty_mean": [],
        "synthetic_reward_mean": [],
        "ages_candidate_return_mean": [],
        "ages_candidate_return_std": [],
        "ages_selected_return_mean": [],
        "ages_selected_adv_mean": [],
        "ages_selected_adv_min": [],
        "ages_selected_adv_max": [],
        "ages_accept_rate": [],
        "ages_buffer_size": [],
        "ages_uncertainty_mean": [],
        "ages_generation_wall_time": [],
        "offline_bc_mse_real": [],
        "offline_bc_mae_real": [],
        "offline_ref_mse_mixed": [],
    }

    fieldnames = [
        "epoch",
        "update",
        "train_actor_loss",
        "train_actor_rl_loss",
        "train_actor_rl_loss_raw",
        "train_actor_rl_loss_scaled",
        "train_critic_loss",
        "train_bc_loss",
        "train_bc_ref_loss",
        "train_bc_ref_alpha",
        "actor_rl_coef",
        "critic_warmup_active",
        "offline_bc_mse_real",
        "offline_bc_mae_real",
        "offline_ref_mse_mixed",
        "temperature_loss",
        "dynamics_train_loss",
        "dynamics_val_loss",
        "dynamics_reward_loss",
        "dynamics_done_loss",
        "uncertainty_mean",
        "root_uncertainty_mean",
        "synthetic_return_mean",
        "ages_candidate_return_mean",
        "ages_candidate_return_std",
        "ages_selected_return_mean",
        "ages_selected_adv_mean",
        "ages_selected_adv_min",
        "ages_selected_adv_max",
        "ages_accept_rate",
        "ages_buffer_size",
        "ages_uncertainty_mean",
        "ages_generation_wall_time",
        "ages_sim_ratio",
        "ages_select_ratio",
        "ages_action_temperature",
        "dynamics_qpos_root_loss",
        "dynamics_qvel_root_loss",
        "dynamics_qpos_root_next_mse",
        "dynamics_qvel_root_next_mse",
        "formula_reward_mae",
        "online_avg_return",
        "online_avg_length",
        "online_survived_full",
        "wall_time_sec",
    ]

    _save_checkpoint(
        output_dir / "best_model",
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        dynamics=dynamics,
        update_step=update_step,
        epoch=0,
        obs_mean=obs_mean,
        obs_std=obs_std,
        root_stats=root_stats,
        args=args,
        metrics={"dynamics_val_metrics": dynamics_val_metrics, "dynamics_train_metrics": dynamics_train_metrics},
    )
    _save_checkpoint(
        output_dir / "epoch_0000",
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        dynamics=dynamics,
        update_step=update_step,
        epoch=0,
        obs_mean=obs_mean,
        obs_std=obs_std,
        root_stats=root_stats,
        args=args,
        metrics={"dynamics_val_metrics": dynamics_val_metrics, "dynamics_train_metrics": dynamics_train_metrics},
    )

    with (
        progress_path.open("w", encoding="utf-8", newline="") as csv_file,
        log_path.open("w", encoding="utf-8") as log_file,
        ages_stats_path.open("w", encoding="utf-8") as ages_stats_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        initial_bc_metrics = _evaluate_actor_bc_real(
            actor=actor,
            arrays=train,
            device=device,
            max_transitions=args.offline_bc_eval_transitions,
            seed=args.seed + 50_000,
        )
        metric_lists["offline_bc_mse_real"].append(initial_bc_metrics["mse"])
        metric_lists["offline_bc_mae_real"].append(initial_bc_metrics["mae"])
        initial_ref_batch = None
        if reference_actor is not None:
            initial_ref_batch = _make_real_batch(
                train,
                min(train.size, args.offline_bc_eval_transitions),
                device,
                reward_scale=args.reward_scale,
            )
        initial_ref_mse = _evaluate_actor_reference_mse(
            actor=actor,
            reference_actor=reference_actor,
            batch=initial_ref_batch,
        )
        metric_lists["offline_ref_mse_mixed"].append(initial_ref_mse)
        initial_online_metrics = None
        if args.online_eval_interval > 0:
            initial_online_metrics = _run_online_eval(
                output_dir / "epoch_0000",
                args=args,
                epoch=0,
                update_step=0,
                output_dir=output_dir,
            )
            online_eval_metrics.append(initial_online_metrics)
            if _online_better(initial_online_metrics, best_online):
                best_online = initial_online_metrics
                _save_checkpoint(
                    output_dir / "best_online",
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    temperature=temperature,
                    dynamics=dynamics,
                    update_step=0,
                    epoch=0,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    root_stats=root_stats,
                    args=args,
                    metrics={"online_eval": initial_online_metrics, "initial_bc_metrics": initial_bc_metrics},
                )
        initial_row = {
            "epoch": 0,
            "update": 0,
            "train_actor_loss": float("nan"),
            "train_actor_rl_loss": float("nan"),
            "train_actor_rl_loss_raw": float("nan"),
            "train_actor_rl_loss_scaled": float("nan"),
            "train_critic_loss": float("nan"),
            "train_bc_loss": float("nan"),
            "train_bc_ref_loss": float("nan"),
            "train_bc_ref_alpha": float(args.bc_ref_alpha),
            "actor_rl_coef": float(args.actor_rl_coef),
            "critic_warmup_active": 0,
            "offline_bc_mse_real": initial_bc_metrics["mse"],
            "offline_bc_mae_real": initial_bc_metrics["mae"],
            "offline_ref_mse_mixed": initial_ref_mse,
            "temperature_loss": float("nan"),
            "dynamics_train_loss": float(dynamics_train_metrics["dynamics_loss"]),
            "dynamics_val_loss": float(dynamics_val_metrics["next_obs_mse"]),
            "dynamics_reward_loss": float(dynamics_train_metrics["dynamics_reward_loss"]),
            "dynamics_done_loss": float(dynamics_train_metrics["dynamics_done_loss"]),
            "uncertainty_mean": float("nan"),
            "root_uncertainty_mean": float("nan"),
            "synthetic_return_mean": float("nan"),
            **_empty_ages_metrics(),
            "dynamics_qpos_root_loss": float(dynamics_train_metrics.get("dynamics_qpos_root_loss", float("nan"))),
            "dynamics_qvel_root_loss": float(dynamics_train_metrics.get("dynamics_qvel_root_loss", float("nan"))),
            "dynamics_qpos_root_next_mse": float(dynamics_val_metrics.get("qpos_root_next_mse", float("nan"))),
            "dynamics_qvel_root_next_mse": float(dynamics_val_metrics.get("qvel_root_next_mse", float("nan"))),
            "formula_reward_mae": float(dynamics_val_metrics.get("formula_reward_mae", float("nan"))),
            "online_avg_return": float("nan")
            if initial_online_metrics is None
            else float(initial_online_metrics["avg_return"]),
            "online_avg_length": float("nan")
            if initial_online_metrics is None
            else float(initial_online_metrics["avg_length"]),
            "online_survived_full": ""
            if initial_online_metrics is None
            else int(initial_online_metrics["survived_full"]),
            "wall_time_sec": float(time.perf_counter() - start_time),
        }
        writer.writerow(initial_row)
        csv_file.flush()
        print(json.dumps(initial_row, sort_keys=True))

        epoch = 0
        for epoch in range(1, args.epochs + 1):
            start_count = max(1, int(round(args.rollout_batch_size * args.model_rollout_ratio)))
            start_idx = torch.randint(0, train.size, (min(start_count, train.size),), device=device)
            start_obs = torch.as_tensor(
                train.observation[start_idx.detach().cpu().numpy()],
                dtype=torch.float32,
                device=device,
            )
            start_np_idx = start_idx.detach().cpu().numpy()
            start_qpos_root = None
            start_qvel_root = None
            if args.predict_root:
                assert train.qpos_root is not None and train.qvel_root is not None
                start_qpos_root = torch.as_tensor(
                    train.qpos_root[start_np_idx],
                    dtype=torch.float32,
                    device=device,
                )
                start_qvel_root = torch.as_tensor(
                    train.qvel_root[start_np_idx],
                    dtype=torch.float32,
                    device=device,
                )
            synthetic_batch, rollout_metrics = _generate_model_rollouts(
                actor=actor,
                dynamics=dynamics,
                start_obs=start_obs,
                start_qpos_root=start_qpos_root,
                start_qvel_root=start_qvel_root,
                rollout_horizon=args.rollout_horizon,
                mopo_penalty_coef=args.mopo_penalty_coef,
                obs_mean=obs_mean_t,
                obs_std=obs_std_t,
                force_last_action=args.force_last_action,
                done_threshold=args.done_threshold,
                reward_scale=args.reward_scale,
                formula_reward=bool(args.formula_reward),
                formula_done=bool(args.formula_done),
                observable_reward_context=getattr(args, "_observable_reward_context", None),
            )
            metric_lists["uncertainty_mean"].append(rollout_metrics["uncertainty_mean"])
            metric_lists["synthetic_reward_mean"].append(rollout_metrics["synthetic_reward_mean"])
            ages_metrics = _empty_ages_metrics()
            ages_metrics.update(
                {
                    "ages_sim_ratio": float(args.ages_sim_ratio),
                    "ages_select_ratio": float(args.ages_select_ratio),
                    "ages_action_temperature": float(args.ages_action_temperature),
                    "ages_buffer_size": float(_ages_buffer_size(ages_buffer)),
                }
            )
            if args.use_ages and epoch % int(args.ages_generation_interval) == 0:
                ages_batch, ages_metrics = _generate_ages_rollouts(
                    actor=actor,
                    dynamics=dynamics,
                    arrays=train,
                    device=device,
                    num_dataset_states=args.ages_num_dataset_states,
                    trajectories_per_state=args.ages_trajectories_per_state,
                    sim_rollout_length=args.ages_sim_rollout_length,
                    inject_rollout_length=args.ages_inject_rollout_length,
                    action_temperature=args.ages_action_temperature,
                    select_ratio=args.ages_select_ratio,
                    select_mode=args.ages_select_mode,
                    mopo_penalty_coef=args.mopo_penalty_coef,
                    reward_scale=args.reward_scale,
                    discount=args.discount,
                    obs_mean=obs_mean_t,
                    obs_std=obs_std_t,
                    force_last_action=args.force_last_action,
                    done_threshold=args.done_threshold,
                    formula_reward=bool(args.formula_reward),
                    formula_done=bool(args.formula_done),
                    observable_reward_context=getattr(args, "_observable_reward_context", None),
                    score_use_uncertainty=bool(args.ages_score_use_uncertainty),
                    score_use_mopo_penalty=bool(args.ages_score_use_mopo_penalty),
                )
                ages_buffer.append((epoch, ages_batch))
                ages_buffer = _trim_ages_buffer(
                    ages_buffer,
                    current_epoch=epoch,
                    retain_epochs=args.ages_buffer_retain_epochs,
                    max_size=args.ages_max_buffer_size,
                )
                ages_metrics["ages_buffer_size"] = float(_ages_buffer_size(ages_buffer))
                ages_metrics["ages_sim_ratio"] = float(args.ages_sim_ratio)
                ages_metrics["ages_select_ratio"] = float(args.ages_select_ratio)
                ages_metrics["ages_action_temperature"] = float(args.ages_action_temperature)
                ages_stats_file.write(json.dumps({"epoch": epoch, "update": update_step, **ages_metrics}) + "\n")
                ages_stats_file.flush()
            for key in (
                "ages_candidate_return_mean",
                "ages_candidate_return_std",
                "ages_selected_return_mean",
                "ages_selected_adv_mean",
                "ages_selected_adv_min",
                "ages_selected_adv_max",
                "ages_accept_rate",
                "ages_buffer_size",
                "ages_uncertainty_mean",
                "ages_generation_wall_time",
            ):
                metric_lists[key].append(float(ages_metrics.get(key, float("nan"))))

            updates_per_epoch = args.updates_per_epoch
            if updates_per_epoch is None:
                updates_per_epoch = max(1, math.ceil(train.size / args.batch_size))
            critic_warmup_active = epoch <= int(args.critic_warmup_epochs)
            epoch_actor_losses = []
            epoch_actor_rl_losses = []
            epoch_actor_rl_scaled_losses = []
            epoch_critic_losses = []
            epoch_bc_losses = []
            epoch_bc_ref_losses = []
            epoch_temp_losses = []
            for _ in range(updates_per_epoch):
                real_count = int(round(args.batch_size * args.real_ratio))
                real_count = min(args.batch_size, max(0, real_count))
                generated_count = args.batch_size - real_count
                ages_count = 0
                ages_available = args.use_ages and _ages_buffer_size(ages_buffer) >= int(args.ages_min_buffer_size)
                if generated_count > 0 and ages_available:
                    ages_count = int(round(generated_count * float(args.ages_sim_ratio)))
                    ages_count = min(generated_count, max(0, ages_count))
                synthetic_count = generated_count - ages_count
                batches = []
                real_batch = None
                synthetic_update_batch = None
                synthetic_update_parts = []
                if real_count > 0:
                    real_batch = _make_real_batch(
                        train,
                        real_count,
                        device,
                        reward_scale=args.reward_scale,
                    )
                    batches.append(real_batch)
                if synthetic_count > 0:
                    synthetic_update_parts.append(_sample_from_torch_batch(synthetic_batch, synthetic_count))
                if ages_count > 0:
                    synthetic_update_parts.append(_sample_from_ages_buffer(ages_buffer, ages_count))
                if synthetic_update_parts:
                    synthetic_update_batch = (
                        _concat_batches(synthetic_update_parts)
                        if len(synthetic_update_parts) > 1
                        else synthetic_update_parts[0]
                    )
                    batches.append(synthetic_update_batch)
                if not batches:
                    raise RuntimeError("No samples available for policy update; check real_ratio/ages settings.")
                batch = _concat_batches(batches) if len(batches) > 1 else batches[0]

                critic_info = update_critic(
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    temperature=temperature,
                    batch=batch,  # type: ignore[arg-type]
                    min_v=args.critic_min_v,
                    max_v=args.critic_max_v,
                    num_bins=args.critic_num_bins,
                    gamma=args.discount,
                    n_step=1,
                    device=device,
                    use_amp=False,
                    grad_scaler=None,
                )
                actor_info: dict[str, torch.Tensor] = {}
                temperature_info: dict[str, torch.Tensor] = {}
                if critic_warmup_active:
                    if args.auto_alpha:
                        with torch.no_grad():
                            _, warmup_info = actor(observations=batch["actor_observation"], training=True)
                            warmup_entropy = -warmup_info["log_prob"].mean()
                        temperature_info = update_temperature(
                            temperature=temperature,
                            entropy=warmup_entropy,
                            target_entropy=args.target_entropy,
                        )
                elif update_step % args.actor_update_period == 0:
                    actor_info = _update_actor_bc_mopo(
                        actor=actor,
                        critic=critic,
                        temperature=temperature,
                        reference_actor=reference_actor,
                        mixed_batch=batch,
                        real_batch=real_batch,
                        synthetic_batch=synthetic_update_batch,
                        bc_alpha=args.bc_alpha,
                        bc_ref_alpha=args.bc_ref_alpha,
                        bc_ref_on=args.bc_ref_on,
                        actor_rl_coef=args.actor_rl_coef,
                        grad_clip_norm=args.grad_clip_norm,
                    )
                    if args.auto_alpha:
                        temperature_info = update_temperature(
                            temperature=temperature,
                            entropy=actor_info["actor/entropy"],
                            target_entropy=args.target_entropy,
                        )
                update_target_network(target_critic)
                update_step += 1

                record = {
                    "step": update_step,
                    "critic/loss": float(critic_info["critic/loss"].item()),
                    "actor/loss": float(actor_info["actor/loss"].item()) if actor_info else float("nan"),
                    "actor/rl_loss": float(actor_info["actor/rl_loss"].item()) if actor_info else float("nan"),
                    "actor/rl_loss_raw": float(actor_info["actor/rl_loss_raw"].item()) if actor_info else float("nan"),
                    "actor/rl_loss_scaled": float(actor_info["actor/rl_loss_scaled"].item())
                    if actor_info
                    else float("nan"),
                    "actor/bc_loss": float(actor_info["actor/bc_loss"].item()) if actor_info else float("nan"),
                    "actor/bc_ref_loss": float(actor_info["actor/bc_ref_loss"].item()) if actor_info else float("nan"),
                    "actor/entropy": float(actor_info["actor/entropy"].item()) if actor_info else float("nan"),
                    "critic_warmup_active": float(critic_warmup_active),
                    "temperature/loss": float(temperature_info["temperature/loss"].item())
                    if temperature_info
                    else float("nan"),
                    **rollout_metrics,
                    **ages_metrics,
                }
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                epoch_critic_losses.append(record["critic/loss"])
                metric_lists["train_critic_loss"].append(record["critic/loss"])
                if actor_info:
                    epoch_actor_losses.append(record["actor/loss"])
                    metric_lists["train_actor_loss"].append(record["actor/loss"])
                    epoch_actor_rl_losses.append(record["actor/rl_loss"])
                    metric_lists["train_actor_rl_loss"].append(record["actor/rl_loss"])
                    epoch_actor_rl_scaled_losses.append(record["actor/rl_loss_scaled"])
                    metric_lists["train_actor_rl_loss_scaled"].append(record["actor/rl_loss_scaled"])
                    if args.bc_alpha > 0:
                        epoch_bc_losses.append(record["actor/bc_loss"])
                        metric_lists["train_bc_loss"].append(record["actor/bc_loss"])
                    if args.bc_ref_alpha > 0:
                        epoch_bc_ref_losses.append(record["actor/bc_ref_loss"])
                        metric_lists["train_bc_ref_loss"].append(record["actor/bc_ref_loss"])
                if temperature_info:
                    epoch_temp_losses.append(record["temperature/loss"])
                    metric_lists["temperature_loss"].append(record["temperature/loss"])

                if args.max_train_steps is not None and update_step >= args.max_train_steps:
                    stop_training = True
                    break

            offline_bc_metrics = _evaluate_actor_bc_real(
                actor=actor,
                arrays=train,
                device=device,
                max_transitions=args.offline_bc_eval_transitions,
                seed=args.seed + 50_000 + epoch,
            )
            metric_lists["offline_bc_mse_real"].append(offline_bc_metrics["mse"])
            metric_lists["offline_bc_mae_real"].append(offline_bc_metrics["mae"])
            ref_eval_batch = None
            if reference_actor is not None:
                ref_eval_parts = []
                ref_eval_total = min(
                    args.offline_bc_eval_transitions, train.size + synthetic_batch["observation"].shape[0]
                )
                ref_real_count = min(train.size, max(1, ref_eval_total // 2))
                ref_synth_count = max(0, ref_eval_total - ref_real_count)
                if ref_real_count > 0:
                    ref_eval_parts.append(
                        _make_real_batch(train, ref_real_count, device, reward_scale=args.reward_scale)
                    )
                if ref_synth_count > 0:
                    if args.use_ages and _ages_buffer_size(ages_buffer) > 0:
                        ref_ages_count = ref_synth_count // 2
                        ref_plain_count = ref_synth_count - ref_ages_count
                        if ref_plain_count > 0:
                            ref_eval_parts.append(_sample_from_torch_batch(synthetic_batch, ref_plain_count))
                        if ref_ages_count > 0:
                            ref_eval_parts.append(_sample_from_ages_buffer(ages_buffer, ref_ages_count))
                    else:
                        ref_eval_parts.append(_sample_from_torch_batch(synthetic_batch, ref_synth_count))
                if ref_eval_parts:
                    ref_eval_batch = _concat_batches(ref_eval_parts) if len(ref_eval_parts) > 1 else ref_eval_parts[0]
            offline_ref_mse_mixed = _evaluate_actor_reference_mse(
                actor=actor,
                reference_actor=reference_actor,
                batch=ref_eval_batch,
            )
            metric_lists["offline_ref_mse_mixed"].append(offline_ref_mse_mixed)

            online_metrics = None
            if args.online_eval_interval > 0 and (
                epoch % args.online_eval_interval == 0 or stop_training or epoch == args.epochs
            ):
                eval_checkpoint = output_dir / f"epoch_{epoch:04d}"
                current_metrics = {
                    "dynamics_val_metrics": dynamics_val_metrics,
                    "dynamics_train_metrics": dynamics_train_metrics,
                    "rollout_metrics": rollout_metrics,
                    "ages_metrics": ages_metrics,
                    "offline_bc_metrics": offline_bc_metrics,
                }
                _save_checkpoint(
                    eval_checkpoint,
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    temperature=temperature,
                    dynamics=dynamics,
                    update_step=update_step,
                    epoch=epoch,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    root_stats=root_stats,
                    args=args,
                    metrics=current_metrics,
                )
                online_metrics = _run_online_eval(
                    eval_checkpoint,
                    args=args,
                    epoch=epoch,
                    update_step=update_step,
                    output_dir=output_dir,
                )
                online_eval_metrics.append(online_metrics)
                if _online_better(online_metrics, best_online):
                    best_online = online_metrics
                    _save_checkpoint(
                        output_dir / "best_online",
                        actor=actor,
                        critic=critic,
                        target_critic=target_critic,
                        temperature=temperature,
                        dynamics=dynamics,
                        update_step=update_step,
                        epoch=epoch,
                        obs_mean=obs_mean,
                        obs_std=obs_std,
                        root_stats=root_stats,
                        args=args,
                        metrics={
                            "online_eval": online_metrics,
                            "rollout_metrics": rollout_metrics,
                            "ages_metrics": ages_metrics,
                            "offline_bc_metrics": offline_bc_metrics,
                        },
                    )

            if epoch == 1 or epoch % args.save_interval == 0 or stop_training or epoch == args.epochs:
                _save_checkpoint(
                    output_dir / f"epoch_{epoch:04d}",
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    temperature=temperature,
                    dynamics=dynamics,
                    update_step=update_step,
                    epoch=epoch,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    root_stats=root_stats,
                    args=args,
                    metrics={
                        "dynamics_val_metrics": dynamics_val_metrics,
                        "dynamics_train_metrics": dynamics_train_metrics,
                        "rollout_metrics": rollout_metrics,
                        "ages_metrics": ages_metrics,
                        "offline_bc_metrics": offline_bc_metrics,
                    },
                )

            row = {
                "epoch": epoch,
                "update": update_step,
                "train_actor_loss": float(np.nanmean(epoch_actor_losses)) if epoch_actor_losses else float("nan"),
                "train_actor_rl_loss": float(np.nanmean(epoch_actor_rl_losses))
                if epoch_actor_rl_losses
                else float("nan"),
                "train_actor_rl_loss_raw": float(np.nanmean(epoch_actor_rl_losses))
                if epoch_actor_rl_losses
                else float("nan"),
                "train_actor_rl_loss_scaled": float(np.nanmean(epoch_actor_rl_scaled_losses))
                if epoch_actor_rl_scaled_losses
                else float("nan"),
                "train_critic_loss": float(np.mean(epoch_critic_losses)) if epoch_critic_losses else float("nan"),
                "train_bc_loss": float(np.mean(epoch_bc_losses)) if epoch_bc_losses else float("nan"),
                "train_bc_ref_loss": float(np.mean(epoch_bc_ref_losses)) if epoch_bc_ref_losses else float("nan"),
                "train_bc_ref_alpha": float(args.bc_ref_alpha),
                "actor_rl_coef": float(args.actor_rl_coef),
                "critic_warmup_active": int(critic_warmup_active),
                "offline_bc_mse_real": offline_bc_metrics["mse"],
                "offline_bc_mae_real": offline_bc_metrics["mae"],
                "offline_ref_mse_mixed": offline_ref_mse_mixed,
                "temperature_loss": float(np.nanmean(epoch_temp_losses)) if epoch_temp_losses else float("nan"),
                "dynamics_train_loss": float(dynamics_train_metrics["dynamics_loss"]),
                "dynamics_val_loss": float(dynamics_val_metrics["next_obs_mse"]),
                "dynamics_reward_loss": float(dynamics_train_metrics["dynamics_reward_loss"]),
                "dynamics_done_loss": float(dynamics_train_metrics["dynamics_done_loss"]),
                "uncertainty_mean": float(rollout_metrics["uncertainty_mean"]),
                "root_uncertainty_mean": float(rollout_metrics.get("root_uncertainty_mean", float("nan"))),
                "synthetic_return_mean": float(rollout_metrics["synthetic_reward_mean"]),
                "ages_candidate_return_mean": float(ages_metrics.get("ages_candidate_return_mean", float("nan"))),
                "ages_candidate_return_std": float(ages_metrics.get("ages_candidate_return_std", float("nan"))),
                "ages_selected_return_mean": float(ages_metrics.get("ages_selected_return_mean", float("nan"))),
                "ages_selected_adv_mean": float(ages_metrics.get("ages_selected_adv_mean", float("nan"))),
                "ages_selected_adv_min": float(ages_metrics.get("ages_selected_adv_min", float("nan"))),
                "ages_selected_adv_max": float(ages_metrics.get("ages_selected_adv_max", float("nan"))),
                "ages_accept_rate": float(ages_metrics.get("ages_accept_rate", float("nan"))),
                "ages_buffer_size": float(ages_metrics.get("ages_buffer_size", float("nan"))),
                "ages_uncertainty_mean": float(ages_metrics.get("ages_uncertainty_mean", float("nan"))),
                "ages_generation_wall_time": float(ages_metrics.get("ages_generation_wall_time", float("nan"))),
                "ages_sim_ratio": float(ages_metrics.get("ages_sim_ratio", float("nan"))),
                "ages_select_ratio": float(ages_metrics.get("ages_select_ratio", float("nan"))),
                "ages_action_temperature": float(ages_metrics.get("ages_action_temperature", float("nan"))),
                "dynamics_qpos_root_loss": float(dynamics_train_metrics.get("dynamics_qpos_root_loss", float("nan"))),
                "dynamics_qvel_root_loss": float(dynamics_train_metrics.get("dynamics_qvel_root_loss", float("nan"))),
                "dynamics_qpos_root_next_mse": float(dynamics_val_metrics.get("qpos_root_next_mse", float("nan"))),
                "dynamics_qvel_root_next_mse": float(dynamics_val_metrics.get("qvel_root_next_mse", float("nan"))),
                "formula_reward_mae": float(dynamics_val_metrics.get("formula_reward_mae", float("nan"))),
                "online_avg_return": float("nan") if online_metrics is None else float(online_metrics["avg_return"]),
                "online_avg_length": float("nan") if online_metrics is None else float(online_metrics["avg_length"]),
                "online_survived_full": "" if online_metrics is None else int(online_metrics["survived_full"]),
                "wall_time_sec": float(time.perf_counter() - start_time),
            }
            writer.writerow(row)
            csv_file.flush()
            print(json.dumps(row, sort_keys=True))

            if float(dynamics_val_metrics["next_obs_mse"]) < best_model_loss:
                best_model_loss = float(dynamics_val_metrics["next_obs_mse"])
                _save_checkpoint(
                    output_dir / "best_model",
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    temperature=temperature,
                    dynamics=dynamics,
                    update_step=update_step,
                    epoch=epoch,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    root_stats=root_stats,
                    args=args,
                    metrics={
                        "dynamics_val_metrics": dynamics_val_metrics,
                        "rollout_metrics": rollout_metrics,
                        "ages_metrics": ages_metrics,
                        "offline_bc_metrics": offline_bc_metrics,
                    },
                )

            if stop_training:
                break

    _save_checkpoint(
        output_dir / "final",
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        dynamics=dynamics,
        update_step=update_step,
        epoch=epoch,
        obs_mean=obs_mean,
        obs_std=obs_std,
        root_stats=root_stats,
        args=args,
        metrics={"dynamics_val_metrics": dynamics_val_metrics, "best_online": best_online, "ages_metrics": ages_metrics},
    )

    metrics = {
        "status": "passed",
        "seed": args.seed,
        "device": str(device),
        "dataset_path": str(args.dataset.expanduser().resolve()),
        "bc_checkpoint": None if args.bc_checkpoint is None else str(args.bc_checkpoint.expanduser().resolve()),
        "dynamics_checkpoint": None if dynamics_checkpoint is None else str(dynamics_checkpoint),
        "skip_dynamics_training": bool(args.skip_dynamics_training),
        "output_dir": str(output_dir),
        **asdict(split),
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "max_train_steps": args.max_train_steps,
        "update_steps": update_step,
        "batch_size": args.batch_size,
        "real_ratio": args.real_ratio,
        "rollout_horizon": args.rollout_horizon,
        "mopo_penalty_coef": args.mopo_penalty_coef,
        "reward_scale": args.reward_scale,
        "predict_root": bool(args.predict_root),
        "predict_reward_head": bool(args.predict_reward_head),
        "predict_done_head": bool(args.predict_done_head),
        "formula_reward": bool(args.formula_reward),
        "formula_done": bool(args.formula_done),
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "temperature_lr": args.temperature_lr,
        "actor_rl_coef": args.actor_rl_coef,
        "bc_alpha": args.bc_alpha,
        "bc_ref_alpha": args.bc_ref_alpha,
        "bc_ref_on": args.bc_ref_on,
        "use_ages": bool(args.use_ages),
        "ages_action_temperature": args.ages_action_temperature,
        "ages_sim_ratio": args.ages_sim_ratio,
        "ages_select_ratio": args.ages_select_ratio,
        "ages_num_dataset_states": args.ages_num_dataset_states,
        "ages_trajectories_per_state": args.ages_trajectories_per_state,
        "ages_sim_rollout_length": args.ages_sim_rollout_length,
        "ages_inject_rollout_length": args.ages_inject_rollout_length,
        "ages_buffer_retain_epochs": args.ages_buffer_retain_epochs,
        "ages_buffer_final_size": _ages_buffer_size(ages_buffer),
        "critic_warmup_epochs": args.critic_warmup_epochs,
        "actor_weight_normalization": args.actor_weight_normalization,
        "normalize_obs": args.normalize_obs,
        "reward_relabel": relabel_info,
        "dynamics_train_metrics": dynamics_train_metrics,
        "dynamics_val_metrics": dynamics_val_metrics,
        "train_actor_loss": _stats(metric_lists["train_actor_loss"]),
        "train_actor_rl_loss": _stats(metric_lists["train_actor_rl_loss"]),
        "train_actor_rl_loss_scaled": _stats(metric_lists["train_actor_rl_loss_scaled"]),
        "train_critic_loss": _stats(metric_lists["train_critic_loss"]),
        "train_bc_loss": _stats(metric_lists["train_bc_loss"]),
        "train_bc_ref_loss": _stats(metric_lists["train_bc_ref_loss"]),
        "offline_bc_mse_real": _stats(metric_lists["offline_bc_mse_real"]),
        "offline_bc_mae_real": _stats(metric_lists["offline_bc_mae_real"]),
        "offline_ref_mse_mixed": _stats(metric_lists["offline_ref_mse_mixed"]),
        "temperature_loss": _stats(metric_lists["temperature_loss"]),
        "uncertainty_mean": _stats(metric_lists["uncertainty_mean"]),
        "synthetic_return_mean": _stats(metric_lists["synthetic_reward_mean"]),
        "ages_candidate_return_mean": _stats(metric_lists["ages_candidate_return_mean"]),
        "ages_candidate_return_std": _stats(metric_lists["ages_candidate_return_std"]),
        "ages_selected_return_mean": _stats(metric_lists["ages_selected_return_mean"]),
        "ages_selected_adv_mean": _stats(metric_lists["ages_selected_adv_mean"]),
        "ages_selected_adv_min": _stats(metric_lists["ages_selected_adv_min"]),
        "ages_selected_adv_max": _stats(metric_lists["ages_selected_adv_max"]),
        "ages_accept_rate": _stats(metric_lists["ages_accept_rate"]),
        "ages_buffer_size": _stats(metric_lists["ages_buffer_size"]),
        "ages_uncertainty_mean": _stats(metric_lists["ages_uncertainty_mean"]),
        "ages_generation_wall_time": _stats(metric_lists["ages_generation_wall_time"]),
        "online_eval_interval": args.online_eval_interval,
        "online_eval_count": len(online_eval_metrics),
        "online_eval_last": online_eval_metrics[-1] if online_eval_metrics else None,
        "best_online_eval": best_online,
        "best_model_checkpoint": str((output_dir / "best_model").resolve()),
        "best_online_checkpoint": str((output_dir / "best_online").resolve()) if best_online is not None else None,
        "final_checkpoint": str((output_dir / "final").resolve()),
        "progress_csv": str(progress_path.resolve()),
        "train_log": str(log_path.resolve()),
        "online_eval_csv": str((output_dir / "online_eval.csv").resolve()) if online_eval_metrics else None,
        "wall_time_sec": float(time.perf_counter() - start_time),
    }
    _write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MuJoCo Playground G1 HRLG AGES-MOPO baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--no-bc-checkpoint",
        action="store_true",
        help="Start from a randomly initialized actor instead of warm-starting from BC.",
    )
    parser.add_argument(
        "--dynamics-only",
        action="store_true",
        help="Train and save only the dynamics ensemble, then exit before policy updates.",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint directory containing dynamics_ensemble.pt.",
    )
    parser.add_argument(
        "--skip-dynamics-training",
        action="store_true",
        help="Load --dynamics-checkpoint and skip dynamics training before policy updates.",
    )
    parser.add_argument(
        "--allow-dynamics-split-mismatch",
        action="store_true",
        help="Allow tiny smoke tests to reuse a dynamics checkpoint trained on a different trajectory split.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-split-seed",
        type=int,
        default=None,
        help="Seed used only for train/val trajectory split; defaults to --seed.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--updates-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--temperature-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--actor-weight-normalization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-trajectories", type=int, default=None)
    parser.add_argument("--limit-transitions", type=int, default=None)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-source", choices=["observable", "relabel_env", "zero"], default="relabel_env")
    parser.add_argument("--reward-relabel-batch-size", type=int, default=512)
    parser.add_argument("--dynamics-ensemble-size", type=int, default=5)
    parser.add_argument("--dynamics-hidden-dim", type=int, default=256)
    parser.add_argument("--dynamics-layers", type=int, default=3)
    parser.add_argument("--dynamics-epochs", type=int, default=50)
    parser.add_argument("--dynamics-steps", type=int, default=None)
    parser.add_argument("--predict-root", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--predict-reward-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict-done-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formula-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--formula-done", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--root-loss-coef", type=float, default=1.0)
    parser.add_argument("--qpos-root-loss-coef", type=float, default=1.0)
    parser.add_argument("--qvel-root-loss-coef", type=float, default=1.0)
    parser.add_argument("--quat-loss-coef", type=float, default=1.0)
    parser.add_argument("--reward-loss-coef", type=float, default=1.0)
    parser.add_argument("--done-loss-coef", type=float, default=0.1)
    parser.add_argument("--rollout-horizon", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=4096)
    parser.add_argument("--model-rollout-ratio", type=float, default=1.0)
    parser.add_argument("--real-ratio", type=float, default=0.9)
    parser.add_argument("--mopo-penalty-coef", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=50.0)
    parser.add_argument("--done-threshold", type=float, default=0.5)
    parser.add_argument("--force-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--auto-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--bc-alpha", type=float, default=10.0)
    parser.add_argument("--use-ages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ages-action-temperature", type=float, default=1.0)
    parser.add_argument("--ages-sim-ratio", type=float, default=0.0)
    parser.add_argument("--ages-select-ratio", type=float, default=0.25)
    parser.add_argument("--ages-num-dataset-states", type=int, default=1024)
    parser.add_argument("--ages-trajectories-per-state", type=int, default=4)
    parser.add_argument("--ages-baseline", choices=["same_state_mean"], default="same_state_mean")
    parser.add_argument("--ages-select-mode", choices=["global_top_ratio", "per_state_top1"], default="global_top_ratio")
    parser.add_argument("--ages-buffer-retain-epochs", type=int, default=5)
    parser.add_argument("--ages-sim-rollout-length", type=int, default=1)
    parser.add_argument("--ages-inject-rollout-length", type=int, default=1)
    parser.add_argument("--ages-generation-interval", type=int, default=1)
    parser.add_argument("--ages-min-buffer-size", type=int, default=512)
    parser.add_argument("--ages-max-buffer-size", type=int, default=200000)
    parser.add_argument("--ages-score-use-uncertainty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ages-score-use-mopo-penalty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--actor-rl-coef",
        type=float,
        default=1.0,
        help="Scale applied to the SAC actor RL loss before adding BC/reference regularization.",
    )
    parser.add_argument(
        "--bc-ref-alpha",
        type=float,
        default=0.0,
        help="Weight for a frozen-BC reference action loss on selected observations.",
    )
    parser.add_argument(
        "--bc-ref-on",
        choices=["mixed", "synthetic", "real"],
        default="mixed",
        help="Observation batch used for the frozen-BC reference loss.",
    )
    parser.add_argument(
        "--critic-warmup-epochs",
        type=int,
        default=0,
        help="Number of initial epochs that update critic/temperature but skip actor updates.",
    )
    parser.add_argument("--actor-update-period", type=int, default=1)
    parser.add_argument("--critic-num-blocks", type=int, default=2)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-num-bins", type=int, default=101)
    parser.add_argument("--critic-min-v", type=float, default=-5.0)
    parser.add_argument("--critic-max-v", type=float, default=50.0)
    parser.add_argument("--offline-bc-eval-transitions", type=int, default=8192)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Reserved for compatibility; dynamics eval is run once.",
    )
    parser.add_argument("--online-eval-interval", type=int, default=0)
    parser.add_argument("--online-eval-episodes", type=int, default=50)
    parser.add_argument("--online-eval-num-envs", type=int, default=50)
    parser.add_argument("--online-eval-max-episode-steps", type=int, default=1000)
    parser.add_argument("--online-eval-seed", type=int, default=None)
    parser.add_argument("--online-eval-device", type=str, default="cpu")
    parser.add_argument("--online-eval-noise-level", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()
    if args.no_bc_checkpoint:
        args.bc_checkpoint = None
    if args.target_entropy is None:
        args.target_entropy = 0.5 * hrlg.ACTION_SIZE * math.log(2 * math.pi * math.e * 0.5**2)
    if args.actor_rl_coef < 0:
        raise ValueError("--actor-rl-coef must be non-negative.")
    if args.bc_ref_alpha < 0:
        raise ValueError("--bc-ref-alpha must be non-negative.")
    if args.critic_warmup_epochs < 0:
        raise ValueError("--critic-warmup-epochs must be non-negative.")
    if args.bc_ref_alpha > 0 and args.bc_checkpoint is None:
        raise ValueError("--bc-ref-alpha > 0 requires --bc-checkpoint.")
    if args.use_ages:
        if args.ages_action_temperature < 0:
            raise ValueError("--ages-action-temperature must be non-negative.")
        if not (0.0 <= args.ages_sim_ratio <= 1.0):
            raise ValueError("--ages-sim-ratio must be in [0, 1].")
        if not (0.0 < args.ages_select_ratio <= 1.0):
            raise ValueError("--ages-select-ratio must be in (0, 1].")
        if args.ages_num_dataset_states <= 0:
            raise ValueError("--ages-num-dataset-states must be positive.")
        if args.ages_trajectories_per_state <= 0:
            raise ValueError("--ages-trajectories-per-state must be positive.")
        if args.ages_sim_rollout_length <= 0:
            raise ValueError("--ages-sim-rollout-length must be positive.")
        if args.ages_inject_rollout_length <= 0:
            raise ValueError("--ages-inject-rollout-length must be positive.")
        if args.ages_generation_interval <= 0:
            raise ValueError("--ages-generation-interval must be positive.")
        if args.ages_min_buffer_size < 0:
            raise ValueError("--ages-min-buffer-size must be non-negative.")
        if args.ages_max_buffer_size <= 0:
            raise ValueError("--ages-max-buffer-size must be positive.")
    _run_training(args)


if __name__ == "__main__":
    main()
