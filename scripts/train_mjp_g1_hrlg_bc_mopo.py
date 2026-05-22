#!/usr/bin/env python3
"""BC-initialized MOPO-style training for MuJoCo Playground G1 HRLG.

The current HRLG buffer does not contain exact reward/done labels or the full
MuJoCo state.  This script can relabel rewards/dones by reconstructing the
available simulator state, but marks those labels as approximate because some
reward inputs such as foot air-time history and sampled phase frequency are not
stored in the dataset.
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
from flash_rl.agents.flashSAC.update import update_actor, update_critic, update_target_network, update_temperature
from flash_rl.agents.utils.network import Network
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import constants as hrlg


DEFAULT_DATASET = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"
DEFAULT_BC_CHECKPOINT = REPO_ROOT / "models/mjp_g1_hrlg_bc/seed0_100traj/epoch_0150"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models/mjp_g1_hrlg_bc_mopo/seed0_100traj_h3_pen1"

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
        predict_reward: bool,
        predict_done: bool,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        extra_dim = int(predict_reward) + int(predict_done)
        output_dim = obs_dim + extra_dim
        self.members = nn.ModuleList(
            DynamicsMember(obs_dim + action_dim, output_dim, hidden_dim, layers)
            for _ in range(ensemble_size)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([member(x) for member in self.members], dim=0)

    def split_prediction(self, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        delta = pred[..., : self.obs_dim]
        offset = self.obs_dim
        reward = None
        done_logit = None
        if self.predict_reward:
            reward = pred[..., offset]
            offset += 1
        if self.predict_done:
            done_logit = pred[..., offset]
        return delta, reward, done_logit


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
    if bool(ckpt.get("predict_reward", dynamics.predict_reward)) != bool(dynamics.predict_reward):
        raise ValueError(
            f"Dynamics reward-head mismatch: checkpoint={ckpt.get('predict_reward')} "
            f"current={dynamics.predict_reward}"
        )
    if bool(ckpt.get("predict_done", dynamics.predict_done)) != bool(dynamics.predict_done):
        raise ValueError(
            f"Dynamics done-head mismatch: checkpoint={ckpt.get('predict_done')} "
            f"current={dynamics.predict_done}"
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
        reward = (
            np.zeros(obs.shape[0], dtype=np.float32)
            if dataset_reward is None
            else dataset_reward[traj].reshape(-1).copy()
        )
        terminated = (
            np.zeros(obs.shape[0], dtype=np.float32)
            if dataset_done is None
            else dataset_done[traj].reshape(-1).copy()
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
    )


def _sample_indices(size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, size, (batch_size,), device=device)


def _batch_from_arrays(
    arrays: TransitionArrays,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    idx = indices.detach().cpu().numpy()
    obs = torch.as_tensor(arrays.observation[idx], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(arrays.next_observation[idx], dtype=torch.float32, device=device)
    action = torch.as_tensor(arrays.action[idx], dtype=torch.float32, device=device)
    reward = torch.as_tensor(arrays.reward[idx], dtype=torch.float32, device=device)
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


def _make_real_batch(arrays: TransitionArrays, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return _batch_from_arrays(arrays, _sample_indices(arrays.size, batch_size, device), device)


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
    rollout_horizon: int,
    mopo_penalty_coef: float,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    force_last_action: bool,
    done_threshold: float,
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
    active = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
    ensemble_size = len(dynamics.members)

    for horizon_step in range(rollout_horizon):
        if not bool(torch.any(active)):
            break
        obs_active = obs[active]
        mean, _ = actor.apply("get_mean_and_std", observations=obs_active, training=False)
        action = torch.tanh(mean)
        pred = dynamics(obs_active, action)
        delta, reward_pred, done_logit = dynamics.split_prediction(pred)
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
        variance_mean = next_per_model.var(dim=0).mean(dim=-1)
        uncertainty = variance_mean

        if reward_pred is None:
            reward_mean = -mopo_penalty_coef * uncertainty
        else:
            reward_mean = reward_pred.mean(dim=0)
        penalized_reward = reward_mean - mopo_penalty_coef * uncertainty

        if done_logit is None:
            done = torch.zeros_like(penalized_reward)
        else:
            done = (torch.sigmoid(done_logit.mean(dim=0)) > done_threshold).float()
        truncated = torch.zeros_like(done)
        if horizon_step == rollout_horizon - 1:
            truncated = 1.0 - done

        member_idx = torch.randint(0, ensemble_size, (obs_active.shape[0],), device=device)
        row_idx = torch.arange(obs_active.shape[0], device=device)
        next_sample = next_per_model[member_idx, row_idx]

        observations.append(obs_active)
        actions.append(action)
        next_observations.append(next_sample)
        rewards.append(penalized_reward)
        terminateds.append(done)
        truncateds.append(truncated)
        uncertainties.append(uncertainty)
        raw_rewards.append(reward_mean)

        obs_next = obs.clone()
        active_indices = active.nonzero(as_tuple=False).squeeze(-1)
        obs_next[active_indices] = next_sample
        obs = obs_next
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
    raw_reward_tensor = torch.cat(raw_rewards, dim=0)
    metrics = {
        "synthetic_transition_count": float(synthetic["observation"].shape[0]),
        "uncertainty_mean": float(uncertainty_tensor.mean().item()),
        "uncertainty_max": float(uncertainty_tensor.max().item()),
        "synthetic_reward_mean": float(synthetic["reward"].mean().item()),
        "synthetic_raw_reward_mean": float(raw_reward_tensor.mean().item()),
        "synthetic_done_fraction": float(synthetic["terminated"].mean().item()),
    }
    return synthetic, metrics


def _sample_from_torch_batch(batch: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    size = int(batch["observation"].shape[0])
    idx = torch.randint(0, size, (batch_size,), device=batch["observation"].device)
    return {key: value[idx] for key, value in batch.items()}


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
    reward_losses: list[float] = []
    done_losses: list[float] = []
    dynamics.train()
    for _ in range(total_steps):
        idx = torch.randint(0, train.size, (min(args.model_batch_size, train.size),), device=device)
        pred = dynamics(train_obs[idx], train_action[idx])
        delta, reward, done_logit = dynamics.split_prediction(pred)
        delta_target = train_delta[idx].unsqueeze(0).expand_as(delta)
        delta_loss = F.mse_loss(delta, delta_target)
        loss = delta_loss
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
        reward_losses.append(float(reward_loss.item()))
        done_losses.append(float(done_loss.item()))

    val_metrics = _evaluate_dynamics(
        dynamics=dynamics,
        obs=val_obs,
        action=val_action,
        delta=val_delta,
        reward=val_reward,
        done=val_done,
        obs_mean=torch.as_tensor(args._obs_mean, dtype=torch.float32, device=device),
        obs_std=torch.as_tensor(args._obs_std, dtype=torch.float32, device=device),
        force_last_action=args.force_last_action,
    )
    train_metrics = {
        "dynamics_loss": float(np.mean(losses)),
        "dynamics_delta_loss": float(np.mean(delta_losses)),
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
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    force_last_action: bool,
) -> dict[str, Any]:
    dynamics.eval()
    pred = dynamics(obs, action)
    pred_delta, pred_reward, pred_done_logit = dynamics.split_prediction(pred)
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
        "chunk_mse": {
            name: float(torch.mean(error[:, slc].square()).item())
            for name, slc in SLICES.items()
        },
    }
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
        obs_mean=torch.as_tensor(args._obs_mean, dtype=torch.float32, device=device),
        obs_std=torch.as_tensor(args._obs_std, dtype=torch.float32, device=device),
        force_last_action=args.force_last_action,
    )


def _make_flashsac_networks(
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Network, Network, Network, Network]:
    actor_net = _make_actor(device)
    if args.bc_checkpoint is not None:
        _load_actor_checkpoint(actor_net, args.bc_checkpoint.expanduser().resolve(), device)
    actor = _make_network_bundle(
        actor_net,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_weight_normalization=True,
    )
    if args.bc_checkpoint is None:
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
        lr=args.lr,
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
        lr=args.lr,
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
            "predict_reward": bool(dynamics.predict_reward),
            "predict_done": bool(dynamics.predict_done),
        },
        path / "dynamics_ensemble.pt",
    )
    np.savez(path / "obs_stats.npz", mean=obs_mean, std=obs_std, enabled=np.array(bool(args.normalize_obs)))
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
        "predict_reward": bool(dynamics.predict_reward),
        "predict_done": bool(dynamics.predict_done),
        "normalize_obs": bool(args.normalize_obs),
    }


def _save_dynamics_checkpoint(
    path: Path,
    *,
    dynamics: DynamicsEnsemble,
    update_step: int,
    epoch: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
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
            "predict_reward": bool(dynamics.predict_reward),
            "predict_done": bool(dynamics.predict_done),
        },
        path / "dynamics_ensemble.pt",
    )
    np.savez(path / "obs_stats.npz", mean=obs_mean, std=obs_std, enabled=np.array(bool(args.normalize_obs)))
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
    if args.skip_dynamics_training and args.dynamics_checkpoint is None:
        raise ValueError("--skip-dynamics-training requires --dynamics-checkpoint.")
    if args.dynamics_only and args.skip_dynamics_training:
        raise ValueError("--dynamics-only cannot be combined with --skip-dynamics-training.")
    dynamics_checkpoint = (
        _resolve_dynamics_checkpoint(args.dynamics_checkpoint)
        if args.dynamics_checkpoint is not None
        else None
    )

    train_raw, val_raw, split, relabel_info = _load_transitions(
        args.dataset.expanduser().resolve(),
        val_ratio=args.val_ratio,
        seed=args.seed,
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

    train = _normalize(train_raw, obs_mean, obs_std)
    val = _normalize(val_raw, obs_mean, obs_std)
    obs_mean_t = torch.as_tensor(obs_mean, dtype=torch.float32, device=device)
    obs_std_t = torch.as_tensor(obs_std, dtype=torch.float32, device=device)

    split_info = asdict(split)
    args._split_info = split_info
    args._relabel_info = relabel_info
    if dynamics_checkpoint is not None:
        _validate_dynamics_checkpoint_split(dynamics_checkpoint, split_info)
    config = _public_config(args)
    _write_json(output_dir / "resolved_config.json", config)
    offline_provenance = {
        "status": "offline_only_verified",
        "training_script": str(Path(__file__).resolve()),
        "dataset_path": str(args.dataset.expanduser().resolve()),
        "bc_checkpoint": None if args.bc_checkpoint is None else str(args.bc_checkpoint.expanduser().resolve()),
        "dynamics_checkpoint": None if dynamics_checkpoint is None else str(dynamics_checkpoint),
        "training_data_sources": ["mujoco_playground_hrlg_npz_dataset", "learned_dynamics_rollouts"],
        "uses_online_env_transitions_for_training": False,
        "uses_env_for_reward_relabel": args.reward_source == "relabel_env",
        "uses_observable_dataset_reward": args.reward_source == "observable",
        "uses_pretrained_dynamics": dynamics_checkpoint is not None,
        "skip_dynamics_training": bool(args.skip_dynamics_training),
        "dynamics_only": bool(args.dynamics_only),
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
        predict_reward=split.reward_available,
        predict_done=split.done_available,
    ).to(device)

    loaded_dynamics_metrics: dict[str, Any] | None = None
    if dynamics_checkpoint is not None:
        loaded_ckpt = _load_dynamics_checkpoint(dynamics, dynamics_checkpoint, device)
        loaded_dynamics_metrics = loaded_ckpt.get("metrics", {})
    if args.skip_dynamics_training:
        dynamics_train_metrics = {
            "dynamics_loss": float("nan"),
            "dynamics_delta_loss": float("nan"),
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
                args=args,
                metrics=metrics,
            )
        _write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics

    actor, critic, target_critic, temperature = _make_flashsac_networks(device=device, args=args)

    progress_path = output_dir / "progress.csv"
    log_path = output_dir / "train.log"
    start_time = time.perf_counter()
    update_step = 0
    stop_training = False
    best_model_loss = float(dynamics_val_metrics["next_obs_mse"])
    best_online: dict[str, Any] | None = None
    online_eval_metrics: list[dict[str, Any]] = []
    metric_lists: dict[str, list[float]] = {
        "train_actor_loss": [],
        "train_critic_loss": [],
        "train_bc_loss": [],
        "temperature_loss": [],
        "uncertainty_mean": [],
        "synthetic_reward_mean": [],
    }

    fieldnames = [
        "epoch",
        "update",
        "train_actor_loss",
        "train_critic_loss",
        "train_bc_loss",
        "temperature_loss",
        "dynamics_train_loss",
        "dynamics_val_loss",
        "dynamics_reward_loss",
        "dynamics_done_loss",
        "uncertainty_mean",
        "synthetic_return_mean",
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
        args=args,
        metrics={"dynamics_val_metrics": dynamics_val_metrics, "dynamics_train_metrics": dynamics_train_metrics},
    )

    with (
        progress_path.open("w", encoding="utf-8", newline="") as csv_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start_count = max(1, int(round(args.rollout_batch_size * args.model_rollout_ratio)))
            start_idx = torch.randint(0, train.size, (min(start_count, train.size),), device=device)
            start_obs = torch.as_tensor(
                train.observation[start_idx.detach().cpu().numpy()],
                dtype=torch.float32,
                device=device,
            )
            synthetic_batch, rollout_metrics = _generate_model_rollouts(
                actor=actor,
                dynamics=dynamics,
                start_obs=start_obs,
                rollout_horizon=args.rollout_horizon,
                mopo_penalty_coef=args.mopo_penalty_coef,
                obs_mean=obs_mean_t,
                obs_std=obs_std_t,
                force_last_action=args.force_last_action,
                done_threshold=args.done_threshold,
            )
            metric_lists["uncertainty_mean"].append(rollout_metrics["uncertainty_mean"])
            metric_lists["synthetic_reward_mean"].append(rollout_metrics["synthetic_reward_mean"])

            updates_per_epoch = args.updates_per_epoch
            if updates_per_epoch is None:
                updates_per_epoch = max(1, math.ceil(train.size / args.batch_size))
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_bc_losses = []
            epoch_temp_losses = []
            for _ in range(updates_per_epoch):
                real_count = int(round(args.batch_size * args.real_ratio))
                real_count = min(args.batch_size, max(0, real_count))
                synthetic_count = args.batch_size - real_count
                batches = []
                if real_count > 0:
                    batches.append(_make_real_batch(train, real_count, device))
                if synthetic_count > 0:
                    batches.append(_sample_from_torch_batch(synthetic_batch, synthetic_count))
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
                if update_step % args.actor_update_period == 0:
                    actor_info = update_actor(
                        actor=actor,
                        critic=critic,
                        temperature=temperature,
                        batch=batch,  # type: ignore[arg-type]
                        bc_alpha=args.bc_alpha,
                        device=device,
                        use_amp=False,
                        grad_scaler=None,
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
                    "actor/entropy": float(actor_info["actor/entropy"].item()) if actor_info else float("nan"),
                    "temperature/loss": float(temperature_info["temperature/loss"].item())
                    if temperature_info
                    else float("nan"),
                    **rollout_metrics,
                }
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                epoch_critic_losses.append(record["critic/loss"])
                metric_lists["train_critic_loss"].append(record["critic/loss"])
                if actor_info:
                    epoch_actor_losses.append(record["actor/loss"])
                    metric_lists["train_actor_loss"].append(record["actor/loss"])
                    if args.bc_alpha > 0:
                        with torch.no_grad():
                            pred_action, _ = actor(batch["actor_observation"], training=False)
                            bc_loss = float(F.mse_loss(pred_action, batch["action"]).item())
                        epoch_bc_losses.append(bc_loss)
                        metric_lists["train_bc_loss"].append(bc_loss)
                if temperature_info:
                    epoch_temp_losses.append(record["temperature/loss"])
                    metric_lists["temperature_loss"].append(record["temperature/loss"])

                if args.max_train_steps is not None and update_step >= args.max_train_steps:
                    stop_training = True
                    break

            online_metrics = None
            if args.online_eval_interval > 0 and (
                epoch % args.online_eval_interval == 0 or stop_training or epoch == args.epochs
            ):
                eval_checkpoint = output_dir / f"epoch_{epoch:04d}"
                current_metrics = {
                    "dynamics_val_metrics": dynamics_val_metrics,
                    "dynamics_train_metrics": dynamics_train_metrics,
                    "rollout_metrics": rollout_metrics,
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
                        args=args,
                        metrics={"online_eval": online_metrics, "rollout_metrics": rollout_metrics},
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
                    args=args,
                    metrics={
                        "dynamics_val_metrics": dynamics_val_metrics,
                        "dynamics_train_metrics": dynamics_train_metrics,
                        "rollout_metrics": rollout_metrics,
                    },
                )

            row = {
                "epoch": epoch,
                "update": update_step,
                "train_actor_loss": float(np.nanmean(epoch_actor_losses)) if epoch_actor_losses else float("nan"),
                "train_critic_loss": float(np.mean(epoch_critic_losses)) if epoch_critic_losses else float("nan"),
                "train_bc_loss": float(np.mean(epoch_bc_losses)) if epoch_bc_losses else float("nan"),
                "temperature_loss": float(np.nanmean(epoch_temp_losses)) if epoch_temp_losses else float("nan"),
                "dynamics_train_loss": float(dynamics_train_metrics["dynamics_loss"]),
                "dynamics_val_loss": float(dynamics_val_metrics["next_obs_mse"]),
                "dynamics_reward_loss": float(dynamics_train_metrics["dynamics_reward_loss"]),
                "dynamics_done_loss": float(dynamics_train_metrics["dynamics_done_loss"]),
                "uncertainty_mean": float(rollout_metrics["uncertainty_mean"]),
                "synthetic_return_mean": float(rollout_metrics["synthetic_reward_mean"]),
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
                    args=args,
                    metrics={"dynamics_val_metrics": dynamics_val_metrics, "rollout_metrics": rollout_metrics},
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
        args=args,
        metrics={"dynamics_val_metrics": dynamics_val_metrics, "best_online": best_online},
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
        "normalize_obs": args.normalize_obs,
        "reward_relabel": relabel_info,
        "dynamics_train_metrics": dynamics_train_metrics,
        "dynamics_val_metrics": dynamics_val_metrics,
        "train_actor_loss": _stats(metric_lists["train_actor_loss"]),
        "train_critic_loss": _stats(metric_lists["train_critic_loss"]),
        "train_bc_loss": _stats(metric_lists["train_bc_loss"]),
        "temperature_loss": _stats(metric_lists["temperature_loss"]),
        "uncertainty_mean": _stats(metric_lists["uncertainty_mean"]),
        "synthetic_return_mean": _stats(metric_lists["synthetic_reward_mean"]),
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
    parser = argparse.ArgumentParser(description="Train a MuJoCo Playground G1 HRLG BC-MOPO baseline.")
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--updates-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
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
    parser.add_argument("--reward-loss-coef", type=float, default=1.0)
    parser.add_argument("--done-loss-coef", type=float, default=0.1)
    parser.add_argument("--rollout-horizon", type=int, default=3)
    parser.add_argument("--rollout-batch-size", type=int, default=4096)
    parser.add_argument("--model-rollout-ratio", type=float, default=1.0)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--mopo-penalty-coef", type=float, default=1.0)
    parser.add_argument("--done-threshold", type=float, default=0.5)
    parser.add_argument("--force-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--auto-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--bc-alpha", type=float, default=0.1)
    parser.add_argument("--actor-update-period", type=int, default=1)
    parser.add_argument("--critic-num-blocks", type=int, default=2)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-num-bins", type=int, default=101)
    parser.add_argument("--critic-min-v", type=float, default=-20.0)
    parser.add_argument("--critic-max-v", type=float, default=80.0)
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
    _run_training(args)


if __name__ == "__main__":
    main()
