#!/usr/bin/env python3
"""Relabel G1 HRLG buffers with observable reward and done signals.

The reward here intentionally uses only quantities that are plausible on a real
robot log: actor observation, action, and root pose/velocity.  Contact, torque,
force, and other simulator-only terms are reported as unused metadata.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (
    G1JoystickFlatTerrainHRLG,
    constants as hrlg,
    default_config,
)


DEFAULT_BUFFER = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000_observable_reward.npz"
)

SLICES = {
    "gyro": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 9),
    "joint_pos": slice(9, 38),
    "joint_vel": slice(38, 67),
    "last_act": slice(67, 96),
    "gait": slice(96, 98),
}

SELECTED_TERMS = (
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

UNUSED_SIM_TERMS = (
    "torques",
    "energy",
    "contact_force",
    "feet_air_time",
    "feet_slip",
    "feet_phase",
    "feet_height",
    "feet_clearance",
    "collision",
)


@dataclass(frozen=True)
class ObservableRewardConfig:
    dt: float
    tracking_sigma: float
    bad_orientation_limit_rad: float
    base_height_done_threshold: float
    use_projected_gravity_orientation: bool
    linear_velocity_frame: str
    angular_velocity_source: str


def _stats(array: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _quat_conj(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _rotate_world_to_body(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    zeros = np.zeros(vec.shape[:-1] + (1,), dtype=vec.dtype)
    vec_quat = np.concatenate([zeros, vec], axis=-1)
    return _quat_mul(_quat_mul(_quat_conj(quat_wxyz), vec_quat), quat_wxyz)[..., 1:]


def _make_env_metadata() -> tuple[dict[str, float], dict[str, Any]]:
    cfg = default_config()
    env = G1JoystickFlatTerrainHRLG(config=cfg)
    reward_scales = {name: float(value) for name, value in cfg.reward_config.scales.items()}
    env_info = {
        "dt": float(env.dt),
        "tracking_sigma": float(cfg.reward_config.tracking_sigma),
        "base_height_target": float(cfg.reward_config.base_height_target),
        "default_pose": np.asarray(env._joint_default_pose(), dtype=np.float32),
        "hip_indices": np.asarray(env._hip_indices, dtype=np.int64).reshape(-1),
        "knee_indices": np.asarray(env._knee_indices, dtype=np.int64).reshape(-1),
        "soft_lowers": np.asarray(env._soft_lowers, dtype=np.float32),
        "soft_uppers": np.asarray(env._soft_uppers, dtype=np.float32),
        "reward_scales": reward_scales,
    }
    return reward_scales, env_info


def infer_root_velocity_convention(
    state: np.ndarray,
    qpos_root: np.ndarray,
    qvel_root: np.ndarray,
    *,
    dt: float,
    sample_rows: int = 200_000,
) -> dict[str, Any]:
    obs = state[:, :-1].reshape(-1, state.shape[-1])
    qpos = qpos_root[:, :-1].reshape(-1, qpos_root.shape[-1])
    qvel = qvel_root[:, :-1].reshape(-1, qvel_root.shape[-1])
    qpos_next = qpos_root[:, 1:].reshape(-1, qpos_root.shape[-1])
    if obs.shape[0] > sample_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(obs.shape[0], size=sample_rows, replace=False)
        obs = obs[idx]
        qpos = qpos[idx]
        qvel = qvel[idx]
        qpos_next = qpos_next[idx]

    gyro = obs[:, SLICES["gyro"]] / hrlg.ANG_VEL_SCALE
    ang_raw = qvel[:, 3:6]
    ang_body = _rotate_world_to_body(qpos[:, 3:7], ang_raw)
    pos_diff = (qpos_next[:, :3] - qpos[:, :3]) / dt
    return {
        "linear_velocity_raw_vs_root_position_diff_rmse": float(np.sqrt(np.mean(np.square(qvel[:, :3] - pos_diff)))),
        "angular_velocity_raw_vs_obs_gyro_rmse": float(np.sqrt(np.mean(np.square(ang_raw - gyro)))),
        "angular_velocity_invquat_vs_obs_gyro_rmse": float(np.sqrt(np.mean(np.square(ang_body - gyro)))),
        "chosen_linear_velocity_frame": "world_to_base_via_root_quat",
        "chosen_angular_velocity_source": "obs_gyro_unscaled",
        "note": (
            "qvel_root[0:3] matches root position finite differences and is treated as world linear velocity; "
            "qvel_root[3:6] is close to unscaled policy gyro, but observable reward uses gyro directly."
        ),
    }


def compute_observable_reward(
    obs_t: np.ndarray,
    act_t: np.ndarray,
    obs_tp1: np.ndarray,
    qpos_root_tp1: np.ndarray,
    qvel_root_tp1: np.ndarray,
    *,
    reward_scales: dict[str, float],
    env_info: dict[str, Any],
    config: ObservableRewardConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    command = obs_t[..., SLICES["command"]] / np.asarray(hrlg.CMD_SCALE, dtype=np.float32)
    gyro = obs_tp1[..., SLICES["gyro"]] / hrlg.ANG_VEL_SCALE
    projected_gravity = obs_tp1[..., SLICES["projected_gravity"]]
    joint_pos_rel = obs_tp1[..., SLICES["joint_pos"]] / hrlg.DOF_POS_SCALE
    joint_vel = obs_tp1[..., SLICES["joint_vel"]] / hrlg.DOF_VEL_SCALE
    prev_joint_vel = obs_t[..., SLICES["joint_vel"]] / hrlg.DOF_VEL_SCALE
    last_act = obs_t[..., SLICES["last_act"]]

    quat = qpos_root_tp1[..., 3:7]
    root_lin_world = qvel_root_tp1[..., 0:3]
    base_lin_vel = _rotate_world_to_body(quat, root_lin_world)
    base_height = qpos_root_tp1[..., 2]

    default_pose = np.asarray(env_info["default_pose"], dtype=np.float32)
    hip_indices = np.asarray(env_info["hip_indices"], dtype=np.int64)
    knee_indices = np.asarray(env_info["knee_indices"], dtype=np.int64)
    soft_lowers = np.asarray(env_info["soft_lowers"], dtype=np.float32)
    soft_uppers = np.asarray(env_info["soft_uppers"], dtype=np.float32)
    joint_pos = joint_pos_rel + default_pose

    lin_vel_error = np.sum(np.square(command[..., :2] - base_lin_vel[..., :2]), axis=-1)
    ang_vel_error = np.square(command[..., 2] - gyro[..., 2])
    hip_error = joint_pos[..., hip_indices] - default_pose[hip_indices]
    hip_weight = np.where(
        command[..., 1:2] > 0.1,
        np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    out_of_limits = -np.clip(joint_pos - soft_lowers, None, 0.0)
    out_of_limits += np.clip(joint_pos - soft_uppers, 0.0, None)
    gravity_z = np.clip(-projected_gravity[..., 2], -1.0, 1.0)
    tilt = np.abs(np.arccos(gravity_z))
    bad_orientation = tilt > config.bad_orientation_limit_rad
    bad_height = (
        np.zeros_like(bad_orientation)
        if config.base_height_done_threshold <= 0
        else base_height < config.base_height_done_threshold
    )
    nonfinite = ~(
        np.all(np.isfinite(obs_t), axis=-1)
        & np.all(np.isfinite(obs_tp1), axis=-1)
        & np.all(np.isfinite(qpos_root_tp1), axis=-1)
        & np.all(np.isfinite(qvel_root_tp1), axis=-1)
        & np.all(np.isfinite(act_t), axis=-1)
    )
    done = (bad_orientation | bad_height | nonfinite).astype(np.float32)

    components = {
        "tracking_lin_vel": np.exp(-lin_vel_error / config.tracking_sigma).astype(np.float32),
        "tracking_ang_vel": np.exp(-ang_vel_error / config.tracking_sigma).astype(np.float32),
        "orientation": np.sum(np.square(projected_gravity[..., :2]), axis=-1).astype(np.float32),
        "ang_vel_xy": np.sum(np.square(gyro[..., :2]), axis=-1).astype(np.float32),
        "joint_deviation_hip": np.sum(np.abs(hip_error) * hip_weight, axis=-1).astype(np.float32),
        "joint_deviation_knee": np.sum(
            np.abs(joint_pos[..., knee_indices] - default_pose[knee_indices]),
            axis=-1,
        ).astype(np.float32),
        "pose": np.sum(np.square(joint_pos - default_pose), axis=-1).astype(np.float32),
        "dof_pos_limits": np.sum(out_of_limits, axis=-1).astype(np.float32),
        "stand_still": (
            np.sum(np.abs(joint_pos - default_pose), axis=-1) * (np.linalg.norm(command, axis=-1) < 0.01)
        ).astype(np.float32),
        "action_rate": np.sum(np.square(act_t - last_act), axis=-1).astype(np.float32),
        "base_height": np.square(base_height - float(env_info["base_height_target"])).astype(np.float32),
        "dof_acc": np.sum(np.square((joint_vel - prev_joint_vel) / config.dt), axis=-1).astype(np.float32),
        "termination": done.astype(np.float32),
    }

    reward_terms = {
        name: (float(reward_scales.get(name, 0.0)) * value * config.dt).astype(np.float32)
        for name, value in components.items()
    }
    reward = np.zeros_like(done, dtype=np.float32)
    for value in reward_terms.values():
        reward += value
    return reward.astype(np.float32), done.astype(np.float32), components, reward_terms


def relabel_buffer(
    input_path: Path,
    output_path: Path,
    *,
    bad_orientation_limit_rad: float,
    base_height_done_threshold: float,
    save_padded: bool,
) -> dict[str, Any]:
    reward_scales, env_info = _make_env_metadata()
    config = ObservableRewardConfig(
        dt=float(env_info["dt"]),
        tracking_sigma=float(env_info["tracking_sigma"]),
        bad_orientation_limit_rad=bad_orientation_limit_rad,
        base_height_done_threshold=base_height_done_threshold,
        use_projected_gravity_orientation=True,
        linear_velocity_frame="root_world_velocity_rotated_to_base_frame",
        angular_velocity_source="unscaled_policy_gyro",
    )
    with np.load(input_path) as data:
        state = data["state"].astype(np.float32, copy=False)
        action = data["action"].astype(np.float32, copy=False)
        qpos_root = data["qpos_root"].astype(np.float32, copy=False)
        qvel_root = data["qvel_root"].astype(np.float32, copy=False)

    obs_t = state[:, :-1]
    obs_tp1 = state[:, 1:]
    act_t = action[:, :-1]
    qpos_root_tp1 = qpos_root[:, 1:]
    qvel_root_tp1 = qvel_root[:, 1:]
    reward, done, components, reward_terms = compute_observable_reward(
        obs_t,
        act_t,
        obs_tp1,
        qpos_root_tp1,
        qvel_root_tp1,
        reward_scales=reward_scales,
        env_info=env_info,
        config=config,
    )
    velocity_convention = infer_root_velocity_convention(state, qpos_root, qvel_root, dt=config.dt)

    component_stats = {name: _stats(value) for name, value in components.items()}
    reward_term_stats = {name: _stats(value) for name, value in reward_terms.items()}
    metadata = {
        "source_path": str(input_path.resolve()),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "transition_definition": "reward[t] corresponds to state[t], action[t], state[t+1]",
        "observable_reward_config": asdict(config),
        "selected_terms": list(SELECTED_TERMS),
        "unused_sim_terms": list(UNUSED_SIM_TERMS),
        "reward_scales": {name: reward_scales.get(name, 0.0) for name in SELECTED_TERMS},
        "component_stats": component_stats,
        "reward_term_stats": reward_term_stats,
        "velocity_convention": velocity_convention,
        "reward_stats": _stats(reward),
        "done_count": int(np.sum(done > 0.5)),
        "done_rate": float(np.mean(done > 0.5)),
    }

    arrays: dict[str, Any] = {
        "state": state,
        "action": action,
        "qpos_root": qpos_root,
        "qvel_root": qvel_root,
        "reward": reward,
        "done": done,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for name in SELECTED_TERMS:
        arrays[f"component_{name}"] = components[name]
        arrays[f"reward_{name}"] = reward_terms[name]
    if save_padded:
        reward_padded = np.zeros(state.shape[:2], dtype=np.float32)
        done_padded = np.zeros(state.shape[:2], dtype=np.float32)
        reward_padded[:, :-1] = reward
        done_padded[:, :-1] = done
        arrays["reward_padded"] = reward_padded
        arrays["done_padded"] = done_padded
        metadata["padded_arrays"] = {
            "reward_padded": "shape (N,T), last column set to 0",
            "done_padded": "shape (N,T), last column set to 0",
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel a G1 HRLG buffer with observable reward/done.")
    parser.add_argument("--input", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bad-orientation-limit-rad", type=float, default=1.0)
    parser.add_argument("--base-height-done-threshold", type=float, default=0.0)
    parser.add_argument("--save-padded", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    metadata = relabel_buffer(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        bad_orientation_limit_rad=args.bad_orientation_limit_rad,
        base_height_done_threshold=args.base_height_done_threshold,
        save_padded=args.save_padded,
    )
    print(f"output={args.output.expanduser().resolve()}")
    print(f"reward_shape={metadata['reward_shape']} done_shape={metadata['done_shape']}")
    print(f"reward_stats={json.dumps(metadata['reward_stats'], sort_keys=True)}")
    print(f"done_count={metadata['done_count']} done_rate={metadata['done_rate']:.6g}")
    print("component_stats:")
    for name in SELECTED_TERMS:
        print(f"  {name}: {json.dumps(metadata['component_stats'][name], sort_keys=True)}")
    print("velocity_convention:")
    print(json.dumps(metadata["velocity_convention"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
