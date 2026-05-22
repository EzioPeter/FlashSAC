#!/usr/bin/env python3
"""Check one-step consistency after reconstructing G1 HRLG states from a buffer."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import mjx_env

from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (
    G1JoystickFlatTerrainHRLG,
    constants as hrlg,
    default_config,
)


DEFAULT_BUFFER = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"

SLICES = {
    "gyro": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 9),
    "joint_pos": slice(9, 38),
    "joint_vel": slice(38, 67),
    "last_act": slice(67, 96),
    "gait": slice(96, 98),
}


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def _max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def _summarize(name: str, diff: np.ndarray) -> str:
    return f"{name}: mae={_mae(diff):.6g} rmse={_rmse(diff):.6g} max={_max_abs(diff):.6g}"


def _choose_indices(
    num_trajectories: int,
    trajectory_length: int,
    num_samples: int,
    seed: int,
    include_command_boundary: bool,
) -> tuple[np.ndarray, np.ndarray]:
    episode, step = np.meshgrid(
        np.arange(num_trajectories, dtype=np.int32),
        np.arange(trajectory_length - 1, dtype=np.int32),
        indexing="ij",
    )
    episode = episode.reshape(-1)
    step = step.reshape(-1)
    if not include_command_boundary:
        keep = step % 501 != 500
        episode = episode[keep]
        step = step[keep]
    rng = np.random.default_rng(seed)
    if num_samples > 0 and num_samples < len(step):
        selected = rng.choice(len(step), size=num_samples, replace=False)
        episode = episode[selected]
        step = step[selected]
    return episode, step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--num-samples", type=int, default=8192, help="Use <=0 to check every transition.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-command-boundary", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()

    buffer_path = args.buffer.expanduser().resolve()
    with np.load(buffer_path) as data:
        states = data["state"]
        actions = data["action"]
        qpos_root = data["qpos_root"]
        qvel_root = data["qvel_root"]

    num_trajectories, trajectory_length, obs_dim = states.shape
    if obs_dim != hrlg.POLICY_OBS_SIZE:
        raise ValueError(f"Expected obs dim {hrlg.POLICY_OBS_SIZE}, got {obs_dim}.")

    episodes, steps = _choose_indices(
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        num_samples=args.num_samples,
        seed=args.seed,
        include_command_boundary=args.include_command_boundary,
    )

    cfg = default_config()
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.push_config.magnitude_range = [0.0, 0.0]
    env = G1JoystickFlatTerrainHRLG(config=cfg)
    joint_default_pose = np.asarray(env._joint_default_pose(), dtype=np.float32)
    cmd_scale = np.asarray(hrlg.CMD_SCALE, dtype=np.float32)

    def init_data(qpos, qvel, ctrl):
        return mjx_env.init(env.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

    init_data_fn = jax.jit(jax.vmap(init_data))
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    obs_diffs = []
    qpos_root_diffs = []
    qvel_root_diffs = []
    done_count = 0
    start_time = time.time()

    num_batches = int(np.ceil(len(steps) / args.batch_size))
    for batch_idx, start in enumerate(range(0, len(steps), args.batch_size), start=1):
        end = min(start + args.batch_size, len(steps))
        ep = episodes[start:end]
        st = steps[start:end]
        batch = end - start

        obs_t = states[ep, st]
        action_t = actions[ep, st]
        target_obs_tp1 = states[ep, st + 1]
        target_qpos_root_tp1 = qpos_root[ep, st + 1]
        target_qvel_root_tp1 = qvel_root[ep, st + 1]

        qpos = np.zeros((batch, env.mjx_model.nq), dtype=np.float32)
        qvel = np.zeros((batch, env.mjx_model.nv), dtype=np.float32)
        qpos[:, :7] = qpos_root[ep, st]
        qpos[:, 7:] = obs_t[:, SLICES["joint_pos"]] / hrlg.DOF_POS_SCALE + joint_default_pose
        qvel[:, :6] = qvel_root[ep, st]
        qvel[:, 6:] = obs_t[:, SLICES["joint_vel"]] / hrlg.DOF_VEL_SCALE

        command = obs_t[:, SLICES["command"]] / cmd_scale
        last_act = obs_t[:, SLICES["last_act"]]
        ctrl = joint_default_pose[None, :] + last_act * hrlg.ACTION_SCALE

        template = reset_fn(jax.random.split(jax.random.PRNGKey(args.seed + batch_idx), batch))
        data = init_data_fn(jp.asarray(qpos), jp.asarray(qvel), jp.asarray(ctrl))
        info = dict(template.info)
        info["command"] = jp.asarray(command, dtype=jp.float32)
        info["last_act"] = jp.asarray(last_act, dtype=jp.float32)
        info["last_last_act"] = jp.zeros_like(info["last_act"])
        info["motor_targets"] = jp.asarray(ctrl, dtype=jp.float32)
        info["joint_default_pose"] = jp.asarray(np.broadcast_to(joint_default_pose, (batch, hrlg.ACTION_SIZE)))
        info["policy_step"] = jp.asarray(st, dtype=jp.int32)
        info["step"] = jp.asarray(st % 501, dtype=jp.int32)
        reconstructed = template.replace(data=data, info=info)

        pred = step_fn(reconstructed, jp.asarray(action_t, dtype=jp.float32))

        obs_diffs.append(np.asarray(pred.obs["state"], dtype=np.float32) - target_obs_tp1)
        qpos_root_diffs.append(np.asarray(pred.data.qpos[:, :7], dtype=np.float32) - target_qpos_root_tp1)
        qvel_root_diffs.append(np.asarray(pred.data.qvel[:, :6], dtype=np.float32) - target_qvel_root_tp1)
        done_count += int(np.asarray(pred.done).sum())

        if args.progress_interval > 0 and batch_idx % args.progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"checked_batches={batch_idx}/{num_batches} samples={end}/{len(steps)} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    obs_diff = np.concatenate(obs_diffs, axis=0)
    qpos_root_diff = np.concatenate(qpos_root_diffs, axis=0)
    qvel_root_diff = np.concatenate(qvel_root_diffs, axis=0)

    print(f"buffer={buffer_path}")
    print(f"samples={len(steps)}")
    print(f"batch_size={args.batch_size}")
    print("noise_in_replay=0.0")
    print(f"excluded_command_boundary={not args.include_command_boundary}")
    print(f"done_count={done_count}")
    print(_summarize("obs_state_all", obs_diff))
    for name, slc in SLICES.items():
        print(_summarize(f"obs_state/{name}", obs_diff[:, slc]))
    print(_summarize("qpos_root", qpos_root_diff))
    print(_summarize("qvel_root", qvel_root_diff))
    print(f"elapsed_s={time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
