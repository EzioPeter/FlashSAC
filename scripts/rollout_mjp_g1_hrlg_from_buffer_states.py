#!/usr/bin/env python3
"""Roll out the G1 HRLG policy from states reconstructed from a saved buffer."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import os
import re
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
import torch
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (
    G1JoystickFlatTerrainHRLG,
    constants as hrlg,
    default_config,
)


DEFAULT_BUFFER = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "models/mjp_g1_hrlg/flat_50m_gpu2/G1JoystickFlatTerrainHRLG/seed0-0521-154319"
)

SLICES = {
    "command": slice(6, 9),
    "joint_pos": slice(9, 38),
    "joint_vel": slice(38, 67),
    "last_act": slice(67, 96),
}


def _step_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "actor.pt").exists():
        return path
    candidates = [child for child in path.iterdir() if child.is_dir() and (child / "actor.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"Could not find actor.pt under checkpoint path: {path}")
    return max(candidates, key=_step_number)


def _load_actor(checkpoint: Path, device: torch.device) -> FlashSACActor:
    actor = FlashSACActor(
        num_blocks=2,
        input_dim=hrlg.POLICY_OBS_SIZE,
        hidden_dim=128,
        action_dim=hrlg.ACTION_SIZE,
    ).to(device)
    ckpt = torch.load(checkpoint / "actor.pt", map_location=device)
    state_dict = ckpt["network_state_dict"]
    if state_dict and next(iter(state_dict)).startswith("_orig_mod."):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


@torch.no_grad()
def _policy(actor: FlashSACActor, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.as_tensor(np.array(obs, copy=True), dtype=torch.float32, device=device)
    mean, _ = actor.get_mean_and_std(obs_tensor, training=False)
    return torch.tanh(mean).detach().cpu().numpy().astype(np.float32)


def _select_states(
    states: np.ndarray,
    qpos_root: np.ndarray,
    num_states: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    heights = qpos_root[:, :, 2]
    valid = np.isfinite(states).all(axis=-1) & np.isfinite(qpos_root).all(axis=-1)
    valid &= heights > 0.55
    valid &= heights < 1.05
    valid[:, 500] = False
    episodes, steps = np.nonzero(valid)
    if len(steps) < num_states:
        raise ValueError(f"Only found {len(steps)} valid states, need {num_states}.")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(steps), size=num_states, replace=False)
    order = np.lexsort((steps[selected], episodes[selected]))
    return episodes[selected][order], steps[selected][order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-states", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    buffer_path = args.buffer.expanduser().resolve()
    checkpoint = _resolve_checkpoint(args.checkpoint)
    with np.load(buffer_path) as data:
        states = data["state"]
        qpos_root = data["qpos_root"]
        qvel_root = data["qvel_root"]

    episodes, steps = _select_states(states, qpos_root, args.num_states, args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and os.environ.get("JAX_PLATFORMS") != "cpu" else "cpu")
    actor = _load_actor(checkpoint, device)

    cfg = default_config()
    cfg.push_config.enable = False
    cfg.push_config.magnitude_range = [0.0, 0.0]
    env = G1JoystickFlatTerrainHRLG(config=cfg)
    joint_default_pose = np.asarray(env._joint_default_pose(), dtype=np.float32)
    cmd_scale = np.asarray(hrlg.CMD_SCALE, dtype=np.float32)

    def init_data(qpos, qvel, ctrl):
        return mjx_env.init(env.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

    def refresh_obs(state):
        def refresh_one(single_state):
            info = dict(single_state.info)
            contact = jp.array([
                geoms_colliding(single_state.data, geom_id, env._floor_geom_id)
                for geom_id in env._feet_geom_id
            ])
            obs = env._get_obs(single_state.data, info, contact)
            return single_state.replace(info=info, obs=obs)

        return jax.vmap(refresh_one)(state)

    init_data_fn = jax.jit(jax.vmap(init_data))
    reset_fn = jax.jit(jax.vmap(env.reset))
    refresh_obs_fn = jax.jit(refresh_obs)
    step_fn = jax.jit(jax.vmap(env.step))

    obs0 = states[episodes, steps]
    qpos = np.zeros((args.num_states, env.mjx_model.nq), dtype=np.float32)
    qvel = np.zeros((args.num_states, env.mjx_model.nv), dtype=np.float32)
    qpos[:, :7] = qpos_root[episodes, steps]
    qpos[:, 7:] = obs0[:, SLICES["joint_pos"]] / hrlg.DOF_POS_SCALE + joint_default_pose
    qvel[:, :6] = qvel_root[episodes, steps]
    qvel[:, 6:] = obs0[:, SLICES["joint_vel"]] / hrlg.DOF_VEL_SCALE

    command = obs0[:, SLICES["command"]] / cmd_scale
    last_act = obs0[:, SLICES["last_act"]]
    ctrl = joint_default_pose[None, :] + last_act * hrlg.ACTION_SCALE

    template = reset_fn(jax.random.split(jax.random.PRNGKey(args.seed), args.num_states))
    data = init_data_fn(jp.asarray(qpos), jp.asarray(qvel), jp.asarray(ctrl))
    info = dict(template.info)
    info["command"] = jp.asarray(command, dtype=jp.float32)
    info["last_act"] = jp.asarray(last_act, dtype=jp.float32)
    info["last_last_act"] = jp.zeros_like(info["last_act"])
    info["motor_targets"] = jp.asarray(ctrl, dtype=jp.float32)
    info["joint_default_pose"] = jp.asarray(np.broadcast_to(joint_default_pose, (args.num_states, hrlg.ACTION_SIZE)))
    info["policy_step"] = jp.asarray(steps, dtype=jp.int32)
    info["step"] = jp.asarray(steps % 501, dtype=jp.int32)
    state = template.replace(data=data, info=info)
    state = refresh_obs_fn(state)

    first_done_step = np.full(args.num_states, -1, dtype=np.int32)
    min_height = np.asarray(state.data.qpos[:, 2], dtype=np.float32)
    start_time = time.time()

    for step in range(args.rollout_length):
        obs = np.asarray(state.obs["state"], dtype=np.float32)
        action = _policy(actor, obs, device)
        state = step_fn(state, jp.asarray(action, dtype=jp.float32))
        done = np.asarray(state.done).astype(bool)
        newly_done = (first_done_step < 0) & done
        first_done_step[newly_done] = step + 1
        min_height = np.minimum(min_height, np.asarray(state.data.qpos[:, 2], dtype=np.float32))

        if args.progress_interval > 0 and (step + 1) % args.progress_interval == 0:
            survived = int(np.sum(first_done_step < 0))
            elapsed = time.time() - start_time
            print(
                f"rollout_step={step + 1}/{args.rollout_length} survived={survived}/{args.num_states} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    survival_steps = np.where(first_done_step < 0, args.rollout_length, first_done_step)
    print(f"buffer={buffer_path}")
    print(f"checkpoint={checkpoint}")
    print(f"rollout_length={args.rollout_length}")
    print(f"num_states={args.num_states}")
    print(f"survived_full={int(np.sum(first_done_step < 0))}/{args.num_states}")
    print("selected_states:")
    for idx, (episode, start_step, survived_steps, first_done, height) in enumerate(
        zip(episodes, steps, survival_steps, first_done_step, min_height)
    ):
        done_text = "full" if first_done < 0 else f"done@{int(first_done)}"
        print(
            f"  {idx:02d}: episode={int(episode):03d} step={int(start_step):03d} "
            f"survival_steps={int(survived_steps):04d} {done_text} min_height={float(height):.3f}"
        )
    print(f"elapsed_s={time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
