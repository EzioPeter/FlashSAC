#!/usr/bin/env python3
"""Play the trained G1 HRLG policy in the native MuJoCo viewer."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import torch

from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (
    G1JoystickFlatTerrainHRLG,
    constants as hrlg,
    default_config,
)
from mujoco_playground._src.collision import geoms_colliding


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "models/mjp_g1_hrlg/flat_50m_gpu2/G1JoystickFlatTerrainHRLG/seed0-0521-115545/step48829"
)


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
    obs_tensor = torch.as_tensor(np.array(obs[None], copy=True), dtype=torch.float32, device=device)
    mean, _ = actor.get_mean_and_std(obs_tensor, training=False)
    return torch.tanh(mean)[0].detach().cpu().numpy()


def _copy_state_to_mjdata(model: mujoco.MjModel, data: mujoco.MjData, state: object) -> None:
    mjx_data = state.data
    data.qpos[:] = np.asarray(mjx_data.qpos)
    data.qvel[:] = np.asarray(mjx_data.qvel)
    if hasattr(mjx_data, "ctrl") and data.ctrl.size:
        data.ctrl[:] = np.asarray(mjx_data.ctrl)
    if hasattr(mjx_data, "act") and data.act.size:
        data.act[:] = np.asarray(mjx_data.act)
    if hasattr(mjx_data, "mocap_pos") and data.mocap_pos.size:
        data.mocap_pos[:] = np.asarray(mjx_data.mocap_pos)
    if hasattr(mjx_data, "mocap_quat") and data.mocap_quat.size:
        data.mocap_quat[:] = np.asarray(mjx_data.mocap_quat)
    if hasattr(mjx_data, "xfrc_applied") and data.xfrc_applied.size:
        data.xfrc_applied[:] = np.asarray(mjx_data.xfrc_applied)
    mujoco.mj_forward(model, data)


def _make_set_command_fn(env: G1JoystickFlatTerrainHRLG):
    def set_command(state, command):
        info = dict(state.info)
        info["command"] = command
        contact = jp.array([
            geoms_colliding(state.data, geom_id, env._floor_geom_id)
            for geom_id in env._feet_geom_id
        ])
        obs = env._get_obs(state.data, info, contact)
        return state.replace(info=info, obs=obs)

    return jax.jit(set_command)


def _reset_state(reset_fn, set_command_fn, rng: jax.Array, command: jp.ndarray):
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)
    state = set_command_fn(state, command)
    return rng, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the viewer is closed.")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run this many policy steps without opening a viewer.")
    parser.add_argument("--reset-on-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    actor = _load_actor(checkpoint, device)

    cfg = default_config()
    cfg.push_config.enable = False
    cfg.push_config.magnitude_range = [0.0, 0.0]
    env = G1JoystickFlatTerrainHRLG(config=cfg)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    set_command_fn = _make_set_command_fn(env)

    command = jp.asarray(args.command, dtype=jp.float32)
    rng = jax.random.PRNGKey(args.seed)
    rng, state = _reset_state(reset_fn, set_command_fn, rng, command)

    model = env.mj_model
    data = mujoco.MjData(model)

    def step_policy_once():
        nonlocal rng, state
        obs = np.asarray(state.obs["state"], dtype=np.float32)
        action = _policy(actor, obs, device)
        state = step_fn(state, jp.asarray(action, dtype=jp.float32))
        state = set_command_fn(state, command)
        done = bool(np.asarray(state.done))
        if done and args.reset_on_done:
            rng, state = _reset_state(reset_fn, set_command_fn, rng, command)
        return done

    if args.headless_steps > 0:
        for _ in range(args.headless_steps):
            step_policy_once()
        _copy_state_to_mjdata(model, data, state)
        print(f"headless_steps={args.headless_steps}")
        print(f"qpos_xy={data.qpos[0]:.3f},{data.qpos[1]:.3f}")
        print(f"base_height={data.qpos[2]:.3f}")
        print(f"command={tuple(float(x) for x in args.command)}")
        return

    from mujoco import viewer as mujoco_viewer

    _copy_state_to_mjdata(model, data, state)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 120

        step_count = 0
        while viewer.is_running():
            loop_start = time.time()
            step_policy_once()
            step_count += 1

            _copy_state_to_mjdata(model, data, state)
            if args.follow_camera:
                viewer.cam.lookat[:] = (data.qpos[0], data.qpos[1], 0.8)
            viewer.sync()

            if args.max_steps > 0 and step_count >= args.max_steps:
                break
            if args.realtime:
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, env.dt - elapsed))


if __name__ == "__main__":
    main()
