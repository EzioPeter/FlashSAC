#!/usr/bin/env python3
"""Collect G1 HRLG rollout buffers from a trained MuJoCo Playground policy."""

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

from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (
    G1JoystickFlatTerrainHRLG,
    constants as hrlg,
    default_config,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "models/mjp_g1_hrlg/flat_50m_gpu2/G1JoystickFlatTerrainHRLG/seed0-0521-154319"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-push", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() and os.environ.get("JAX_PLATFORMS") != "cpu" else "cpu")
    actor = _load_actor(checkpoint, device)

    cfg = default_config()
    if not args.enable_push:
        cfg.push_config.enable = False
        cfg.push_config.magnitude_range = [0.0, 0.0]
    env = G1JoystickFlatTerrainHRLG(config=cfg)

    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    rng = jax.random.PRNGKey(args.seed)
    reset_rngs = jax.random.split(rng, args.num_trajectories)
    state = reset_fn(reset_rngs)

    states = np.empty((args.num_trajectories, args.trajectory_length, hrlg.POLICY_OBS_SIZE), dtype=np.float32)
    actions = np.empty((args.num_trajectories, args.trajectory_length, hrlg.ACTION_SIZE), dtype=np.float32)
    qpos_root = np.empty((args.num_trajectories, args.trajectory_length, 7), dtype=np.float32)
    qvel_root = np.empty((args.num_trajectories, args.trajectory_length, 6), dtype=np.float32)

    start_time = time.time()
    done_count = 0
    for step in range(args.trajectory_length):
        obs = np.asarray(state.obs["state"], dtype=np.float32)
        action = _policy(actor, obs, device)

        states[:, step] = obs
        actions[:, step] = action
        qpos_root[:, step] = np.asarray(state.data.qpos[:, :7], dtype=np.float32)
        qvel_root[:, step] = np.asarray(state.data.qvel[:, :6], dtype=np.float32)

        state = step_fn(state, jp.asarray(action, dtype=jp.float32))
        done_count += int(np.asarray(state.done).sum())

        if args.progress_interval > 0 and (step + 1) % args.progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"collected_steps={step + 1}/{args.trajectory_length} "
                f"elapsed_s={elapsed:.1f} done_count={done_count}",
                flush=True,
            )

    np.savez(
        output,
        state=states,
        action=actions,
        qpos_root=qpos_root,
        qvel_root=qvel_root,
    )

    elapsed = time.time() - start_time
    print(f"checkpoint={checkpoint}")
    print(f"output={output}")
    print(f"state_shape={states.shape}")
    print(f"action_shape={actions.shape}")
    print(f"qpos_root_shape={qpos_root.shape}")
    print(f"qvel_root_shape={qvel_root.shape}")
    print(f"done_count={done_count}")
    print(f"elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
