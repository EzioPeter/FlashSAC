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


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--enable-push", action="store_true")
    parser.add_argument(
        "--attempt-batch-size",
        type=int,
        default=0,
        help="0 means collect all remaining trajectories at once.",
    )
    parser.add_argument(
        "--discard-done-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discard and resample trajectories that terminate before trajectory-length.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Maximum rollout attempts when discarding terminated trajectories.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    actor = _load_actor(checkpoint, device)

    cfg = default_config()
    if not args.enable_push:
        cfg.push_config.enable = False
        cfg.push_config.magnitude_range = [0.0, 0.0]
    env = G1JoystickFlatTerrainHRLG(config=cfg)

    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    states = np.empty((args.num_trajectories, args.trajectory_length, hrlg.POLICY_OBS_SIZE), dtype=np.float32)
    actions = np.empty((args.num_trajectories, args.trajectory_length, hrlg.ACTION_SIZE), dtype=np.float32)
    qpos_root = np.empty((args.num_trajectories, args.trajectory_length, 7), dtype=np.float32)
    qvel_root = np.empty((args.num_trajectories, args.trajectory_length, 6), dtype=np.float32)

    start_time = time.time()
    rng = jax.random.PRNGKey(args.seed)
    accepted = 0
    attempts = 0
    discarded_done = 0
    total_done_steps = 0
    while accepted < args.num_trajectories and attempts < args.max_attempts:
        batch_size = args.num_trajectories - accepted
        if args.attempt_batch_size > 0:
            batch_size = min(batch_size, args.attempt_batch_size)
        attempts += batch_size
        rng, reset_rng = jax.random.split(rng)
        state = reset_fn(jax.random.split(reset_rng, batch_size))
        batch_done = np.zeros(batch_size, dtype=bool)
        batch_states = np.empty((batch_size, args.trajectory_length, hrlg.POLICY_OBS_SIZE), dtype=np.float32)
        batch_actions = np.empty((batch_size, args.trajectory_length, hrlg.ACTION_SIZE), dtype=np.float32)
        batch_qpos_root = np.empty((batch_size, args.trajectory_length, 7), dtype=np.float32)
        batch_qvel_root = np.empty((batch_size, args.trajectory_length, 6), dtype=np.float32)

        for step in range(args.trajectory_length):
            obs = np.asarray(state.obs["state"], dtype=np.float32)
            action = _policy(actor, obs, device)

            batch_states[:, step] = obs
            batch_actions[:, step] = action
            batch_qpos_root[:, step] = np.asarray(state.data.qpos[:, :7], dtype=np.float32)
            batch_qvel_root[:, step] = np.asarray(state.data.qvel[:, :6], dtype=np.float32)

            state = step_fn(state, jp.asarray(action, dtype=jp.float32))
            done = np.asarray(state.done, dtype=bool)
            batch_done |= done
            total_done_steps += int(done.sum())

        keep = ~batch_done if args.discard_done_trajectories else np.ones(batch_size, dtype=bool)
        num_keep = min(int(keep.sum()), args.num_trajectories - accepted)
        if num_keep:
            keep_indices = np.nonzero(keep)[0][:num_keep]
            end = accepted + num_keep
            states[accepted:end] = batch_states[keep_indices]
            actions[accepted:end] = batch_actions[keep_indices]
            qpos_root[accepted:end] = batch_qpos_root[keep_indices]
            qvel_root[accepted:end] = batch_qvel_root[keep_indices]
            accepted = end
        discarded_done += int((~keep).sum())

        if args.progress_interval > 0 and (attempts % args.progress_interval == 0 or accepted == args.num_trajectories):
            elapsed = time.time() - start_time
            print(
                f"accepted={accepted}/{args.num_trajectories} attempts={attempts} "
                f"discarded_done={discarded_done} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if accepted < args.num_trajectories:
        raise RuntimeError(
            f"Collected only {accepted}/{args.num_trajectories} trajectories after "
            f"{attempts} attempts. Increase --max-attempts or use --no-discard-done-trajectories."
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
    print(f"accepted_trajectories={accepted}")
    print(f"attempts={attempts}")
    print(f"discarded_done_trajectories={discarded_done}")
    print(f"total_done_steps={total_done_steps}")
    print(f"elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
