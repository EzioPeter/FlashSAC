#!/usr/bin/env python3
"""Evaluate a MuJoCo Playground G1 HRLG behavior cloning checkpoint."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

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


DEFAULT_CHECKPOINT = REPO_ROOT / "models/mjp_g1_hrlg_bc/seed0_100traj/best"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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
    if (path / "best" / "actor.pt").exists():
        return path / "best"
    if (path / "final" / "actor.pt").exists():
        return path / "final"
    return max(candidates, key=_step_number)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _load_actor(checkpoint: Path, device: torch.device) -> tuple[FlashSACActor, np.ndarray, np.ndarray, bool]:
    ckpt = torch.load(checkpoint / "actor.pt", map_location=device, weights_only=False)
    actor = FlashSACActor(
        num_blocks=2,
        input_dim=hrlg.POLICY_OBS_SIZE,
        hidden_dim=128,
        action_dim=hrlg.ACTION_SIZE,
    ).to(device)
    state_dict = ckpt["network_state_dict"]
    if state_dict and next(iter(state_dict)).startswith("_orig_mod."):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    actor.load_state_dict(state_dict)
    actor.eval()

    stats_path = checkpoint / "obs_stats.npz"
    if stats_path.exists():
        with np.load(stats_path) as stats:
            obs_mean = stats["mean"].astype(np.float32)
            obs_std = stats["std"].astype(np.float32)
            normalize_obs = bool(np.asarray(stats["enabled"]).item())
    else:
        normalize_obs = bool(ckpt.get("obs_normalization", False))
        obs_mean = ckpt.get("obs_mean", torch.zeros(hrlg.POLICY_OBS_SIZE)).detach().cpu().numpy().astype(np.float32)
        obs_std = ckpt.get("obs_std", torch.ones(hrlg.POLICY_OBS_SIZE)).detach().cpu().numpy().astype(np.float32)
    obs_std = np.maximum(obs_std, 1e-6).astype(np.float32)
    return actor, obs_mean, obs_std, normalize_obs


@torch.no_grad()
def _policy(
    actor: FlashSACActor,
    obs: np.ndarray,
    *,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    normalize_obs: bool,
    device: torch.device,
) -> np.ndarray:
    if normalize_obs:
        obs = (obs - obs_mean) / obs_std
    obs_tensor = torch.as_tensor(np.array(obs, copy=True), dtype=torch.float32, device=device)
    mean, _ = actor.get_mean_and_std(obs_tensor, training=False)
    return torch.tanh(mean).detach().cpu().numpy().astype(np.float32)


def _make_env(noise_level: float) -> G1JoystickFlatTerrainHRLG:
    cfg = default_config()
    cfg.push_config.enable = False
    cfg.push_config.magnitude_range = [0.0, 0.0]
    cfg.noise_config.level = float(noise_level)
    return G1JoystickFlatTerrainHRLG(config=cfg)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    device = _resolve_device(args.device)
    actor, obs_mean, obs_std, normalize_obs = _load_actor(checkpoint, device)

    env = _make_env(args.noise_level)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    returns: list[float] = []
    lengths: list[int] = []
    survived: list[bool] = []
    done_count = 0
    rng = jax.random.PRNGKey(args.seed)
    start_time = time.time()

    episodes_remaining = args.num_eval_episodes
    while episodes_remaining > 0:
        batch_size = min(args.num_envs, episodes_remaining)
        rng, reset_rng = jax.random.split(rng)
        state = reset_fn(jax.random.split(reset_rng, batch_size))
        batch_returns = np.zeros(batch_size, dtype=np.float64)
        batch_lengths = np.zeros(batch_size, dtype=np.int32)
        batch_done = np.zeros(batch_size, dtype=bool)

        for _ in range(args.max_episode_steps):
            obs = np.asarray(state.obs["state"], dtype=np.float32)
            action = _policy(
                actor,
                obs,
                obs_mean=obs_mean,
                obs_std=obs_std,
                normalize_obs=normalize_obs,
                device=device,
            )
            state = step_fn(state, jp.asarray(action, dtype=jp.float32))
            reward = np.asarray(state.reward, dtype=np.float64)
            done = np.asarray(state.done, dtype=bool)
            active = ~batch_done
            batch_returns += reward * active
            batch_lengths += active.astype(np.int32)
            newly_done = active & done
            done_count += int(np.sum(newly_done))
            batch_done |= done
            if bool(np.all(batch_done)):
                break

        returns.extend(float(x) for x in batch_returns)
        lengths.extend(int(x) for x in batch_lengths)
        survived.extend(bool(x) for x in ~batch_done)
        episodes_remaining -= batch_size

    metrics = {
        "checkpoint": str(checkpoint),
        "num_eval_episodes": int(args.num_eval_episodes),
        "num_envs": int(args.num_envs),
        "max_episode_steps": int(args.max_episode_steps),
        "noise_level": float(args.noise_level),
        "normalize_obs": bool(normalize_obs),
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "survival_rate": float(np.mean(survived)),
        "survived_full": int(np.sum(survived)),
        "done_count": int(done_count),
        "episode_returns": returns,
        "episode_lengths": lengths,
        "episode_survived": survived,
        "wall_time_sec": float(time.time() - start_time),
    }
    if args.output is not None:
        _write_json(args.output.expanduser().resolve(), metrics)
    if not getattr(args, "quiet", False):
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MuJoCo Playground G1 HRLG BC actor.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
