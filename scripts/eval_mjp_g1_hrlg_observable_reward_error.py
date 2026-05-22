#!/usr/bin/env python3
"""Compare observable reward relabeling against MuJoCo Playground true reward."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
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

from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (
    G1JoystickFlatTerrainHRLG,
    default_config,
)
from scripts.eval_mjp_g1_hrlg_bc import (
    _load_actor,
    _policy,
    _resolve_checkpoint,
    _resolve_device,
)
from scripts.relabel_mjp_g1_hrlg_observable_reward import (
    SELECTED_TERMS,
    ObservableRewardConfig,
    _make_env_metadata,
    _stats,
    compute_observable_reward,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "models/mjp_g1_hrlg/flat_50m_gpu2/G1JoystickFlatTerrainHRLG/seed0-0521-154319/step48829"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/observable_reward_eval_expert_step48829.json"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_component_csv(path: Path, component_stats: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "mean", "std", "min", "max"])
        writer.writeheader()
        for name, stats in component_stats.items():
            writer.writerow({"component": name, **stats})


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
    reward_scales, env_info = _make_env_metadata()
    config = ObservableRewardConfig(
        dt=float(env_info["dt"]),
        tracking_sigma=float(env_info["tracking_sigma"]),
        bad_orientation_limit_rad=float(args.bad_orientation_limit_rad),
        base_height_done_threshold=float(args.base_height_done_threshold),
        use_projected_gravity_orientation=True,
        linear_velocity_frame="root_world_velocity_rotated_to_base_frame",
        angular_velocity_source="unscaled_policy_gyro",
    )

    rng = jax.random.PRNGKey(args.seed)
    episodes_remaining = int(args.num_eval_episodes)
    start_time = time.time()

    true_rewards: list[np.ndarray] = []
    observable_rewards: list[np.ndarray] = []
    reward_errors: list[np.ndarray] = []
    true_dones: list[np.ndarray] = []
    observable_dones: list[np.ndarray] = []
    component_values: dict[str, list[np.ndarray]] = {name: [] for name in SELECTED_TERMS}
    reward_term_values: dict[str, list[np.ndarray]] = {name: [] for name in SELECTED_TERMS}
    episode_true_returns: list[float] = []
    episode_observable_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_survived: list[bool] = []

    while episodes_remaining > 0:
        batch_size = min(int(args.num_envs), episodes_remaining)
        rng, reset_rng = jax.random.split(rng)
        state = reset_fn(jax.random.split(reset_rng, batch_size))

        batch_done = np.zeros(batch_size, dtype=bool)
        batch_true_returns = np.zeros(batch_size, dtype=np.float64)
        batch_observable_returns = np.zeros(batch_size, dtype=np.float64)
        batch_lengths = np.zeros(batch_size, dtype=np.int32)

        for _ in range(int(args.max_episode_steps)):
            obs_t = np.asarray(state.obs["state"], dtype=np.float32)
            action = _policy(
                actor,
                obs_t,
                obs_mean=obs_mean,
                obs_std=obs_std,
                normalize_obs=normalize_obs,
                device=device,
            )
            state = step_fn(state, jp.asarray(action, dtype=jp.float32))
            obs_tp1 = np.asarray(state.obs["state"], dtype=np.float32)
            qpos_root_tp1 = np.asarray(state.data.qpos[:, :7], dtype=np.float32)
            qvel_root_tp1 = np.asarray(state.data.qvel[:, :6], dtype=np.float32)
            true_reward = np.asarray(state.reward, dtype=np.float32)
            true_done = np.asarray(state.done, dtype=bool)

            obs_reward, obs_done, components, reward_terms = compute_observable_reward(
                obs_t,
                action,
                obs_tp1,
                qpos_root_tp1,
                qvel_root_tp1,
                reward_scales=reward_scales,
                env_info=env_info,
                config=config,
            )

            active = ~batch_done
            if np.any(active):
                true_rewards.append(true_reward[active].astype(np.float32))
                observable_rewards.append(obs_reward[active].astype(np.float32))
                reward_errors.append((obs_reward[active] - true_reward[active]).astype(np.float32))
                true_dones.append(true_done[active].astype(np.float32))
                observable_dones.append(obs_done[active].astype(np.float32))
                for name in SELECTED_TERMS:
                    component_values[name].append(components[name][active].astype(np.float32))
                    reward_term_values[name].append(reward_terms[name][active].astype(np.float32))

            batch_true_returns += true_reward * active
            batch_observable_returns += obs_reward * active
            batch_lengths += active.astype(np.int32)
            batch_done |= true_done
            if bool(np.all(batch_done)):
                break

        episode_true_returns.extend(float(x) for x in batch_true_returns)
        episode_observable_returns.extend(float(x) for x in batch_observable_returns)
        episode_lengths.extend(int(x) for x in batch_lengths)
        episode_survived.extend(bool(x) for x in ~batch_done)
        episodes_remaining -= batch_size

    true_reward_arr = np.concatenate(true_rewards) if true_rewards else np.asarray([], dtype=np.float32)
    obs_reward_arr = np.concatenate(observable_rewards) if observable_rewards else np.asarray([], dtype=np.float32)
    reward_error_arr = np.concatenate(reward_errors) if reward_errors else np.asarray([], dtype=np.float32)
    true_done_arr = np.concatenate(true_dones) if true_dones else np.asarray([], dtype=np.float32)
    obs_done_arr = np.concatenate(observable_dones) if observable_dones else np.asarray([], dtype=np.float32)
    done_match = obs_done_arr == true_done_arr
    false_positive = (obs_done_arr > 0.5) & (true_done_arr < 0.5)
    false_negative = (obs_done_arr < 0.5) & (true_done_arr > 0.5)

    component_stats = {
        name: _stats(np.concatenate(values)) if values else {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        for name, values in component_values.items()
    }
    reward_term_stats = {
        name: _stats(np.concatenate(values)) if values else {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        for name, values in reward_term_values.items()
    }

    metrics = {
        "checkpoint": str(checkpoint),
        "num_eval_episodes": int(args.num_eval_episodes),
        "num_envs": int(args.num_envs),
        "max_episode_steps": int(args.max_episode_steps),
        "noise_level": float(args.noise_level),
        "normalize_obs": bool(normalize_obs),
        "observable_reward_config": asdict(config),
        "reward_scales": {name: float(reward_scales.get(name, 0.0)) for name in SELECTED_TERMS},
        "num_transitions": int(true_reward_arr.size),
        "true_reward_stats": _stats(true_reward_arr),
        "observable_reward_stats": _stats(obs_reward_arr),
        "reward_error_stats": _stats(reward_error_arr),
        "reward_mae": float(np.mean(np.abs(reward_error_arr))),
        "reward_rmse": float(np.sqrt(np.mean(np.square(reward_error_arr)))),
        "reward_bias": float(np.mean(reward_error_arr)),
        "reward_corr": _safe_corr(true_reward_arr, obs_reward_arr),
        "done_accuracy": float(np.mean(done_match)),
        "false_positive_done_rate": float(np.mean(false_positive)),
        "false_negative_done_rate": float(np.mean(false_negative)),
        "true_done_count": int(np.sum(true_done_arr > 0.5)),
        "observable_done_count": int(np.sum(obs_done_arr > 0.5)),
        "component_stats": component_stats,
        "reward_term_stats": reward_term_stats,
        "avg_true_return": float(np.mean(episode_true_returns)),
        "avg_observable_return": float(np.mean(episode_observable_returns)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "survived_full": int(np.sum(episode_survived)),
        "episode_true_returns": episode_true_returns,
        "episode_observable_returns": episode_observable_returns,
        "episode_lengths": episode_lengths,
        "episode_survived": episode_survived,
        "wall_time_sec": float(time.time() - start_time),
        "notes": [
            "Observable reward intentionally omits simulator-only terms such as feet/contact/torque rewards.",
            "orientation and ang_vel_xy are approximated from policy observation/root data, not torso sensors.",
            "Default noise_level=0 isolates reward reconstruction error from observation noise.",
        ],
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        _write_json(output, metrics)
        csv_output = (
            args.csv_output.expanduser().resolve()
            if args.csv_output
            else output.with_suffix(".components.csv")
        )
        _write_component_csv(csv_output, component_stats)
        metrics["component_csv"] = str(csv_output)
        _write_json(output, metrics)
    if not args.quiet:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare observable reward to MuJoCo Playground true reward.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--bad-orientation-limit-rad", type=float, default=1.0)
    parser.add_argument("--base-height-done-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
