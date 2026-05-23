#!/usr/bin/env python3
"""Evaluate root-aware G1 HRLG world models on held-out trajectory groups."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_mjp_g1_hrlg_bc_mopo import (
    DynamicsEnsemble,
    TransitionArrays,
    _evaluate_dynamics_arrays,
    _load_dynamics_checkpoint,
    _load_obs_stats_from_dynamics_checkpoint,
    _make_observable_reward_context,
    _normalize,
)


def _build_external_arrays(dataset: Path, test_trajectories: list[int]) -> TransitionArrays:
    with np.load(dataset) as data:
        state = data["state"].astype(np.float32, copy=False)
        action = data["action"].astype(np.float32, copy=False)
        qpos_root = data["qpos_root"].astype(np.float32, copy=False)
        qvel_root = data["qvel_root"].astype(np.float32, copy=False)
        reward = data["reward"].astype(np.float32, copy=False)
        done = data["done"].astype(np.float32, copy=False)
    traj = np.asarray(test_trajectories, dtype=np.int32)
    steps = np.broadcast_to(np.arange(state.shape[1] - 1, dtype=np.int32), (len(traj), state.shape[1] - 1))
    traj_ids = np.broadcast_to(traj[:, None], (len(traj), state.shape[1] - 1))
    return TransitionArrays(
        observation=state[traj, :-1].reshape(-1, state.shape[-1]).copy(),
        action=action[traj, :-1].reshape(-1, action.shape[-1]).copy(),
        next_observation=state[traj, 1:].reshape(-1, state.shape[-1]).copy(),
        reward=reward[traj].reshape(-1).copy(),
        terminated=done[traj].reshape(-1).copy(),
        truncated=np.zeros(len(traj) * (state.shape[1] - 1), dtype=np.float32),
        trajectory=traj_ids.reshape(-1).copy(),
        step=steps.reshape(-1).copy(),
        qpos_root=qpos_root[traj, :-1].reshape(-1, qpos_root.shape[-1]).copy(),
        qvel_root=qvel_root[traj, :-1].reshape(-1, qvel_root.shape[-1]).copy(),
        next_qpos_root=qpos_root[traj, 1:].reshape(-1, qpos_root.shape[-1]).copy(),
        next_qvel_root=qvel_root[traj, 1:].reshape(-1, qvel_root.shape[-1]).copy(),
    )


def _naive_next_equals_current(arrays: TransitionArrays) -> dict[str, float | dict[str, float]]:
    obs_error = arrays.observation - arrays.next_observation
    result: dict[str, float | dict[str, float]] = {
        "next_obs_mse": float(np.mean(np.square(obs_error))),
        "next_obs_mae": float(np.mean(np.abs(obs_error))),
    }
    if arrays.qpos_root is not None and arrays.next_qpos_root is not None:
        qpos_error = arrays.qpos_root - arrays.next_qpos_root
        result["qpos_root_next_mse"] = float(np.mean(np.square(qpos_error)))
        result["qpos_xyz_mse"] = float(np.mean(np.square(qpos_error[:, :3])))
        result["qpos_quat_mse"] = float(np.mean(np.square(qpos_error[:, 3:7])))
    if arrays.qvel_root is not None and arrays.next_qvel_root is not None:
        qvel_error = arrays.qvel_root - arrays.next_qvel_root
        result["qvel_root_next_mse"] = float(np.mean(np.square(qvel_error)))
    return result


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    cfg = json.load(open(checkpoint / "dynamics_config.json", encoding="utf-8"))
    obs_mean, obs_std, normalize_obs = _load_obs_stats_from_dynamics_checkpoint(checkpoint)

    dynamics = DynamicsEnsemble(
        ensemble_size=int(cfg["dynamics_ensemble_size"]),
        obs_dim=int(cfg["obs_dim"]),
        action_dim=int(cfg["action_dim"]),
        hidden_dim=int(cfg["dynamics_hidden_dim"]),
        layers=int(cfg["dynamics_layers"]),
        predict_root=bool(cfg.get("predict_root", False)),
        predict_reward=bool(cfg.get("predict_reward", False)),
        predict_done=bool(cfg.get("predict_done", False)),
    ).to(device)
    _load_dynamics_checkpoint(dynamics, checkpoint, device)

    test_trajectories = list(range(args.external_start, args.external_end))
    raw_arrays = _build_external_arrays(dataset, test_trajectories)
    arrays = _normalize(raw_arrays, obs_mean, obs_std) if normalize_obs else raw_arrays
    eval_args = SimpleNamespace(
        _obs_mean=obs_mean,
        _obs_std=obs_std,
        force_last_action=args.force_last_action,
        formula_reward=args.formula_reward,
        formula_done=args.formula_done,
        _observable_reward_context=_make_observable_reward_context(device)
        if args.formula_reward or args.formula_done
        else None,
    )
    metrics = _evaluate_dynamics_arrays(
        dynamics=dynamics,
        arrays=arrays,
        device=device,
        args=eval_args,
    )
    result: dict[str, object] = {
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "external_test_trajectories": test_trajectories,
        "external_test_transitions": int(raw_arrays.size),
        "normalize_obs": bool(normalize_obs),
        "metrics": metrics,
        "naive_next_equals_current": _naive_next_equals_current(arrays),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="External-test a G1 HRLG root-aware dynamics checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--external-start", type=int, required=True)
    parser.add_argument("--external-end", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--formula-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formula-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-last-action", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = evaluate(args)
    metrics = result["metrics"]
    print(f"output={args.output.expanduser().resolve()}")
    print(
        json.dumps(
            {
                "next_obs_mse": metrics.get("next_obs_mse"),
                "qpos_root_next_mse": metrics.get("qpos_root_next_mse"),
                "qvel_root_next_mse": metrics.get("qvel_root_next_mse"),
                "formula_reward_mae": metrics.get("formula_reward_mae"),
                "formula_reward_mse": metrics.get("formula_reward_mse"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
