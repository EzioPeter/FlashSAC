#!/usr/bin/env python3
"""Offline behavior cloning for MuJoCo Playground G1 HRLG actor observations."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import constants as hrlg


DEFAULT_DATASET = REPO_ROOT / "outputs/mjp_g1_hrlg_seed0_0521_154319_step48829_buffer_100x1000.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models/mjp_g1_hrlg_bc/seed0_100traj"


@dataclass(frozen=True)
class DatasetSplit:
    train_trajectories: list[int]
    val_trajectories: list[int]
    train_transitions: int
    val_transitions: int
    obs_dim: int
    action_dim: int


class ObservationActionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(f"Mismatched observation/action rows: {observations.shape[0]} vs {actions.shape[0]}")
        self.observations = torch.from_numpy(observations.astype(np.float32, copy=False))
        self.actions = torch.from_numpy(actions.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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


def _load_dataset(
    path: Path,
    *,
    val_ratio: float,
    seed: int,
    limit_trajectories: int | None,
    limit_transitions: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetSplit]:
    with np.load(path) as data:
        states = data["state"].astype(np.float32, copy=False)
        actions = data["action"].astype(np.float32, copy=False)

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

    train_obs = states[train_traj].reshape(-1, states.shape[-1])
    train_actions = actions[train_traj].reshape(-1, actions.shape[-1])
    val_obs = states[val_traj].reshape(-1, states.shape[-1])
    val_actions = actions[val_traj].reshape(-1, actions.shape[-1])

    if limit_transitions is not None:
        if limit_transitions < 2:
            raise ValueError("--limit-transitions must be >= 2.")
        train_limit = min(limit_transitions, train_obs.shape[0])
        val_limit = min(max(1, limit_transitions // 5), val_obs.shape[0])
        train_idx = np.sort(rng.choice(train_obs.shape[0], size=train_limit, replace=False))
        val_idx = np.sort(rng.choice(val_obs.shape[0], size=val_limit, replace=False))
        train_obs = train_obs[train_idx]
        train_actions = train_actions[train_idx]
        val_obs = val_obs[val_idx]
        val_actions = val_actions[val_idx]

    split = DatasetSplit(
        train_trajectories=[int(x) for x in train_traj.tolist()],
        val_trajectories=[int(x) for x in val_traj.tolist()],
        train_transitions=int(train_obs.shape[0]),
        val_transitions=int(val_obs.shape[0]),
        obs_dim=int(states.shape[-1]),
        action_dim=int(actions.shape[-1]),
    )
    return train_obs, train_actions, val_obs, val_actions, split


def _fit_obs_stats(observations: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = observations.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std


def _make_loader(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ObservationActionDataset(observations, actions),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def _make_actor(device: torch.device) -> FlashSACActor:
    return FlashSACActor(
        num_blocks=2,
        input_dim=hrlg.POLICY_OBS_SIZE,
        hidden_dim=128,
        action_dim=hrlg.ACTION_SIZE,
    ).to(device)


def _predict(actor: FlashSACActor, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = actor.get_mean_and_std(observations, training=training)
    return torch.tanh(mean), std


@torch.no_grad()
def _evaluate_bc(
    actor: FlashSACActor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, float]:
    actor.eval()
    losses = []
    maes = []
    max_abs_errors = []
    std_means = []
    for observations, actions in loader:
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        predicted, std = _predict(actor, observations, training=False)
        error = predicted - actions
        losses.append(float(torch.mean(error.square()).item()))
        maes.append(float(torch.mean(torch.abs(error)).item()))
        max_abs_errors.append(float(torch.max(torch.abs(error)).item()))
        std_means.append(float(torch.mean(std).item()))
    return {
        "mse": float(np.mean(losses)) if losses else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "max_abs_error": float(np.max(max_abs_errors)) if max_abs_errors else float("nan"),
        "policy_std_mean": float(np.mean(std_means)) if std_means else float("nan"),
    }


def _save_checkpoint(
    path: Path,
    *,
    actor: FlashSACActor,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    epoch: int,
    val_metrics: dict[str, float],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    args: argparse.Namespace,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["dataset"] = str(args.dataset.expanduser().resolve())
    config["output_dir"] = str(args.output_dir.expanduser().resolve())
    torch.save(
        {
            "network_state_dict": actor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "update_step": int(update_step),
            "epoch": int(epoch),
            "val_metrics": val_metrics,
            "obs_normalization": bool(args.normalize_obs),
            "obs_mean": torch.from_numpy(obs_mean),
            "obs_std": torch.from_numpy(obs_std),
            "config": config,
        },
        path / "actor.pt",
    )
    np.savez(path / "obs_stats.npz", mean=obs_mean, std=obs_std, enabled=np.array(bool(args.normalize_obs)))


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
        seed=args.online_eval_seed if args.online_eval_seed is not None else args.seed + 10_000 + epoch,
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


def _run_training(args: argparse.Namespace) -> dict[str, Any]:
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_obs, train_actions, val_obs, val_actions, split = _load_dataset(
        args.dataset.expanduser().resolve(),
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_trajectories=args.limit_trajectories,
        limit_transitions=args.limit_transitions,
    )
    obs_mean, obs_std = _fit_obs_stats(train_obs)
    if args.normalize_obs:
        train_obs = (train_obs - obs_mean) / obs_std
        val_obs = (val_obs - obs_mean) / obs_std
    else:
        obs_mean = np.zeros_like(obs_mean)
        obs_std = np.ones_like(obs_std)

    train_loader = _make_loader(
        train_obs,
        train_actions,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = _make_loader(
        val_obs,
        val_actions,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed + 1,
        num_workers=args.num_workers,
    )

    actor = _make_actor(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["dataset"] = str(args.dataset.expanduser().resolve())
    config["output_dir"] = str(output_dir)
    _write_json(output_dir / "resolved_config.json", config)
    _write_json(
        output_dir / "offline_provenance.json",
        {
            "status": "offline_only_verified",
            "training_script": str(Path(__file__).resolve()),
            "dataset_path": str(args.dataset.expanduser().resolve()),
            "training_data_sources": ["mujoco_playground_hrlg_npz_dataset"],
            "uses_online_env_transitions": False,
            "split_by": "trajectory",
            **asdict(split),
        },
    )

    progress_path = output_dir / "progress.csv"
    log_path = output_dir / "train.log"
    best_val_mse = float("inf")
    best_metrics: dict[str, float] | None = None
    best_update = 0
    update_step = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    online_eval_metrics: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    stop_training = False

    with (
        progress_path.open("w", encoding="utf-8", newline="") as csv_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        fieldnames = [
            "epoch",
            "update",
            "train_mse",
            "val_mse",
            "val_mae",
            "val_max_abs_error",
            "policy_std_mean",
            "wall_time_sec",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            actor.train()
            epoch_losses = []
            for observations, actions in train_loader:
                observations = observations.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                predicted, _ = _predict(actor, observations, training=True)
                loss = F.mse_loss(predicted, actions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite BC loss at update {update_step + 1}: {float(loss.item())}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip_norm)
                optimizer.step()

                update_step += 1
                loss_value = float(loss.item())
                epoch_losses.append(loss_value)
                train_losses.append(loss_value)

                if args.max_train_steps is not None and update_step >= args.max_train_steps:
                    stop_training = True
                    break

            train_mse = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            should_eval = epoch == 1 or epoch % args.eval_interval == 0 or stop_training or epoch == args.epochs
            if should_eval:
                val_metrics = _evaluate_bc(actor, val_loader, device=device)
                val_losses.append(float(val_metrics["mse"]))
                if val_metrics["mse"] < best_val_mse:
                    best_val_mse = float(val_metrics["mse"])
                    best_metrics = val_metrics
                    best_update = update_step
                    _save_checkpoint(
                        output_dir / "best",
                        actor=actor,
                        optimizer=optimizer,
                        update_step=update_step,
                        epoch=epoch,
                        val_metrics=val_metrics,
                        obs_mean=obs_mean,
                        obs_std=obs_std,
                        args=args,
                    )
            else:
                val_metrics = {
                    "mse": float("nan"),
                    "mae": float("nan"),
                    "max_abs_error": float("nan"),
                    "policy_std_mean": float("nan"),
                }

            if epoch == 1 or epoch % args.save_interval == 0 or stop_training or epoch == args.epochs:
                _save_checkpoint(
                    output_dir / f"epoch_{epoch:04d}",
                    actor=actor,
                    optimizer=optimizer,
                    update_step=update_step,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    args=args,
                )

            record = {
                "epoch": epoch,
                "update": update_step,
                "train_mse": train_mse,
                "val_mse": float(val_metrics["mse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_max_abs_error": float(val_metrics["max_abs_error"]),
                "policy_std_mean": float(val_metrics["policy_std_mean"]),
                "wall_time_sec": float(time.perf_counter() - start_time),
            }
            writer.writerow(record)
            csv_file.flush()
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
            log_file.flush()
            print(json.dumps(record, sort_keys=True))

            should_online_eval = (
                args.online_eval_interval > 0
                and (epoch % args.online_eval_interval == 0 or stop_training or epoch == args.epochs)
            )
            if should_online_eval:
                eval_checkpoint = output_dir / f"epoch_{epoch:04d}"
                if not (eval_checkpoint / "actor.pt").exists():
                    _save_checkpoint(
                        eval_checkpoint,
                        actor=actor,
                        optimizer=optimizer,
                        update_step=update_step,
                        epoch=epoch,
                        val_metrics=val_metrics,
                        obs_mean=obs_mean,
                        obs_std=obs_std,
                        args=args,
                    )
                online_eval_metrics.append(
                    _run_online_eval(
                        eval_checkpoint,
                        args=args,
                        epoch=epoch,
                        update_step=update_step,
                        output_dir=output_dir,
                    )
                )

            if stop_training:
                break

    final_val_metrics = _evaluate_bc(actor, val_loader, device=device)
    _save_checkpoint(
        output_dir / "final",
        actor=actor,
        optimizer=optimizer,
        update_step=update_step,
        epoch=epoch,
        val_metrics=final_val_metrics,
        obs_mean=obs_mean,
        obs_std=obs_std,
        args=args,
    )

    metrics = {
        "status": "passed",
        "seed": args.seed,
        "device": str(device),
        "dataset_path": str(args.dataset.expanduser().resolve()),
        "output_dir": str(output_dir),
        **asdict(split),
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "max_train_steps": args.max_train_steps,
        "update_steps": update_step,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "normalize_obs": args.normalize_obs,
        "train_mse_last": float(train_losses[-1]) if train_losses else None,
        "train_mse_mean": float(np.mean(train_losses)) if train_losses else None,
        "best_val_mse": best_val_mse,
        "best_update": best_update,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_val_metrics,
        "online_eval_interval": args.online_eval_interval,
        "online_eval_count": len(online_eval_metrics),
        "online_eval_last": online_eval_metrics[-1] if online_eval_metrics else None,
        "online_eval_csv": str((output_dir / "online_eval.csv").resolve()) if online_eval_metrics else None,
        "online_eval_jsonl": str((output_dir / "online_eval.jsonl").resolve()) if online_eval_metrics else None,
        "best_checkpoint": str((output_dir / "best").resolve()),
        "final_checkpoint": str((output_dir / "final").resolve()),
        "progress_csv": str(progress_path.resolve()),
        "train_log": str(log_path.resolve()),
        "wall_time_sec": float(time.perf_counter() - start_time),
    }
    _write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MuJoCo Playground G1 HRLG behavior cloning actor.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--online-eval-interval", type=int, default=0)
    parser.add_argument("--online-eval-episodes", type=int, default=50)
    parser.add_argument("--online-eval-num-envs", type=int, default=50)
    parser.add_argument("--online-eval-max-episode-steps", type=int, default=1000)
    parser.add_argument("--online-eval-seed", type=int, default=None)
    parser.add_argument("--online-eval-device", type=str, default="cpu")
    parser.add_argument("--online-eval-noise-level", type=float, default=1.0)
    parser.add_argument("--limit-trajectories", type=int, default=None)
    parser.add_argument("--limit-transitions", type=int, default=None)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _run_training(args)


if __name__ == "__main__":
    main()
