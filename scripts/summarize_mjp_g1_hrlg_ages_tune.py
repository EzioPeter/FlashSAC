#!/usr/bin/env python3
"""Summarize G1 HRLG AGES-MOPO tuning runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FIELDS = [
    "run_id",
    "stage",
    "output_dir",
    "status",
    "exit_code",
    "actor_rl_coef",
    "bc_alpha",
    "bc_ref_alpha",
    "bc_ref_on",
    "real_ratio",
    "mopo_penalty_coef",
    "reward_scale",
    "ages_action_temperature",
    "ages_sim_ratio",
    "ages_select_ratio",
    "ages_num_dataset_states",
    "ages_trajectories_per_state",
    "ages_sim_rollout_length",
    "ages_inject_rollout_length",
    "ages_buffer_retain_epochs",
    "best_epoch",
    "best_return",
    "best_length",
    "best_survived_full",
    "final_return",
    "final_length",
    "final_survived_full",
    "final_offline_bc_mse_real",
    "final_offline_ref_mse",
    "max_ages_selected_adv_mean",
    "mean_ages_accept_rate",
    "score",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _stage_from_name(path: Path) -> str:
    name = path.name
    if "stageB" in name:
        return "B"
    if "stageC" in name:
        return "C"
    marker = "_A"
    if marker in name:
        return "A"
    return ""


def _best_online(rows: list[dict[str, str]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for row in rows:
        candidate = {
            "epoch": _int(row.get("epoch")),
            "avg_return": _float(row.get("avg_return")),
            "avg_length": _float(row.get("avg_length")),
            "survived_full": _int(row.get("survived_full")),
        }
        if not best or (
            candidate["survived_full"],
            candidate["avg_length"],
            candidate["avg_return"],
        ) > (
            best["survived_full"],
            best["avg_length"],
            best["avg_return"],
        ):
            best = candidate
    return best


def _summarize_run(path: Path) -> dict[str, Any]:
    cfg = _read_json(path / "resolved_config.json")
    metrics = _read_json(path / "metrics.json")
    progress = _read_csv(path / "progress.csv")
    online = _read_csv(path / "online_eval.csv")
    exit_code_text = (path / "exit_code").read_text(encoding="utf-8").strip() if (path / "exit_code").exists() else ""
    exit_code = _int(exit_code_text, -1 if not metrics else 0)
    status = metrics.get("status") or ("passed" if exit_code == 0 else "failed")

    final_progress = progress[-1] if progress else {}
    final_online = online[-1] if online else {}
    best = _best_online(online)
    if not best and progress:
        pseudo = [
            {
                "epoch": row.get("epoch", ""),
                "avg_return": row.get("online_avg_return", ""),
                "avg_length": row.get("online_avg_length", ""),
                "survived_full": row.get("online_survived_full", ""),
            }
            for row in progress
            if row.get("online_avg_return", "") not in ("", "nan", "NaN")
        ]
        best = _best_online(pseudo)

    best_return = _float(best.get("avg_return"))
    best_length = _float(best.get("avg_length"))
    best_survived = _int(best.get("survived_full"))
    score = best_return + 0.01 * best_length + 0.2 * best_survived

    adv_values = [_float(row.get("ages_selected_adv_mean")) for row in progress]
    adv_values = [value for value in adv_values if math.isfinite(value)]
    accept_values = [_float(row.get("ages_accept_rate")) for row in progress]
    accept_values = [value for value in accept_values if math.isfinite(value)]

    return {
        "run_id": path.name,
        "stage": _stage_from_name(path),
        "output_dir": str(path),
        "status": status,
        "exit_code": exit_code,
        "actor_rl_coef": cfg.get("actor_rl_coef", ""),
        "bc_alpha": cfg.get("bc_alpha", ""),
        "bc_ref_alpha": cfg.get("bc_ref_alpha", ""),
        "bc_ref_on": cfg.get("bc_ref_on", ""),
        "real_ratio": cfg.get("real_ratio", ""),
        "mopo_penalty_coef": cfg.get("mopo_penalty_coef", ""),
        "reward_scale": cfg.get("reward_scale", ""),
        "ages_action_temperature": cfg.get("ages_action_temperature", ""),
        "ages_sim_ratio": cfg.get("ages_sim_ratio", ""),
        "ages_select_ratio": cfg.get("ages_select_ratio", ""),
        "ages_num_dataset_states": cfg.get("ages_num_dataset_states", ""),
        "ages_trajectories_per_state": cfg.get("ages_trajectories_per_state", ""),
        "ages_sim_rollout_length": cfg.get("ages_sim_rollout_length", ""),
        "ages_inject_rollout_length": cfg.get("ages_inject_rollout_length", ""),
        "ages_buffer_retain_epochs": cfg.get("ages_buffer_retain_epochs", ""),
        "best_epoch": _int(best.get("epoch")),
        "best_return": best_return,
        "best_length": best_length,
        "best_survived_full": best_survived,
        "final_return": _float(final_online.get("avg_return", final_progress.get("online_avg_return"))),
        "final_length": _float(final_online.get("avg_length", final_progress.get("online_avg_length"))),
        "final_survived_full": _int(final_online.get("survived_full", final_progress.get("online_survived_full"))),
        "final_offline_bc_mse_real": _float(final_progress.get("offline_bc_mse_real")),
        "final_offline_ref_mse": _float(final_progress.get("offline_ref_mse_mixed")),
        "max_ages_selected_adv_mean": max(adv_values) if adv_values else math.nan,
        "mean_ages_accept_rate": sum(accept_values) / len(accept_values) if accept_values else math.nan,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/mjp_g1_hrlg_ages_mopo"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mjp_g1_hrlg_ages_tune50_step120000_summary.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    if args.root.exists():
        for path in sorted(args.root.iterdir()):
            if path.is_dir() and (path / "resolved_config.json").exists():
                runs.append(_summarize_run(path))
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in runs:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    print(json.dumps({"output": str(args.output), "runs": len(runs)}, sort_keys=True))


if __name__ == "__main__":
    main()
