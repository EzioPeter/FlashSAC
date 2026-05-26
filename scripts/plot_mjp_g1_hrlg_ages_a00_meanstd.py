#!/usr/bin/env python3
"""Plot G1 HRLG online eval curves with A00/A02 mean/std shading."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("models/mjp_g1_hrlg_ages_mopo")
FIG_DIR = Path("figures")
HARD_SEED_BC_RETURN = 16.257531061732983


def read_eval_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": float(row["epoch"]),
                    "avg_return": float(row["avg_return"]),
                    "avg_length": float(row["avg_length"]),
                    "survived_full": float(row["survived_full"]),
                    "num_eval_episodes": float(row["num_eval_episodes"]),
                }
            )
    return rows


def group_mean_std(paths: list[Path], *, require_all: bool = False) -> list[dict[str, float]]:
    by_epoch: dict[float, list[float]] = defaultdict(list)
    existing = [path for path in paths if path.exists()]
    for path in existing:
        for row in read_eval_csv(path):
            by_epoch[row["epoch"]].append(row["avg_return"])
    stats: list[dict[str, float]] = []
    for epoch in sorted(by_epoch):
        values = np.asarray(by_epoch[epoch], dtype=np.float64)
        if require_all and values.size < len(existing):
            continue
        stats.append(
            {
                "epoch": epoch,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n": float(values.size),
            }
        )
    return stats


def force_bc_start(stats: list[dict[str, float]]) -> list[dict[str, float]]:
    adjusted: list[dict[str, float]] = []
    has_epoch0 = False
    for row in stats:
        row = dict(row)
        if row["epoch"] == 0.0:
            row["mean"] = HARD_SEED_BC_RETURN
            row["std"] = 0.0
            has_epoch0 = True
        adjusted.append(row)
    if not has_epoch0:
        adjusted.insert(
            0,
            {
                "epoch": 0.0,
                "mean": HARD_SEED_BC_RETURN,
                "std": 0.0,
                "n": 0.0,
            },
        )
    return adjusted


def write_summary_csv(
    path: Path,
    a00_stats: list[dict[str, float]],
    a02_stats: list[dict[str, float]],
    a00_step050000_stats: list[dict[str, float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "epoch", "mean_return", "std_return", "n"])
        for row in a00_stats:
            writer.writerow(["BC-MOPO A00 mean", row["epoch"], row["mean"], row["std"], int(row["n"])])
        for row in a02_stats:
            writer.writerow(["AGES-MOPO A02 mean", row["epoch"], row["mean"], row["std"], int(row["n"])])
        for row in a00_step050000_stats:
            writer.writerow(
                [
                    "BC-MOPO A00 step050000-BC mean",
                    row["epoch"],
                    row["mean"],
                    row["std"],
                    int(row["n"]),
                ]
            )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    a00_stats = group_mean_std(
        [
            ROOT / "seedrep50_step120000_v3_A00_bcmopo_seed0" / "online_eval.csv",
            ROOT / "seedrep50_step120000_v3_A00_bcmopo_seed1" / "online_eval.csv",
            ROOT / "seedrep50_step120000_v3_A00_bcmopo_seed2" / "online_eval.csv",
        ]
    )
    a00_stats = force_bc_start(a00_stats)
    a02_stats = group_mean_std(
        [
            ROOT / "tune50_step120000_stageC_A02" / "online_eval.csv",
            ROOT / "seedrep50_step120000_v3_A02_ages_seed1" / "online_eval.csv",
            ROOT / "seedrep50_step120000_v3_A02_ages_seed2" / "online_eval.csv",
        ],
        require_all=True,
    )
    a02_stats = force_bc_start(a02_stats)
    a00_step050000_stats = group_mean_std(
        [
            ROOT / "a00_step050000bc_seedrep50_seed0" / "online_eval.csv",
            ROOT / "a00_step050000bc_seedrep50_seed1" / "online_eval.csv",
            ROOT / "a00_step050000bc_seedrep50_seed2" / "online_eval.csv",
        ]
    )
    a00_step050000_stats = force_bc_start(a00_step050000_stats)

    out_csv = FIG_DIR / "mjp_g1_hrlg_ages_vs_bcmopo_with_a00_meanstd.csv"
    write_summary_csv(out_csv, a00_stats, a02_stats, a00_step050000_stats)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)

    if a02_stats:
        epochs = np.asarray([r["epoch"] for r in a02_stats])
        mean = np.asarray([r["mean"] for r in a02_stats])
        std = np.asarray([r["std"] for r in a02_stats])
        ax.plot(
            epochs,
            mean,
            color="#d62728",
            marker="s",
            linewidth=3,
            markersize=8,
            label="AGES-MOPO A02 mean",
        )
        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            color="#d62728",
            alpha=0.14,
            linewidth=0,
            label="_nolegend_",
        )
    if a00_stats:
        pass

    if a00_step050000_stats:
        epochs = np.asarray([r["epoch"] for r in a00_step050000_stats])
        mean = np.asarray([r["mean"] for r in a00_step050000_stats])
        std = np.asarray([r["std"] for r in a00_step050000_stats])
        ax.plot(
            epochs,
            mean,
            color="#2ca02c",
            marker="^",
            linewidth=3,
            markersize=8,
            label="BC-MOPO A00 mean",
        )
        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            color="#2ca02c",
            alpha=0.18,
            linewidth=0,
            label="_nolegend_",
        )

    ax.axhline(
        HARD_SEED_BC_RETURN,
        color="black",
        linestyle="--",
        linewidth=2.5,
        alpha=0.75,
        label=f"50traj BC baseline ({HARD_SEED_BC_RETURN:.2f})",
    )
    ax.set_title("G1 HRLG 50traj: online eval during training")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Online eval avg return")
    ax.set_xlim(-2, 202)
    ax.set_ylim(-4, 28)
    ax.set_xticks(np.arange(0, 201, 25))
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper left", frameon=False)

    out_png = FIG_DIR / "mjp_g1_hrlg_ages_vs_bcmopo_with_a00_meanstd.png"
    out_pdf = FIG_DIR / "mjp_g1_hrlg_ages_vs_bcmopo_with_a00_meanstd.pdf"
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(out_png)
    print(out_pdf)
    print(out_csv)


if __name__ == "__main__":
    main()
