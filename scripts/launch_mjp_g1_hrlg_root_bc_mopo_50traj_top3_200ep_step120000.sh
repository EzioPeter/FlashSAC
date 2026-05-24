#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
ONLINE_EVAL_DEVICE="${ONLINE_EVAL_DEVICE:-cuda:0}"

DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000_observable_reward.npz"
BC_CHECKPOINT="models/mjp_g1_hrlg_bc/seed0_0523_202023_step120000_dr_warmstart_50traj_100ksteps_noeval/final"
WORLD_MODEL_CKPT="models/mjp_g1_hrlg_bc_mopo/root_world_model_seed0_0523_202023_step120000_dr_warmstart_50traj_observable/best_model"
LOG_ROOT="outputs/mjp_g1_hrlg_bc_mopo_50traj_top3_200ep_step120000"
SUMMARY_CSV="${LOG_ROOT}/summary.csv"

mkdir -p "${LOG_ROOT}"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "JAX_PLATFORMS=cuda"
  "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "OMP_NUM_THREADS=2"
  "MKL_NUM_THREADS=2"
  "NUMEXPR_NUM_THREADS=2"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log "missing required file: $1"
    exit 1
  fi
}

write_summary() {
  uv run python - <<'PY'
import csv
import json
import math
from pathlib import Path

log_root = Path("outputs/mjp_g1_hrlg_bc_mopo_50traj_top3_200ep_step120000")
rows = []

def as_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan

for out_dir in sorted(Path("models/mjp_g1_hrlg_bc_mopo").glob("root_50traj_stageA*_bc_mopo_200ep_eval10_step120000")):
    cfg = {}
    cfg_path = out_dir / "queue_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    exit_code = (out_dir / "exit_code").read_text().strip() if (out_dir / "exit_code").exists() else ""
    status = "ok" if exit_code == "0" else ("running" if not exit_code else "failed")

    eval_rows = []
    eval_csv = out_dir / "online_eval.csv"
    if eval_csv.exists():
        with eval_csv.open(newline="") as f:
            eval_rows = list(csv.DictReader(f))
    best = {}
    final = {}
    if eval_rows:
        best = max(eval_rows, key=lambda r: (as_float(r.get("avg_length")), as_float(r.get("avg_return"))))
        final = eval_rows[-1]

    rows.append({
        "id": cfg.get("id", out_dir.name),
        "output_dir": str(out_dir),
        "actor_rl_coef": cfg.get("actor_rl_coef", ""),
        "bc_alpha": cfg.get("bc_alpha", ""),
        "bc_ref_alpha": cfg.get("bc_ref_alpha", ""),
        "bc_ref_on": cfg.get("bc_ref_on", ""),
        "reward_scale": cfg.get("reward_scale", ""),
        "real_ratio": cfg.get("real_ratio", ""),
        "mopo_penalty_coef": cfg.get("mopo_penalty_coef", ""),
        "best_epoch": best.get("epoch", ""),
        "best_return": best.get("avg_return", ""),
        "best_length": best.get("avg_length", ""),
        "best_survived_full": best.get("survived_full", ""),
        "final_epoch": final.get("epoch", ""),
        "final_return": final.get("avg_return", ""),
        "final_length": final.get("avg_length", ""),
        "final_survived_full": final.get("survived_full", ""),
        "status": status,
        "exit_code": exit_code,
    })

fieldnames = [
    "id", "output_dir", "actor_rl_coef", "bc_alpha", "bc_ref_alpha", "bc_ref_on",
    "reward_scale", "real_ratio", "mopo_penalty_coef",
    "best_epoch", "best_return", "best_length", "best_survived_full",
    "final_epoch", "final_return", "final_length", "final_survived_full",
    "status", "exit_code",
]
log_root.mkdir(parents=True, exist_ok=True)
with (log_root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {log_root / 'summary.csv'} rows={len(rows)}")
PY
}

launch_job() {
  local id="$1"
  local actor_rl_coef="$2"
  local bc_alpha="$3"
  local bc_ref_alpha="$4"
  local bc_ref_on="$5"
  local reward_scale="$6"
  local real_ratio="$7"
  local mopo_penalty="$8"

  local output_dir="models/mjp_g1_hrlg_bc_mopo/root_50traj_stage${id}_bc_mopo_200ep_eval10_step120000"
  mkdir -p "${output_dir}"

  if [[ -f "${output_dir}/exit_code" ]] && [[ "$(cat "${output_dir}/exit_code")" == "0" ]]; then
    log "skip completed ${id}: ${output_dir}"
    return
  fi

  cat >"${output_dir}/queue_config.json" <<JSON
{
  "id": "${id}",
  "source": "stageA_top3",
  "limit_trajectories": 50,
  "epochs": 200,
  "online_eval_interval": 10,
  "online_eval_episodes": 50,
  "actor_rl_coef": ${actor_rl_coef},
  "bc_alpha": ${bc_alpha},
  "bc_ref_alpha": ${bc_ref_alpha},
  "bc_ref_on": "${bc_ref_on}",
  "reward_scale": ${reward_scale},
  "real_ratio": ${real_ratio},
  "mopo_penalty_coef": ${mopo_penalty},
  "dataset": "${DATASET}",
  "bc_checkpoint": "${BC_CHECKPOINT}",
  "world_model_checkpoint": "${WORLD_MODEL_CKPT}"
}
JSON

  (
    date '+%Y-%m-%d %H:%M:%S' >"${output_dir}/start_time"
    set +e
    env "${COMMON_ENV[@]}" uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      --dataset "${DATASET}" \
      --reward-source observable \
      --limit-trajectories 50 \
      --dynamics-checkpoint "${WORLD_MODEL_CKPT}" \
      --bc-checkpoint "${BC_CHECKPOINT}" \
      --skip-dynamics-training \
      --predict-root \
      --formula-reward \
      --formula-done \
      --no-predict-reward-head \
      --no-predict-done-head \
      --rollout-horizon 1 \
      --actor-lr 1e-5 \
      --critic-lr 3e-4 \
      --temperature-lr 3e-4 \
      --critic-min-v -5 \
      --critic-max-v 20 \
      --critic-warmup-epochs 5 \
      --actor-update-period 4 \
      --no-actor-weight-normalization \
      --epochs 200 \
      --online-eval-interval 10 \
      --online-eval-episodes 50 \
      --online-eval-num-envs 50 \
      --online-eval-max-episode-steps 1000 \
      --save-interval 10 \
      --device "${DEVICE}" \
      --online-eval-device "${ONLINE_EVAL_DEVICE}" \
      --seed 0 \
      --batch-size 1024 \
      --model-batch-size 1024 \
      --rollout-batch-size 4096 \
      --actor-rl-coef "${actor_rl_coef}" \
      --bc-alpha "${bc_alpha}" \
      --bc-ref-alpha "${bc_ref_alpha}" \
      --bc-ref-on "${bc_ref_on}" \
      --reward-scale "${reward_scale}" \
      --real-ratio "${real_ratio}" \
      --mopo-penalty-coef "${mopo_penalty}" \
      --output-dir "${output_dir}" \
      >"${output_dir}/train_stdout.log" 2>&1
    code=$?
    echo "${code}" >"${output_dir}/exit_code"
    date '+%Y-%m-%d %H:%M:%S' >"${output_dir}/end_time"
    exit "${code}"
  ) &

  local pid=$!
  echo "${pid}" >"${output_dir}/pid"
  PIDS+=("${pid}")
  log "launched ${id} pid=${pid}: ${output_dir}"
}

require_file "${DATASET}"
require_file "${BC_CHECKPOINT}/actor.pt"
require_file "${WORLD_MODEL_CKPT}/dynamics_ensemble.pt"

PIDS=()
launch_job "A11" 0.003 100 30 synthetic 5 0.95 10
launch_job "A06" 0.001 300 30 mixed 5 0.95 10
launch_job "A22" 0.001 300 30 synthetic 10 0.95 20

log "all top3 50traj 200ep jobs launched; waiting"
for pid in "${PIDS[@]}"; do
  wait "${pid}" || true
  write_summary || true
done

write_summary
log "finished. Summary: ${SUMMARY_CSV}"
