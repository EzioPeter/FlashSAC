#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MAX_JOBS="${MAX_JOBS:-2}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
ONLINE_EVAL_DEVICE="${ONLINE_EVAL_DEVICE:-cuda:0}"

RAW_DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000.npz"
DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000_observable_reward.npz"
BC_CHECKPOINT="models/mjp_g1_hrlg_bc/seed0_0523_202023_step120000_dr_warmstart_25traj_100ksteps_noeval/final"
WORLD_MODEL_DIR="models/mjp_g1_hrlg_bc_mopo/root_world_model_seed0_0523_202023_step120000_dr_warmstart_25traj_observable"
WORLD_MODEL_CKPT="${WORLD_MODEL_DIR}/best_model"

LOG_ROOT="outputs/mjp_g1_hrlg_bc_mopo_tune25_stageA"
SUMMARY_CSV="outputs/mjp_g1_hrlg_bc_mopo_tune25_stageA_summary.csv"
mkdir -p "${LOG_ROOT}" "${WORLD_MODEL_DIR}"

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

run_preflight() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  local exit_file="${LOG_ROOT}/${name}.exit_code"
  log "start ${name}"
  set +e
  env "${COMMON_ENV[@]}" "$@" >"${log_file}" 2>&1
  local code=$?
  set -e
  echo "${code}" >"${exit_file}"
  if [[ "${code}" -ne 0 ]]; then
    log "${name} failed with exit code ${code}; see ${log_file}"
    exit "${code}"
  fi
  log "done ${name}"
}

ensure_relabel() {
  if [[ -f "${DATASET}" ]]; then
    log "relabeled dataset exists: ${DATASET}"
    return
  fi
  if [[ ! -f "${RAW_DATASET}" ]]; then
    log "missing raw dataset: ${RAW_DATASET}"
    exit 1
  fi
  run_preflight relabel_step120000 \
    uv run python scripts/relabel_mjp_g1_hrlg_observable_reward.py \
      --input "${RAW_DATASET}" \
      --output "${DATASET}" \
      --no-save-padded
}

ensure_world_model() {
  if [[ -f "${WORLD_MODEL_CKPT}/dynamics_ensemble.pt" ]]; then
    log "world model exists: ${WORLD_MODEL_CKPT}"
    return
  fi
  run_preflight world_model_25traj_train \
    uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      --dataset "${DATASET}" \
      --reward-source observable \
      --limit-trajectories 25 \
      --output-dir "${WORLD_MODEL_DIR}" \
      --seed 0 \
      --device "${DEVICE}" \
      --dynamics-only \
      --predict-root \
      --formula-reward \
      --formula-done \
      --no-predict-reward-head \
      --no-predict-done-head \
      --dynamics-ensemble-size 5 \
      --dynamics-hidden-dim 256 \
      --dynamics-layers 3 \
      --dynamics-epochs 50 \
      --model-batch-size 1024 \
      --dynamics-lr 3e-4
}

run_external_test() {
  local output_json="${LOG_ROOT}/world_model_25traj_external_test.json"
  if [[ -f "${output_json}" ]]; then
    log "external world model test exists: ${output_json}"
    return
  fi
  run_preflight world_model_25traj_external_test \
    uv run python scripts/eval_mjp_g1_hrlg_root_world_model_external.py \
      --dataset "${DATASET}" \
      --checkpoint "${WORLD_MODEL_CKPT}" \
      --external-start 25 \
      --external-end 100 \
      --output "${output_json}" \
      --device "${DEVICE}" \
      --formula-reward \
      --formula-done
}

wait_for_slot() {
  while [[ "${ACTIVE_JOBS}" -ge "${MAX_JOBS}" ]]; do
    wait -n || true
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    write_summary || true
  done
}

launch_job() {
  local id="$1"
  local short_name="$2"
  local actor_rl_coef="$3"
  local bc_alpha="$4"
  local bc_ref_alpha="$5"
  local bc_ref_on="$6"
  local reward_scale="$7"
  local real_ratio="$8"
  local mopo_penalty="$9"

  local output_dir="models/mjp_g1_hrlg_bc_mopo/tune25_stageA_${id}_${short_name}"
  mkdir -p "${output_dir}"

  if [[ -f "${output_dir}/exit_code" ]] && [[ "$(cat "${output_dir}/exit_code")" == "0" ]]; then
    log "skip completed ${id}: ${output_dir}"
    return
  fi

  cat >"${output_dir}/queue_config.json" <<JSON
{
  "stage": "A",
  "id": "${id}",
  "short_name": "${short_name}",
  "actor_rl_coef": ${actor_rl_coef},
  "bc_alpha": ${bc_alpha},
  "bc_ref_alpha": ${bc_ref_alpha},
  "bc_ref_on": "${bc_ref_on}",
  "reward_scale": ${reward_scale},
  "real_ratio": ${real_ratio},
  "mopo_penalty_coef": ${mopo_penalty},
  "limit_trajectories": 25,
  "dataset": "${DATASET}",
  "world_model_checkpoint": "${WORLD_MODEL_CKPT}",
  "bc_checkpoint": "${BC_CHECKPOINT}"
}
JSON

  (
    echo "$$" >"${output_dir}/launcher_pid"
    date '+%Y-%m-%d %H:%M:%S' >"${output_dir}/start_time"
    set +e
    env "${COMMON_ENV[@]}" uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
      --dataset "${DATASET}" \
      --reward-source observable \
      --limit-trajectories 25 \
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
      --epochs 20 \
      --online-eval-interval 10 \
      --online-eval-episodes 10 \
      --online-eval-num-envs 10 \
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
  ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
  log "launched ${id} pid=${pid}: ${output_dir}"
}

write_summary() {
  uv run python - <<'PY'
import csv
import json
import math
from pathlib import Path

summary_path = Path("outputs/mjp_g1_hrlg_bc_mopo_tune25_stageA_summary.csv")
rows = []

def to_float(value):
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan

for out_dir in sorted(Path("models/mjp_g1_hrlg_bc_mopo").glob("tune25_stageA_A*")):
    cfg_path = out_dir / "queue_config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}
    status = "running"
    exit_code = ""
    exit_path = out_dir / "exit_code"
    if exit_path.exists():
        exit_code = exit_path.read_text().strip()
        status = "ok" if exit_code == "0" else "failed"

    eval_rows = []
    eval_csv = out_dir / "online_eval.csv"
    if eval_csv.exists():
        with eval_csv.open(newline="") as f:
            eval_rows = list(csv.DictReader(f))

    progress_rows = []
    progress_csv = out_dir / "progress.csv"
    if progress_csv.exists():
        with progress_csv.open(newline="") as f:
            progress_rows = list(csv.DictReader(f))

    best = {}
    final = {}
    if eval_rows:
        best = max(eval_rows, key=lambda r: (to_float(r.get("avg_length")), to_float(r.get("avg_return"))))
        final = eval_rows[-1]

    final_progress = progress_rows[-1] if progress_rows else {}
    max_offline_bc = math.nan
    min_offline_bc = math.nan
    if progress_rows:
        vals = [to_float(r.get("offline_bc_mse_real")) for r in progress_rows]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            min_offline_bc = min(vals)
            max_offline_bc = max(vals)

    bad_reason = ""
    final_len = to_float(final.get("avg_length"))
    final_bc_mse = to_float(final_progress.get("offline_bc_mse_real"))
    if not math.isnan(final_len) and final_len < 500:
        bad_reason = "avg_length_lt_500"
    if not math.isnan(final_bc_mse) and final_bc_mse > 0.05:
        bad_reason = (bad_reason + ";offline_bc_mse_gt_0.05").strip(";")

    rows.append({
        "stage": "A",
        "output_dir": str(out_dir),
        "id": cfg.get("id", out_dir.name),
        "actor_rl_coef": cfg.get("actor_rl_coef", ""),
        "bc_alpha": cfg.get("bc_alpha", ""),
        "bc_ref_alpha": cfg.get("bc_ref_alpha", ""),
        "bc_ref_on": cfg.get("bc_ref_on", ""),
        "reward_scale": cfg.get("reward_scale", ""),
        "real_ratio": cfg.get("real_ratio", ""),
        "mopo_penalty_coef": cfg.get("mopo_penalty_coef", ""),
        "actor_lr": "1e-5",
        "critic_warmup_epochs": 5,
        "actor_update_period": 4,
        "best_epoch": best.get("epoch", ""),
        "best_return": best.get("avg_return", ""),
        "best_length": best.get("avg_length", ""),
        "best_survived_full": best.get("survived_full", best.get("num_survived_full", "")),
        "final_epoch": final.get("epoch", ""),
        "final_return": final.get("avg_return", ""),
        "final_length": final.get("avg_length", ""),
        "final_survived_full": final.get("survived_full", final.get("num_survived_full", "")),
        "offline_bc_mse_real_final": final_progress.get("offline_bc_mse_real", ""),
        "offline_ref_mse_final": final_progress.get("offline_ref_mse_mixed", ""),
        "min_offline_bc_mse_real": min_offline_bc,
        "max_offline_bc_mse_real": max_offline_bc,
        "status": status,
        "exit_code": exit_code,
        "bad_reason": bad_reason,
    })

fieldnames = [
    "stage", "id", "output_dir", "actor_rl_coef", "bc_alpha", "bc_ref_alpha",
    "bc_ref_on", "reward_scale", "real_ratio", "mopo_penalty_coef",
    "actor_lr", "critic_warmup_epochs", "actor_update_period",
    "best_epoch", "best_return", "best_length", "best_survived_full",
    "final_epoch", "final_return", "final_length", "final_survived_full",
    "offline_bc_mse_real_final", "offline_ref_mse_final",
    "min_offline_bc_mse_real", "max_offline_bc_mse_real",
    "status", "exit_code", "bad_reason",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {summary_path} with {len(rows)} rows")
PY
}

ensure_relabel
ensure_world_model
run_external_test

if [[ ! -f "${DATASET}" ]]; then
  log "missing relabeled dataset after preflight: ${DATASET}"
  exit 1
fi
if [[ ! -f "${WORLD_MODEL_CKPT}/dynamics_ensemble.pt" ]]; then
  log "missing world model checkpoint after preflight: ${WORLD_MODEL_CKPT}/dynamics_ensemble.pt"
  exit 1
fi
if [[ ! -f "${BC_CHECKPOINT}/actor.pt" ]]; then
  log "missing 25traj BC checkpoint: ${BC_CHECKPOINT}/actor.pt"
  exit 1
fi

ACTIVE_JOBS=0

while IFS='|' read -r id short_name actor_rl_coef bc_alpha bc_ref_alpha bc_ref_on reward_scale real_ratio mopo_penalty; do
  [[ -z "${id}" || "${id}" == \#* ]] && continue
  wait_for_slot
  launch_job "${id}" "${short_name}" "${actor_rl_coef}" "${bc_alpha}" "${bc_ref_alpha}" "${bc_ref_on}" "${reward_scale}" "${real_ratio}" "${mopo_penalty}"
done <<'CONFIGS'
A01|rl0_bc100_ref30_rs5_r095_pen10|0.0|100|30|mixed|5|0.95|10
A02|rl0_bc300_ref100_rs5_r095_pen10|0.0|300|100|mixed|5|0.95|10
A03|rl0001_bc100_ref30_rs5_r095_pen10|0.001|100|30|mixed|5|0.95|10
A04|rl0003_bc100_ref30_rs5_r095_pen10|0.003|100|30|mixed|5|0.95|10
A05|rl001_bc100_ref30_rs5_r095_pen10|0.01|100|30|mixed|5|0.95|10
A06|rl0001_bc300_ref30_rs5_r095_pen10|0.001|300|30|mixed|5|0.95|10
A07|rl0003_bc300_ref30_rs5_r095_pen10|0.003|300|30|mixed|5|0.95|10
A08|rl0001_bc100_ref100_rs5_r095_pen10|0.001|100|100|mixed|5|0.95|10
A09|rl0003_bc100_ref100_rs5_r095_pen10|0.003|100|100|mixed|5|0.95|10
A10|rl0001_bc100_ref30_syn_rs5_r095_pen10|0.001|100|30|synthetic|5|0.95|10
A11|rl0003_bc100_ref30_syn_rs5_r095_pen10|0.003|100|30|synthetic|5|0.95|10
A12|rl0001_bc100_ref30_rs10_r095_pen10|0.001|100|30|mixed|10|0.95|10
A13|rl0003_bc100_ref30_rs10_r095_pen10|0.003|100|30|mixed|10|0.95|10
A14|rl0001_bc100_ref30_rs5_r095_pen5|0.001|100|30|mixed|5|0.95|5
A15|rl0003_bc100_ref30_rs5_r095_pen5|0.003|100|30|mixed|5|0.95|5
A16|rl0001_bc100_ref30_rs5_r099_pen10|0.001|100|30|mixed|5|0.99|10
A17|rl0003_bc100_ref30_rs5_r099_pen10|0.003|100|30|mixed|5|0.99|10
A18|rl0001_bc300_ref100_rs5_r099_pen10|0.001|300|100|mixed|5|0.99|10
A19|rl0003_bc300_ref100_rs5_r099_pen10|0.003|300|100|mixed|5|0.99|10
A20|rl0001_bc100_ref30_rs10_r099_pen20|0.001|100|30|mixed|10|0.99|20
A21|rl0003_bc100_ref30_rs10_r099_pen20|0.003|100|30|mixed|10|0.99|20
A22|rl0001_bc300_ref30_syn_rs10_r095_pen20|0.001|300|30|synthetic|10|0.95|20
A23|rl0003_bc300_ref30_syn_rs10_r095_pen20|0.003|300|30|synthetic|10|0.95|20
A24|rl001_bc300_ref100_rs5_r099_pen20|0.01|300|100|mixed|5|0.99|20
CONFIGS

log "all Stage A jobs submitted; waiting for remaining jobs"
while [[ "${ACTIVE_JOBS}" -gt 0 ]]; do
  wait -n || true
  ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  write_summary || true
done

write_summary
log "Stage A queue finished. Summary: ${SUMMARY_CSV}"
