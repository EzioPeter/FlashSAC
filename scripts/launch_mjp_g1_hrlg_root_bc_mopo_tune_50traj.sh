#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_JOBS="${MAX_JOBS:-2}"
RUN_STAGE="${RUN_STAGE:-A}"

DATASET="outputs/mjp_g1_hrlg_seed0_0522_232057_step97658_buffer_100x1000_observable_reward.npz"
BC_CHECKPOINT="models/mjp_g1_hrlg_bc/seed0_0522_232057_step97658_50traj_200ep_noeval/final"
DYNAMICS_CHECKPOINT="models/mjp_g1_hrlg_bc_mopo/root_world_model_seed0_0522_232057_step97658_50traj_observable/best_model"
SUMMARY_CSV="outputs/mjp_g1_hrlg_bc_mopo_tune50_summary.csv"
CONFIG_CSV="outputs/mjp_g1_hrlg_bc_mopo_tune50_configs.csv"
QUEUE_LOG_DIR="models/mjp_g1_hrlg_bc_mopo/queue_logs"
mkdir -p "$QUEUE_LOG_DIR" outputs
QUEUE_LOG="$QUEUE_LOG_DIR/tune50_stage_${RUN_STAGE}_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "Missing required path: $path"
    exit 1
  fi
}

require_file "$DATASET"
require_file "$BC_CHECKPOINT/actor.pt"
require_file "$DYNAMICS_CHECKPOINT/dynamics_ensemble.pt"

COMMON_ARGS=(
  --dataset "$DATASET"
  --reward-source observable
  --limit-trajectories 50
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT"
  --bc-checkpoint "$BC_CHECKPOINT"
  --predict-root
  --formula-reward
  --formula-done
  --no-predict-reward-head
  --no-predict-done-head
  --skip-dynamics-training
  --rollout-horizon 1
  --no-actor-weight-normalization
  --online-eval-episodes 50
  --online-eval-num-envs 50
  --online-eval-max-episode-steps 1000
  --save-interval 5
  --device cuda:0
  --online-eval-device cuda:0
  --seed 0
  --batch-size 1024
  --model-batch-size 1024
  --rollout-batch-size 4096
  --dynamics-ensemble-size 5
  --dynamics-hidden-dim 256
  --dynamics-layers 3
)

STAGE_A_ARGS=(
  --epochs 20
  --online-eval-interval 5
  --actor-lr 1e-5
  --critic-lr 3e-4
  --temperature-lr 3e-4
  --actor-update-period 4
  --critic-warmup-epochs 5
)

write_config_header() {
  cat >"$CONFIG_CSV" <<'EOF'
name,output_dir,actor_rl_coef,bc_alpha,bc_ref_alpha,bc_ref_on,reward_scale,critic_min_v,critic_max_v,real_ratio,mopo_penalty_coef,critic_warmup_epochs,epochs
EOF
}

append_config() {
  local name="$1"
  local output_dir="$2"
  local actor_rl_coef="$3"
  local bc_alpha="$4"
  local bc_ref_alpha="$5"
  local bc_ref_on="$6"
  local reward_scale="$7"
  local critic_min_v="$8"
  local critic_max_v="$9"
  local real_ratio="${10}"
  local mopo_penalty_coef="${11}"
  local critic_warmup_epochs="${12}"
  local epochs="${13}"
  echo "$name,$output_dir,$actor_rl_coef,$bc_alpha,$bc_ref_alpha,$bc_ref_on,$reward_scale,$critic_min_v,$critic_max_v,$real_ratio,$mopo_penalty_coef,$critic_warmup_epochs,$epochs" >>"$CONFIG_CSV"
}

summarize_results() {
  python_bin=".venv/bin/python"
  "$python_bin" - "$CONFIG_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import math
import sys
from pathlib import Path

config_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])

def as_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan

with config_csv.open(newline="") as f:
    configs = list(csv.DictReader(f))

fieldnames = [
    "name",
    "output_dir",
    "actor_rl_coef",
    "bc_alpha",
    "bc_ref_alpha",
    "reward_scale",
    "real_ratio",
    "mopo_penalty_coef",
    "critic_warmup_epochs",
    "best_epoch",
    "best_return",
    "best_length",
    "best_survived_full",
    "final_return",
    "final_length",
    "final_survived_full",
    "final_offline_bc_mse_real",
    "min_offline_bc_mse_real",
    "max_offline_bc_mse_real",
    "status",
]

rows = []
for cfg in configs:
    out = Path(cfg["output_dir"])
    online_path = out / "online_eval.csv"
    progress_path = out / "progress.csv"
    exit_path = out / "exit_code"
    status = "pending"
    if exit_path.exists():
        status = f"exit={exit_path.read_text().strip()}"
    elif progress_path.exists():
        status = "running"

    online_rows = []
    if online_path.exists():
        with online_path.open(newline="") as f:
            online_rows = list(csv.DictReader(f))
    progress_rows = []
    if progress_path.exists():
        with progress_path.open(newline="") as f:
            progress_rows = list(csv.DictReader(f))

    best = None
    final = None
    if online_rows:
        best = max(online_rows, key=lambda row: as_float(row.get("avg_return")))
        final = online_rows[-1]
    offline_vals = [as_float(row.get("offline_bc_mse_real")) for row in progress_rows]
    offline_vals = [value for value in offline_vals if not math.isnan(value)]

    rows.append(
        {
            "name": cfg["name"],
            "output_dir": cfg["output_dir"],
            "actor_rl_coef": cfg["actor_rl_coef"],
            "bc_alpha": cfg["bc_alpha"],
            "bc_ref_alpha": cfg["bc_ref_alpha"],
            "reward_scale": cfg["reward_scale"],
            "real_ratio": cfg["real_ratio"],
            "mopo_penalty_coef": cfg["mopo_penalty_coef"],
            "critic_warmup_epochs": cfg["critic_warmup_epochs"],
            "best_epoch": "" if best is None else best.get("epoch", ""),
            "best_return": "" if best is None else best.get("avg_return", ""),
            "best_length": "" if best is None else best.get("avg_length", ""),
            "best_survived_full": "" if best is None else best.get("survived_full", ""),
            "final_return": "" if final is None else final.get("avg_return", ""),
            "final_length": "" if final is None else final.get("avg_length", ""),
            "final_survived_full": "" if final is None else final.get("survived_full", ""),
            "final_offline_bc_mse_real": "" if not progress_rows else progress_rows[-1].get("offline_bc_mse_real", ""),
            "min_offline_bc_mse_real": "" if not offline_vals else min(offline_vals),
            "max_offline_bc_mse_real": "" if not offline_vals else max(offline_vals),
            "status": status,
        }
    )

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(summary_csv)
PY
}

SEM="$(mktemp -u)"
mkfifo "$SEM"
exec 9<>"$SEM"
rm "$SEM"
for _ in $(seq 1 "$MAX_JOBS"); do
  printf "." >&9
done

run_limited() {
  local name="$1"
  local out_dir="$2"
  shift 2
  read -r -n 1 _ <&9
  mkdir -p "$out_dir"
  (
    set +e
    log "START name=$name out=$out_dir"
    "$@" >"$out_dir/train_stdout.log" 2>&1 &
    local cmd_pid=$!
    echo "$cmd_pid" >"$out_dir/pid"
    echo "$(date '+%F %T')" >"$out_dir/start_time.txt"
    wait "$cmd_pid"
    local status=$?
    echo "$status" >"$out_dir/exit_code"
    echo "$(date '+%F %T')" >"$out_dir/end_time.txt"
    if [[ "$status" -eq 0 ]]; then
      log "DONE name=$name status=$status out=$out_dir"
    else
      log "FAIL name=$name status=$status out=$out_dir log=$out_dir/train_stdout.log"
    fi
    summarize_results >>"$QUEUE_LOG" 2>&1
    printf "." >&9
    exit "$status"
  ) &
}

launch_stage_a() {
  write_config_header

  append_config A1 models/mjp_g1_hrlg_bc_mopo/tune50_a1_rl001_bc100_ref10_rs10_pen5 0.01 100 10 mixed 10 -5 20 0.95 5 5 20
  append_config A2 models/mjp_g1_hrlg_bc_mopo/tune50_a2_rl005_bc100_ref10_rs10_pen5 0.05 100 10 mixed 10 -5 20 0.95 5 5 20
  append_config A3 models/mjp_g1_hrlg_bc_mopo/tune50_a3_rl001_bc300_ref30_rs10_pen5 0.01 300 30 mixed 10 -5 20 0.95 5 5 20
  append_config A4 models/mjp_g1_hrlg_bc_mopo/tune50_a4_rl001_bc100_ref30_rs5_pen10 0.01 100 30 mixed 5 -5 20 0.95 10 5 20
  append_config A5 models/mjp_g1_hrlg_bc_mopo/tune50_a5_rl001_bc100_ref10_rs10_real1 0.01 100 10 mixed 10 -5 20 1.0 5 5 20
  append_config A6 models/mjp_g1_hrlg_bc_mopo/tune50_a6_rl000_bc100_ref30_rs10_pen5 0.0 100 30 mixed 10 -5 20 0.95 5 5 20

  while IFS=, read -r name out actor_rl bc_alpha bc_ref bc_ref_on reward_scale critic_min critic_max real_ratio penalty warmup epochs; do
    [[ "$name" == "name" ]] && continue
    run_limited "$name" "$out" \
      uv run python scripts/train_mjp_g1_hrlg_bc_mopo.py \
        "${COMMON_ARGS[@]}" \
        "${STAGE_A_ARGS[@]}" \
        --output-dir "$out" \
        --actor-rl-coef "$actor_rl" \
        --bc-alpha "$bc_alpha" \
        --bc-ref-alpha "$bc_ref" \
        --bc-ref-on "$bc_ref_on" \
        --reward-scale "$reward_scale" \
        --critic-min-v "$critic_min" \
        --critic-max-v "$critic_max" \
        --real-ratio "$real_ratio" \
        --mopo-penalty-coef "$penalty" \
        --critic-warmup-epochs "$warmup" \
        --epochs "$epochs"
  done <"$CONFIG_CSV"
}

log "Tune50 queue started RUN_STAGE=$RUN_STAGE MAX_JOBS=$MAX_JOBS"
log "Dataset=$DATASET"
log "BC checkpoint=$BC_CHECKPOINT"
log "Dynamics checkpoint=$DYNAMICS_CHECKPOINT"
log "No 100traj BC checkpoint is referenced by this queue."

case "$RUN_STAGE" in
  A)
    launch_stage_a
    ;;
  *)
    log "Unsupported RUN_STAGE=$RUN_STAGE. Currently implemented: A"
    exit 1
    ;;
esac

status=0
wait || status=1
summarize_results >>"$QUEUE_LOG" 2>&1 || status=1
log "Tune50 queue finished status=$status summary=$SUMMARY_CSV"
exit "$status"
