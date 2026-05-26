#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "[%s] launcher failed at line %s with exit_code=%s\n" "$(date "+%F %T")" "$LINENO" "$code" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MAX_JOBS="${MAX_JOBS:-2}"
SMOKE="${SMOKE:-1}"
RUN_STAGE="${RUN_STAGE:-ALL}"
FORCE="${FORCE:-0}"

TRAIN_SCRIPT="scripts/train_mjp_g1_hrlg_ages_mopo.py"
SUMMARY_SCRIPT="scripts/summarize_mjp_g1_hrlg_ages_tune.py"
ROOT_DIR="models/mjp_g1_hrlg_ages_mopo"
SUMMARY_CSV="outputs/mjp_g1_hrlg_ages_tune50_step120000_summary.csv"
DATASET="outputs/mjp_g1_hrlg_seed0_0523_202023_step120000_dr_warmstart_buffer_100x1000_observable_reward.npz"
BC_CKPT="models/mjp_g1_hrlg_bc/seed0_0523_202023_step120000_dr_warmstart_50traj_100ksteps_noeval/final"
DYN_CKPT="models/mjp_g1_hrlg_bc_mopo/root_world_model_seed0_0523_202023_step120000_dr_warmstart_50traj_observable/best_model"

mkdir -p outputs "$ROOT_DIR"

COMMON_ARGS="\
--dataset $DATASET \
--reward-source observable \
--limit-trajectories 50 \
--dynamics-checkpoint $DYN_CKPT \
--bc-checkpoint $BC_CKPT \
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
--device cuda:0 \
--online-eval-device cuda:0 \
--seed 0 \
--online-eval-max-episode-steps 1000 \
--save-interval 10"

STAGE_A_ARGS="\
--epochs 20 \
--online-eval-interval 5 \
--online-eval-episodes 10 \
--online-eval-num-envs 10 \
--ages-generation-interval 1 \
--ages-min-buffer-size 512 \
--ages-max-buffer-size 200000 \
--ages-buffer-retain-epochs 5 \
--ages-baseline same_state_mean \
--ages-select-mode global_top_ratio \
--ages-score-use-uncertainty \
--ages-score-use-mopo-penalty \
--ages-inject-rollout-length 1"

STAGE_B_ARGS="\
--epochs 80 \
--online-eval-interval 10 \
--online-eval-episodes 20 \
--online-eval-num-envs 20 \
--ages-generation-interval 1 \
--ages-min-buffer-size 512 \
--ages-max-buffer-size 200000 \
--ages-buffer-retain-epochs 5 \
--ages-baseline same_state_mean \
--ages-select-mode global_top_ratio \
--ages-score-use-uncertainty \
--ages-score-use-mopo-penalty \
--ages-inject-rollout-length 1"

STAGE_C_ARGS="\
--epochs 200 \
--online-eval-interval 20 \
--online-eval-episodes 50 \
--online-eval-num-envs 50 \
--ages-generation-interval 1 \
--ages-min-buffer-size 512 \
--ages-max-buffer-size 200000 \
--ages-buffer-retain-epochs 5 \
--ages-baseline same_state_mean \
--ages-select-mode global_top_ratio \
--ages-score-use-uncertainty \
--ages-score-use-mopo-penalty \
--ages-inject-rollout-length 1"

declare -A PARAMS
declare -A OUTDIRS

PARAMS[A00]="--no-use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 10 --reward-scale 5"
OUTDIRS[A00]="$ROOT_DIR/tune50_step120000_A00_bcmopo_a11_real050_control"
PARAMS[A01]="--no-use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 20 --reward-scale 10"
OUTDIRS[A01]="$ROOT_DIR/tune50_step120000_A01_bcmopo_a22_real050_control"
PARAMS[A02]="--use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 10 --reward-scale 5 --ages-action-temperature 1.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.10 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A02]="$ROOT_DIR/tune50_step120000_A02_ages_real050_h10_temp1_sel010"
PARAMS[A03]="--use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 20 --reward-scale 5 --ages-action-temperature 1.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.05 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 20"
OUTDIRS[A03]="$ROOT_DIR/tune50_step120000_A03_ages_real050_h20_temp1_sel005_pen20"
PARAMS[A04]="--use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 20 --reward-scale 5 --ages-action-temperature 3.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.10 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A04]="$ROOT_DIR/tune50_step120000_A04_ages_real050_h10_temp3_sel010_pen20"
PARAMS[A05]="--use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 30 --reward-scale 5 --ages-action-temperature 3.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.05 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 20"
OUTDIRS[A05]="$ROOT_DIR/tune50_step120000_A05_ages_real050_h20_temp3_sel005_pen30"
PARAMS[A06]="--use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 30 --reward-scale 5 --ages-action-temperature 5.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.05 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A06]="$ROOT_DIR/tune50_step120000_A06_ages_real050_h10_temp5_sel005_pen30"
PARAMS[A07]="--use-ages --actor-rl-coef 0.003 --bc-alpha 100 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 40 --reward-scale 5 --ages-action-temperature 5.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.03 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 20"
OUTDIRS[A07]="$ROOT_DIR/tune50_step120000_A07_ages_real050_h20_temp5_sel003_pen40"
PARAMS[A08]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 20 --reward-scale 10 --ages-action-temperature 1.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.10 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A08]="$ROOT_DIR/tune50_step120000_A08_ages_stronganchor_real050_h10_temp1_sel010"
PARAMS[A09]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 30 --reward-scale 10 --ages-action-temperature 1.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.05 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 20"
OUTDIRS[A09]="$ROOT_DIR/tune50_step120000_A09_ages_stronganchor_real050_h20_temp1_sel005_pen30"
PARAMS[A10]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 30 --reward-scale 10 --ages-action-temperature 3.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.10 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A10]="$ROOT_DIR/tune50_step120000_A10_ages_stronganchor_real050_h10_temp3_sel010_pen30"
PARAMS[A11]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 40 --reward-scale 10 --ages-action-temperature 3.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.05 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 20"
OUTDIRS[A11]="$ROOT_DIR/tune50_step120000_A11_ages_stronganchor_real050_h20_temp3_sel005_pen40"
PARAMS[A12]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.75 --mopo-penalty-coef 30 --reward-scale 10 --ages-action-temperature 5.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.05 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A12]="$ROOT_DIR/tune50_step120000_A12_ages_real075_h10_temp5_sel005_pen30"
PARAMS[A13]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.75 --mopo-penalty-coef 40 --reward-scale 10 --ages-action-temperature 5.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.03 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 20"
OUTDIRS[A13]="$ROOT_DIR/tune50_step120000_A13_ages_real075_h20_temp5_sel003_pen40"
PARAMS[A14]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 50 --reward-scale 10 --ages-action-temperature 7.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.03 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A14]="$ROOT_DIR/tune50_step120000_A14_ages_stronganchor_real050_h10_temp7_sel003_pen50"
PARAMS[A15]="--use-ages --actor-rl-coef 0.001 --bc-alpha 300 --bc-ref-alpha 30 --bc-ref-on synthetic --real-ratio 0.50 --mopo-penalty-coef 60 --reward-scale 10 --ages-action-temperature 9.0 --ages-sim-ratio 1.0 --ages-select-ratio 0.02 --ages-num-dataset-states 1024 --ages-trajectories-per-state 4 --ages-sim-rollout-length 10"
OUTDIRS[A15]="$ROOT_DIR/tune50_step120000_A15_ages_stronganchor_real050_h10_temp9_sel002_pen60"

RUNNING_PIDS=()

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "missing required path: $1" >&2
    exit 1
  fi
}

gpu_python_count() {
  # The queue itself enforces MAX_JOBS.  Keep this guard side-effect free so
  # nohup runs cannot be killed by shell errexit/pipefail interactions.
  echo 0
}

prune_running() {
  local kept=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kept+=("$pid")
    fi
  done
  RUNNING_PIDS=("${kept[@]}")
}

wait_for_internal_slot() {
  while true; do
    prune_running
    if (( ${#RUNNING_PIDS[@]} < MAX_JOBS )); then
      return
    fi
    wait -n || true
  done
}

wait_for_gpu_slot() {
  while true; do
    local count
    count="$(gpu_python_count)"
    if (( count < MAX_JOBS )); then
      return
    fi
    log "GPU already has $count python processes, waiting for a free slot"
    sleep 60
  done
}

wait_for_all() {
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" || true
  done
  RUNNING_PIDS=()
}

run_summary() {
  uv run python "$SUMMARY_SCRIPT" --root "$ROOT_DIR" --output "$SUMMARY_CSV"
}

launch_job() {
  local output_dir="$1"
  local cmd="$2"
  mkdir -p "$output_dir"
  if [[ "$FORCE" != "1" && -f "$output_dir/exit_code" && "$(cat "$output_dir/exit_code")" == "0" ]]; then
    log "skip completed $output_dir"
    return
  fi
  wait_for_internal_slot
  wait_for_gpu_slot
  log "launch $output_dir"
  printf '%s\n' "$cmd" > "$output_dir/command.txt"
  (
    date '+%F %T' > "$output_dir/started_at.txt"
    set +e
    bash -lc "$cmd" > "$output_dir/train_stdout.log" 2>&1
    code=$?
    echo "$code" > "$output_dir/exit_code"
    date '+%F %T' > "$output_dir/finished_at.txt"
    exit "$code"
  ) &
  local pid=$!
  echo "$pid" > "$output_dir/job.pid"
  RUNNING_PIDS+=("$pid")
}

run_config() {
  local id="$1"
  local stage_args="$2"
  local output_dir="$3"
  local cmd="uv run python $TRAIN_SCRIPT $COMMON_ARGS $stage_args ${PARAMS[$id]} --output-dir $output_dir"
  launch_job "$output_dir" "$cmd"
}

select_stage_a_ids() {
  uv run python - "$SUMMARY_CSV" <<'PY'
import csv, math, re, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
rows = [r for r in rows if r.get("stage") == "A"]
def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")
def ok(row):
    if row.get("status") not in ("passed", "completed"):
        return False
    if str(row.get("exit_code")) not in ("0", "0.0"):
        return False
    if f(row, "best_length") < 500:
        return False
    if f(row, "final_offline_bc_mse_real") > 0.03:
        return False
    ref = f(row, "final_offline_ref_mse")
    if math.isfinite(ref) and ref > 0.03:
        return False
    return True
valid = [r for r in rows if ok(r)] or rows
valid.sort(key=lambda r: f(r, "score"), reverse=True)
ids = []
for row in valid:
    m = re.search(r"(A\d{2})", row.get("run_id", ""))
    if m and m.group(1) not in ids:
        ids.append(m.group(1))
    if len(ids) >= 4:
        break
print(" ".join(ids))
PY
}

select_stage_b_ids() {
  uv run python - "$SUMMARY_CSV" <<'PY'
import csv, math, re, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
rows = [r for r in rows if r.get("stage") == "B"]
def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")
def ok(row):
    if row.get("status") not in ("passed", "completed"):
        return False
    if str(row.get("exit_code")) not in ("0", "0.0"):
        return False
    best_len = f(row, "best_length")
    final_len = f(row, "final_length")
    if math.isfinite(best_len) and math.isfinite(final_len) and final_len < 0.8 * best_len:
        return False
    if f(row, "final_offline_bc_mse_real") > 0.05:
        return False
    return True
valid = [r for r in rows if ok(r)] or rows
valid.sort(key=lambda r: (f(r, "best_return"), f(r, "best_length"), f(r, "best_survived_full")), reverse=True)
ids = []
for row in valid:
    m = re.search(r"(A\d{2})", row.get("run_id", ""))
    if m and m.group(1) not in ids:
        ids.append(m.group(1))
    if len(ids) >= 2:
        break
print(" ".join(ids))
PY
}

run_smoke() {
  log "running smoke checks"
  bash -n "$0"
  uv run python -m py_compile "$TRAIN_SCRIPT" "$SUMMARY_SCRIPT"
  uv run python "$TRAIN_SCRIPT" --help >/tmp/mjp_g1_hrlg_ages_help.txt
  local smoke_dir="$ROOT_DIR/smoke_$(date '+%m%d_%H%M%S')"
  local smoke_args="\
--dataset $DATASET \
--reward-source observable \
--limit-trajectories 3 \
--allow-dynamics-split-mismatch \
--dynamics-checkpoint $DYN_CKPT \
--bc-checkpoint $BC_CKPT \
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
--critic-warmup-epochs 0 \
--actor-update-period 1 \
--no-actor-weight-normalization \
--device cuda:0 \
--online-eval-device cuda:0 \
--seed 0 \
--epochs 1 \
--max-train-steps 2 \
--updates-per-epoch 2 \
--online-eval-interval 1 \
--online-eval-episodes 2 \
--online-eval-num-envs 2 \
--online-eval-max-episode-steps 20 \
--save-interval 1 \
--use-ages \
--actor-rl-coef 0.001 \
--bc-alpha 10 \
--bc-ref-alpha 1 \
--bc-ref-on synthetic \
--real-ratio 0.5 \
--mopo-penalty-coef 1 \
--reward-scale 1 \
--ages-action-temperature 1 \
--ages-sim-ratio 1 \
--ages-select-ratio 0.5 \
--ages-num-dataset-states 8 \
--ages-trajectories-per-state 2 \
--ages-sim-rollout-length 1 \
--ages-inject-rollout-length 1 \
--ages-generation-interval 1 \
--ages-min-buffer-size 1 \
--ages-max-buffer-size 1024 \
--ages-score-use-uncertainty \
--ages-score-use-mopo-penalty \
--output-dir $smoke_dir"
  uv run python "$TRAIN_SCRIPT" $smoke_args > "$smoke_dir.stdout" 2>&1
  uv run python "$SUMMARY_SCRIPT" --root "$ROOT_DIR" --output "$SUMMARY_CSV"
  log "smoke passed: $smoke_dir"
}

require_path "$DATASET"
require_path "$BC_CKPT/actor.pt"
require_path "$DYN_CKPT/dynamics_ensemble.pt"

if [[ "$SMOKE" == "1" ]]; then
  run_smoke
  exit 0
fi

if [[ "$RUN_STAGE" == "A" || "$RUN_STAGE" == "ALL" ]]; then
  log "starting Stage A"
  for id in A00 A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 A14 A15; do
    run_config "$id" "$STAGE_A_ARGS" "${OUTDIRS[$id]}"
  done
  wait_for_all
  run_summary
fi

if [[ "$RUN_STAGE" == "B" || "$RUN_STAGE" == "ALL" ]]; then
  run_summary
  read -r -a top_a_ids <<< "$(select_stage_a_ids)"
  log "Stage B selected: ${top_a_ids[*]:-none}"
  for id in "${top_a_ids[@]:-}"; do
    [[ -n "$id" ]] || continue
    run_config "$id" "$STAGE_B_ARGS" "$ROOT_DIR/tune50_step120000_stageB_${id}"
  done
  wait_for_all
  run_summary
fi

if [[ "$RUN_STAGE" == "C" || "$RUN_STAGE" == "ALL" ]]; then
  run_summary
  read -r -a top_b_ids <<< "$(select_stage_b_ids)"
  log "Stage C selected: ${top_b_ids[*]:-none}"
  for id in "${top_b_ids[@]:-}"; do
    [[ -n "$id" ]] || continue
    run_config "$id" "$STAGE_C_ARGS" "$ROOT_DIR/tune50_step120000_stageC_${id}"
  done
  wait_for_all
  run_summary
fi

log "AGES tune queue finished"
