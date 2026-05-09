#!/usr/bin/env bash
# V5 Overnight Master — runs 4 experiment phases sequentially.
#
# Phase A: Architecture ablation (Flat / 2-Stage T / 2-Stage M / 3-Stage Full)
# Phase B: Stage 3 extended Optuna (60 trial, wider search)
# Phase C: Stage 3 k-threshold ablation (k = 0.4 / 0.5 / 0.7 / 1.0)
# Phase D: ZigZag config dist sweep (16 configs, fast)
#
# Time budget: ~7-10 hours. Phase D is fast (~5 min). Phase A is heaviest.
# Each phase logs to its own file under logs/overnight/.
# If a phase fails, the next phase still runs (we want partial results).

set -u  # Don't set -e: we want to continue even on phase failures.

PROJECT_ROOT="/Users/yurutkenozgun/Projects/hierarchical-trading-signal-classifier"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs/overnight"
mkdir -p "$LOG_DIR"

START_TS=$(date +%s)
START_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")
MASTER_LOG="${LOG_DIR}/master_${START_TS}.log"

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg" | tee -a "$MASTER_LOG"
}

run_phase() {
    local name="$1"
    local script="$2"
    local phase_log="${LOG_DIR}/${name}_${START_TS}.log"

    log "=== START $name ==="
    log "  script: $script"
    log "  log:    $phase_log"
    local t0=$(date +%s)

    cd "$PROJECT_ROOT"
    "$PYTHON" "$script" >"$phase_log" 2>&1
    local rc=$?

    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    local elapsed_min=$((elapsed / 60))

    if [[ $rc -eq 0 ]]; then
        log "=== DONE  $name  (${elapsed_min} min, exit 0) ==="
    else
        log "=== FAIL  $name  (${elapsed_min} min, exit $rc) — continuing to next phase ==="
        # Tail the failure for master log visibility
        tail -30 "$phase_log" | sed 's/^/  /' | tee -a "$MASTER_LOG"
    fi
}

log "=================================================="
log "V5 Overnight Master started at $START_HUMAN"
log "Project: $PROJECT_ROOT"
log "Master log: $MASTER_LOG"
log "Phase logs: ${LOG_DIR}/"
log "=================================================="

# Quick fast phase first (D) so even short runs leave something
run_phase "phase_d_zigzag_extended"   "$PROJECT_ROOT/scripts/v5_overnight_phase_d_zigzag_extended.py"

# Phase A — biggest job
run_phase "phase_a_arch_ablation"     "$PROJECT_ROOT/scripts/v5_overnight_phase_a_arch_ablation.py"

# Phase C — medium job, no Optuna so fast
run_phase "phase_c_k_ablation"        "$PROJECT_ROOT/scripts/v5_overnight_phase_c_k_ablation.py"

# Phase B — medium-heavy, double trial budget
run_phase "phase_b_extended_optuna"   "$PROJECT_ROOT/scripts/v5_overnight_phase_b_extended_optuna.py"

END_TS=$(date +%s)
TOTAL=$((END_TS - START_TS))
TOTAL_HOURS=$(echo "scale=2; $TOTAL/3600" | bc)
END_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")

log "=================================================="
log "V5 Overnight Master COMPLETE at $END_HUMAN"
log "Total elapsed: ${TOTAL}s (~${TOTAL_HOURS} hours)"
log ""
log "Output locations:"
log "  reports/Phase3.6_zigzag_extended/      (Phase D — fast)"
log "  reports/Phase5.1_arch_ablation/        (Phase A — main ablation)"
log "  reports/Phase4.6_k_ablation/           (Phase C — k threshold)"
log "  reports/Phase5.2_extended_optuna/      (Phase B — wider HP search)"
log ""
log "data/processed/ contains new tuned OOF files for ablation variants"
log "=================================================="
