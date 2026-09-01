#!/usr/bin/env bash
# Smoke-test every training entry point on CPU or GPU.
#
# Usage:
#   bash scripts/test_algos.sh --device cpu
#   bash scripts/test_algos.sh --device gpu
#
# CPU mode runs one minimal training update per algorithm, sequentially. The
# IPPO job creates the checkpoint consumed by the ego-training jobs. GPU mode
# preserves the larger parallel smoke tests and is the default.

set -uo pipefail

usage() {
    echo "Usage: bash scripts/test_algos.sh [--device cpu|gpu]" >&2
    exit 2
}

DEVICE="gpu"
if [[ $# -gt 0 ]]; then
    [[ "$1" == "--device" && $# -eq 2 ]] || usage
    DEVICE="$2"
fi
[[ "$DEVICE" == "cpu" || "$DEVICE" == "gpu" ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_ROOT/results/test_algos_$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$RESULTS_DIR/summary.log"
mkdir -p "$RESULTS_DIR"

COMMON_FLAGS="logger.mode=disabled logger.log_train_out=false logger.log_eval_out=false local_logger.save_train_out=false local_logger.save_eval_out=false"

if [[ "$DEVICE" == "cpu" ]]; then
    PARTNER_PATH="$RESULTS_DIR/ippo_partner/saved_train_run"
    IPPO_FLAGS="algorithm.TOTAL_TIMESTEPS=128 algorithm.NUM_ENVS=1 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1 local_logger.save_train_out=true hydra.run.dir=$RESULTS_DIR/ippo_partner"
    BRDIV_FLAGS="algorithm.TOTAL_TIMESTEPS=256 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_ENVS=2 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1"
    LBRDIV_FLAGS="$BRDIV_FLAGS"
    COMEDI_FLAGS="algorithm.TOTAL_TIMESTEPS_PER_ITERATION=1152 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_ARGMAX_ROLLOUT_EPS=1 algorithm.NUM_EVAL_EPISODES=1 algorithm.NUM_ENVS=2 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1"
    FCP_FLAGS="algorithm.TOTAL_TIMESTEPS=128 algorithm.PARTNER_POP_SIZE=1 algorithm.NUM_ENVS=1 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1"
    PPO_EGO_FLAGS="algorithm.TOTAL_TIMESTEPS=256 algorithm.NUM_ENVS=2 algorithm.NUM_EGO_TRAIN_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1 algorithm.S5_D_MODEL=16 algorithm.S5_SSM_SIZE=16 algorithm.S5_ACTOR_CRITIC_HIDDEN_DIM=64 algorithm.FC_N_LAYERS=2"
    LIAM_EGO_FLAGS="algorithm.TOTAL_TIMESTEPS=256 algorithm.NUM_ENVS=2 algorithm.NUM_EGO_TRAIN_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1 algorithm.ENCODER_HIDDEN_DIM=16 algorithm.ENCODER_OUTPUT_DIM=8 algorithm.DECODER_HIDDEN_DIM=16 algorithm.POLICY_INPUT_DIM=8"
    MELIBA_EGO_FLAGS="algorithm.TOTAL_TIMESTEPS=256 algorithm.NUM_ENVS=2 algorithm.NUM_EGO_TRAIN_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1 algorithm.ENCODER_STATE_EMBED_DIM=16 algorithm.ENCODER_ACTION_EMBED_DIM=16 algorithm.ENCODER_REWARD_EMBED_DIM=16 algorithm.ENCODER_RNN_HIDDEN_DIM=16 algorithm.ENCODER_LAYERS_BEFORE_RNN=16 algorithm.ENCODER_LAYERS_AFTER_RNN=16 algorithm.ENCODER_LATENT_DIM=16 algorithm.DECODER_STATE_EMBED_DIM=16 algorithm.DECODER_AGENT_CHARACTER_EMBED_DIM=8 algorithm.DECODER_HIDDEN_DIM=16"
    ROTATE_FLAGS="algorithm.NUM_OPEN_ENDED_ITERS=1 algorithm.TIMESTEPS_PER_ITER_PARTNER=1024 algorithm.TIMESTEPS_PER_ITER_EGO=256 algorithm.PARTNER_POP_SIZE=1 algorithm.NUM_ENVS=2 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1 algorithm.EGO_ARGS.UPDATE_EPOCHS=1 algorithm.EGO_ARGS.S5_D_MODEL=16 algorithm.EGO_ARGS.S5_SSM_SIZE=16 algorithm.EGO_ARGS.S5_ACTOR_CRITIC_HIDDEN_DIM=64 algorithm.EGO_ARGS.FC_N_LAYERS=2"
    COLE_FLAGS="algorithm.TOTAL_TIMESTEPS_PER_ITERATION=1024 algorithm.PARTNER_POP_SIZE=2 algorithm.XP_EVAL_ROLLOUT_EPS=1 algorithm.NUM_EVAL_EPISODES=1 algorithm.NUM_ENVS=2 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1 algorithm.SHAPLEY_MAX_ITER=1 algorithm.SHAPLEY_PAGERANK_ITER=2"
    TRAJEDI_FLAGS="algorithm.TOTAL_TIMESTEPS=768 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_ENVS_CONFS=2 algorithm.NUM_ENVS_BR=2 algorithm.NUM_SEEDS=1 algorithm.NUM_CHECKPOINTS=1 algorithm.UPDATE_EPOCHS=1 algorithm.NUM_MINIBATCHES=1"
    FCP_TRAIN_EGO="false"
    PPO_EGO_HELDOUT_EVAL="false"
else
    PARTNER_PATH="eval_teammates/lbf_7x7/ippo/ippo-lbf-7-levels/saved_train_run/"
    IPPO_FLAGS="algorithm.NUM_SEEDS=1"
    BRDIV_FLAGS="algorithm.TOTAL_TIMESTEPS=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1"
    LBRDIV_FLAGS="$BRDIV_FLAGS"
    COMEDI_FLAGS="algorithm.TOTAL_TIMESTEPS_PER_ITERATION=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1"
    FCP_FLAGS="algorithm.TOTAL_TIMESTEPS=1e5 algorithm.NUM_CHECKPOINTS=2 algorithm.ego_train_algorithm.TOTAL_TIMESTEPS=1e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1"
    PPO_EGO_FLAGS="algorithm.TOTAL_TIMESTEPS=1e5 algorithm.NUM_EGO_TRAIN_SEEDS=1"
    LIAM_EGO_FLAGS="$PPO_EGO_FLAGS"
    MELIBA_EGO_FLAGS="$PPO_EGO_FLAGS"
    ROTATE_FLAGS="algorithm.NUM_OPEN_ENDED_ITERS=1 algorithm.TIMESTEPS_PER_ITER_PARTNER=1e5 algorithm.TIMESTEPS_PER_ITER_EGO=1e5 algorithm.NUM_SEEDS=1"
    COLE_FLAGS="algorithm.TOTAL_TIMESTEPS_PER_ITERATION=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1"
    TRAJEDI_FLAGS="algorithm.TOTAL_TIMESTEPS=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1"
    FCP_TRAIN_EGO="true"
    PPO_EGO_HELDOUT_EVAL="true"
fi

JOB_NAMES=()
JOB_CMDS=()

JOB_NAMES+=("ippo")
JOB_CMDS+=("python marl/run.py algorithm=ippo/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_ippo $COMMON_FLAGS $IPPO_FLAGS")

JOB_NAMES+=("brdiv")
JOB_CMDS+=("python teammate_generation/run.py algorithm=brdiv/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_brdiv run_heldout_eval=false train_ego=false $COMMON_FLAGS $BRDIV_FLAGS")

JOB_NAMES+=("lbrdiv")
JOB_CMDS+=("python teammate_generation/run.py algorithm=lbrdiv/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_lbrdiv run_heldout_eval=false train_ego=false $COMMON_FLAGS $LBRDIV_FLAGS")

JOB_NAMES+=("comedi")
JOB_CMDS+=("python teammate_generation/run.py algorithm=comedi/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_comedi run_heldout_eval=false train_ego=false $COMMON_FLAGS $COMEDI_FLAGS")

JOB_NAMES+=("fcp")
JOB_CMDS+=("python teammate_generation/run.py algorithm=fcp/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_fcp run_heldout_eval=false train_ego=$FCP_TRAIN_EGO $COMMON_FLAGS $FCP_FLAGS")

JOB_NAMES+=("ppo_ego")
JOB_CMDS+=("python ego_agent_training/run.py algorithm=ppo_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_ppo_ego algorithm.partner_agent.ippo.path=$PARTNER_PATH run_heldout_eval=$PPO_EGO_HELDOUT_EVAL $COMMON_FLAGS $PPO_EGO_FLAGS")

JOB_NAMES+=("liam_ego")
JOB_CMDS+=("python ego_agent_training/run.py algorithm=liam_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_liam_ego algorithm.partner_agent.ippo.path=$PARTNER_PATH run_heldout_eval=false $COMMON_FLAGS $LIAM_EGO_FLAGS")

JOB_NAMES+=("meliba_ego")
JOB_CMDS+=("python ego_agent_training/run.py algorithm=meliba_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_meliba_ego algorithm.partner_agent.ippo.path=$PARTNER_PATH run_heldout_eval=false $COMMON_FLAGS $MELIBA_EGO_FLAGS")

JOB_NAMES+=("rotate")
JOB_CMDS+=("python open_ended_training/run.py algorithm=rotate/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_rotate run_heldout_eval=false $COMMON_FLAGS $ROTATE_FLAGS")

JOB_NAMES+=("cole")
JOB_CMDS+=("python open_ended_training/run.py algorithm=cole/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_cole run_heldout_eval=false $COMMON_FLAGS $COLE_FLAGS")

JOB_NAMES+=("trajedi")
JOB_CMDS+=("python open_ended_training/run.py algorithm=trajedi/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_trajedi run_heldout_eval=false $COMMON_FLAGS $TRAJEDI_FLAGS")

# Empty means all jobs. Keep IPPO selected when running ego-training jobs in
# CPU mode because it creates their temporary partner checkpoint.
RUN_JOBS=(
    ippo
    brdiv
    lbrdiv
    comedi
    fcp
    ppo_ego
    liam_ego
    meliba_ego
    rotate
    cole
    trajedi
)

should_run() {
    local candidate="$1"
    local selected
    [[ ${#RUN_JOBS[@]} -eq 0 ]] && return 0
    for selected in "${RUN_JOBS[@]}"; do
        [[ "$selected" == "$candidate" ]] && return 0
    done
    return 1
}

print_summary() {
    {
        echo ""
        echo "======================================================"
        echo "  RESULTS"
        echo "======================================================"
        printf "  PASSED (%d):" "${#SUCCEEDED[@]}"; printf " %s" "${SUCCEEDED[@]:-}"; echo ""
        printf "  FAILED (%d):" "${#FAILED[@]}"; printf " %s" "${FAILED[@]:-}"; echo ""
        echo "======================================================"
        echo "  Full logs: $RESULTS_DIR"
    } | tee -a "$LOG"
}

echo "Mode: $DEVICE" | tee "$LOG"
echo "Logs: $RESULTS_DIR" | tee -a "$LOG"
cd "$REPO_ROOT"

if [[ "$DEVICE" == "cpu" ]]; then
    export JAX_PLATFORMS=cpu
    export JAX_AHT_FORCE_CPU_RESTORE=1
    export MPLCONFIGDIR="$RESULTS_DIR/matplotlib"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$MPLCONFIGDIR"

    SUCCEEDED=()
    FAILED=()
    for i in "${!JOB_NAMES[@]}"; do
        name="${JOB_NAMES[$i]}"
        if should_run "$name"; then
            logfile="$RESULTS_DIR/${name}.log"
            echo "[$(date '+%H:%M:%S')] [START] $name (CPU)" | tee -a "$LOG"
            if bash -c "${JOB_CMDS[$i]}" >"$logfile" 2>&1; then
                SUCCEEDED+=("$name")
                echo "[$(date '+%H:%M:%S')] [OK]    $name (CPU)" | tee -a "$LOG"
            else
                FAILED+=("$name")
                echo "[$(date '+%H:%M:%S')] [FAIL]  $name (CPU), see ${name}.log" | tee -a "$LOG"
            fi
        else
            echo "[SKIP]  $name" | tee -a "$LOG"
        fi
    done

    print_summary
    [[ ${#FAILED[@]} -eq 0 ]] && exit 0 || exit 1
fi

mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{ gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 >= 20000) print $1 }'
)

if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "ERROR: no GPUs with at least 20 GB free memory found." | tee -a "$LOG" >&2
    exit 1
fi
echo "GPUs with at least 20 GB free: ${GPUS[*]}" | tee -a "$LOG"

declare -a FREE_GPUS=("${GPUS[@]}")
declare -A PID_GPU
declare -A PID_NAME
SUCCEEDED=()
FAILED=()

reap() {
    local -a done_pids=()
    local pid name gpu
    for pid in "${!PID_GPU[@]}"; do
        kill -0 "$pid" 2>/dev/null || done_pids+=("$pid")
    done
    for pid in "${done_pids[@]}"; do
        name="${PID_NAME[$pid]}"
        gpu="${PID_GPU[$pid]}"
        if wait "$pid"; then
            SUCCEEDED+=("$name")
            echo "[$(date '+%H:%M:%S')] [OK]    $name (GPU $gpu)" | tee -a "$LOG"
        else
            FAILED+=("$name")
            echo "[$(date '+%H:%M:%S')] [FAIL]  $name (GPU $gpu), see ${name}.log" | tee -a "$LOG"
        fi
        FREE_GPUS+=("$gpu")
        unset 'PID_GPU[$pid]'
        unset 'PID_NAME[$pid]'
    done
}

acquire_gpu() {
    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        reap
        [[ ${#FREE_GPUS[@]} -eq 0 ]] && sleep 2
    done
    FREE_GPU="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
}

launch() {
    local name="$1"
    local gpu logfile pid
    shift
    acquire_gpu
    gpu="$FREE_GPU"
    logfile="$RESULTS_DIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] [START] $name (GPU $gpu)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$logfile" 2>&1 &
    pid=$!
    PID_GPU[$pid]="$gpu"
    PID_NAME[$pid]="$name"
}

for i in "${!JOB_NAMES[@]}"; do
    name="${JOB_NAMES[$i]}"
    if should_run "$name"; then
        launch "$name" bash -c "${JOB_CMDS[$i]}"
    else
        echo "[SKIP]  $name" | tee -a "$LOG"
    fi
done

while [[ ${#PID_GPU[@]} -gt 0 ]]; do
    reap
    [[ ${#PID_GPU[@]} -gt 0 ]] && sleep 2
done

print_summary
[[ ${#FAILED[@]} -eq 0 ]] && exit 0 || exit 1
