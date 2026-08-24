#!/bin/bash
# 消融实验：只移除判别器自身的 detach 更新路径 (L_disc)
#
# 与 Stage 1 baseline (ablation=none) 的唯一区别：
#   ablation=no_detach
#   —— 跳过 CADT.py:130-143 的 disc_logits2 = self.discriminator(zsc.detach()) 分支
#   保留 confusion_loss (L_conf) + domain_class_loss (L_dom) + 回归 + 对比 loss
#
# 自动循环三种划分模式: cross_subject / cross_day / within_subject_day
# 结果保存到 /root/zjm/cadt/results_no_detach/{timestamp}_emotion_{split_mode}/
#
# 用法:
#   bash /root/zjm/cadt/script/no_detach_ablation.sh                 # 跑全部 3 种模式
#   bash /root/zjm/cadt/script/no_detach_ablation.sh cross_subject   # 只跑指定模式
#   bash /root/zjm/cadt/script/no_detach_ablation.sh "cross_subject cross_day"  # 跑指定多个

# ==================== 基础配置 ====================
# 要运行的划分模式（空格分隔，可用命令行参数覆盖）
DEFAULT_MODES="cross_subject cross_day within_subject_day"
SPLIT_MODES="${1:-${DEFAULT_MODES}}"

SPLIT_DIR="/vePFS-0x0d/DailyEEG/splits_new"
TARGET_SUBJECT_EMOTION=""    # 留空=全被试

# --- Encoder / 数据 (与 Stage 1 sweep 保持一致) ---
ENCODER_TYPE="cnn"
T_SIZE=8
X_DIM=59
SEQ_LEN=2000
DATA_DIR="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"

# --- 训练 (与 Stage 1 sweep 保持一致) ---
DEVICE=0
NUM_WORKERS=4
BATCH_SIZE=256
N_EPOCH=351
NORM_MODE="day"

# --- Loss 权重 (baseline 默认值，与 Stage 1 一致) ---
SPHERE_WEIGHT=10.0
DISC_WEIGHT=1.0
DOM_WEIGHT=1.0
TAU=1.0

# --- 消融模式 ---
ABLATION="no_detach"

# --- 结果目录 (用户指定) ---
RESULT_BASE="results_no_detach"

# ==================== 逻辑 ====================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="cadt_env"

mkdir -p /vePFS-0x0d/home/zjm/tmp
export TMPDIR=/vePFS-0x0d/home/zjm/tmp

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${CADT_ROOT}"

MODE_ARRAY=(${SPLIT_MODES})
TOTAL_RUNS=${#MODE_ARRAY[@]}

echo "========================================"
echo " No-Detach Ablation Experiment"
echo "========================================"
echo " 划分模式: ${SPLIT_MODES} (共 ${TOTAL_RUNS} 个)"
echo " 划分目录: ${SPLIT_DIR}"
echo " Encoder : ${ENCODER_TYPE}  |  t_size: ${T_SIZE}"
echo " 数据目录: ${DATA_DIR}"
echo " 输入形状假设: x_dim=${X_DIM}, seq_len=${SEQ_LEN}"
echo " 训练: epochs=${N_EPOCH}, batch=${BATCH_SIZE}, device=${DEVICE}"
echo " 标准化: ${NORM_MODE}"
echo " Loss 权重: sphere=${SPHERE_WEIGHT} disc=${DISC_WEIGHT} dom=${DOM_WEIGHT} tau=${TAU}"
echo " 消融模式: ${ABLATION} (只移除判别器自身 detach 更新路径)"
echo " 结果目录: ${CADT_ROOT}/${RESULT_BASE}/"
echo "========================================"
echo ""

# 检查某个 split_mode 是否已成功跑过 (params.txt + results.txt 都存在，且 ablation=no_detach)
# 用法: already_done <split_mode>
# 返回 0=已完成（跳过），1=未完成（需跑）
already_done() {
    local split_mode="$1"
    local base="${CADT_ROOT}/${RESULT_BASE}"
    [ -d "${base}" ] || return 1
    for d in "${base}"/*"_emotion_${split_mode}"; do
        [ -d "${d}" ] || continue
        local pm="${d}/params.txt"
        local rs="${d}/results.txt"
        [ -f "${pm}" ] && [ -f "${rs}" ] || continue
        if grep -q "^ablation: ${ABLATION}$" "${pm}"; then
            return 0
        fi
    done
    return 1
}

RUN_INDEX=0
for MODE in "${MODE_ARRAY[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    echo ">>> [${RUN_INDEX}/${TOTAL_RUNS}] split_mode=${MODE} ablation=${ABLATION}"

    if already_done "${MODE}"; then
        echo ">>> [${RUN_INDEX}/${TOTAL_RUNS}] 已跑过 (params.txt + results.txt 存在且 ablation=${ABLATION})，跳过"
        echo ""
        continue
    fi

    local_args=(
        --x_dim "${X_DIM}"
        --n_targets 11
        --seq_len "${SEQ_LEN}"
        --t_size "${T_SIZE}"
        --encoder_type "${ENCODER_TYPE}"
        --n_epoch "${N_EPOCH}"
        --batch_size "${BATCH_SIZE}"
        --device "${DEVICE}"
        --num_workers "${NUM_WORKERS}"
        --sphere_loss_weight "${SPHERE_WEIGHT}"
        --discriminator_loss_weight "${DISC_WEIGHT}"
        --domain_cls_weight "${DOM_WEIGHT}"
        --contrast_tau "${TAU}"
        --split_mode "${MODE}"
        --split_dir "${SPLIT_DIR}"
        --data_dir "${DATA_DIR}"
        --norm_mode "${NORM_MODE}"
        --ablation "${ABLATION}"
        --result_base "${RESULT_BASE}"
    )
    if [ -n "${TARGET_SUBJECT_EMOTION}" ]; then
        local_args+=(--target_subject "${TARGET_SUBJECT_EMOTION}")
    fi

    python main.py "${local_args[@]}"
    echo ">>> [${RUN_INDEX}/${TOTAL_RUNS}] split_mode=${MODE} done"
    echo ""
done

echo "========================================"
echo " 全部 ${TOTAL_RUNS} 个划分模式训练完成!"
echo " 结果目录: ${CADT_ROOT}/${RESULT_BASE}/"
echo "========================================"
