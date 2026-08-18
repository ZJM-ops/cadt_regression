#!/bin/bash
# DailyEEG 情绪回归 + CADT 域适应测试脚本（仅 Emotion Split 模式）
#
# 用法:
#   bash /root/zjm/cadt/script/dailyeeg_run.sh
#
# 修改参数: 直接编辑下方的参数设置即可
#
# Emotion split 模式:
#   - cross_subject:     按被试划分，训练/验证/测试来自不同被试
#   - cross_day:         按被试-天对划分，训练/验证/测试来自不同天
#       TARGET_SUBJECT_EMOTION: 指定被试则只使用该被试数据（留空=所有被试）
#   - within_subject_day: 在每个被试-天内按时间连续切分
#       TARGET_SUBJECT_EMOTION: 指定被试则只使用该被试数据（留空=所有被试）
#
# Encoder 类型:
#   - cnn:  使用 CNNEncoder + 波形 X.npy (默认)
#           x_dim=59, seq_len=2000
#           数据目录: /vePFS-0x0d/DailyEEG/processed_cadt_addtime_new
#   - mlp:  使用 MLPEncoder + DE 特征 X_vec.npy
#           x_dim=295, seq_len=1, data_dir 指向 DE npy 目录
#           数据目录: /vePFS-0x0d/home/zjm/de_data
#           （仅 DE 特征单独存于此；其余数据仍在 processed_cadt_addtime_new）
#
# 示例:
#   # === CNN + 波形 (cross_subject, day norm, none ablation, t_size=128) ===
#   SPLIT_MODE="cross_subject"
#   ENCODER_TYPE="cnn";  X_DIM=59;  SEQ_LEN=2000
#   DATA_DIR="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"
#   T_SIZE=128;  NORM_MODE="day";  ABLATION="none"
#
#   # === MLP + DE 特征 (cross_day, subject 1, sample norm, none ablation) ===
#   SPLIT_MODE="cross_day";  TARGET_SUBJECT_EMOTION="1"
#   ENCODER_TYPE="mlp";  X_DIM=295;  SEQ_LEN=1
#   DATA_DIR="/vePFS-0x0d/home/zjm/de_data"
#   T_SIZE=128;  NORM_MODE="sample";  ABLATION="none"

# ==================== 参数设置 ====================
# --- Emotion Split 划分 ---
SPLIT_MODE="cross_day"          # cross_subject | cross_day | within_subject_day
SPLIT_DIR="/vePFS-0x0d/DailyEEG/splits_new"

# 批量模式：循环多个被试（仅对 cross_day / within_subject_day 生效）
# 例如: SUBJECTS_TO_RUN="1 2 5 10 15"   留空=单次模式
SUBJECTS_TO_RUN=""

# 单次模式下的目标被试（仅 cross_day / within_subject_day，留空=全被试）
TARGET_SUBJECT_EMOTION=""

# --- Encoder / 数据 ---
# ENCODER_TYPE="cnn"                   # cnn | mlp
# T_SIZE=8                           # 编码器嵌入维度（结果存到 tsize{xx}/）
# X_DIM=59                             # CNN=59 通道，MLP=295 DE 特征维
# SEQ_LEN=2000                         # CNN=2000（10s @200Hz），MLP=1
# DATA_DIR="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"

ENCODER_TYPE="mlp"                    # cnn | mlp
T_SIZE=8                              # 嵌入维度（8/32/128 都行，MLP 也支持）
X_DIM=295                             # DE 特征维 = 59 通道 × 5 频段
SEQ_LEN=1                             # DE 是向量，无时间维
DATA_DIR="/vePFS-0x0d/home/zjm/de_data"   # 只 DE 特征在此处

# --- 训练 ---
DEVICE=0
NUM_WORKERS=4
BATCH_SIZE=256
N_EPOCH=351
SPHERE_WEIGHT=10.0
DISC_WEIGHT=1.0
DOM_WEIGHT=1.0

# --- 标准化 ---
# sample（逐样本）/ day（按天）/ subject（按被试）
# 使用 MLP+DE 时建议 "sample"（DE 特征提取时已做频段归一化）
NORM_MODE="sample"

# --- 消融 ---
# none=完整模型，variant=去掉域相关损失（仅回归+对比）
ABLATION="variant"
# ==================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="cadt_env"

mkdir -p /vePFS-0x0d/home/zjm/tmp
export TMPDIR=/vePFS-0x0d/home/zjm/tmp

# 根据 ENCODER_TYPE 给出默认 x_dim / seq_len 提示（不强制覆盖，尊重用户设置）
echo "========================================"
echo " DailyEEG -> CADT 回归测试 (Emotion Split)"
echo "========================================"
echo " 划分模式: ${SPLIT_MODE}"
echo " 划分目录: ${SPLIT_DIR}"
[ -n "${TARGET_SUBJECT_EMOTION}" ] && echo " 目标被试: ${TARGET_SUBJECT_EMOTION}"
echo " Encoder : ${ENCODER_TYPE}  |  t_size: ${T_SIZE}"
echo " 数据目录: ${DATA_DIR}"
echo " 输入形状假设: x_dim=${X_DIM}, seq_len=${SEQ_LEN}"
echo " 训练: epochs=${N_EPOCH}, batch=${BATCH_SIZE}, device=${DEVICE}"
echo " 标准化: ${NORM_MODE}  |  消融: ${ABLATION}"
echo "========================================"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${CADT_ROOT}"

# 构造基础命令（供单次 / 批量模式共享）
COMMON_ARGS=(
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
  --split_mode "${SPLIT_MODE}"
  --split_dir "${SPLIT_DIR}"
  --data_dir "${DATA_DIR}"
  --norm_mode "${NORM_MODE}"
  --ablation "${ABLATION}"
)

if [ -n "${SUBJECTS_TO_RUN}" ]; then
    # === 批量模式 ===
    SUBJECT_ARRAY=(${SUBJECTS_TO_RUN})
    TOTAL_RUNS=${#SUBJECT_ARRAY[@]}
    echo ""
    echo "批量运行模式: 共 ${TOTAL_RUNS} 个被试 (${SUBJECTS_TO_RUN})"
    echo ""
    RUN_INDEX=0
    for SUBJECT_ID in "${SUBJECT_ARRAY[@]}"; do
        RUN_INDEX=$((RUN_INDEX + 1))
        echo "[$RUN_INDEX/$TOTAL_RUNS] 训练 - 目标被试 ${SUBJECT_ID} ..."
        python main.py "${COMMON_ARGS[@]}" --target_subject "${SUBJECT_ID}"
        echo "[$RUN_INDEX/$TOTAL_RUNS] 被试 ${SUBJECT_ID} 完成"
        echo ""
    done
    echo "========================================"
    echo " 全部 ${TOTAL_RUNS} 个被试训练完成!"
    echo "========================================"
else
    # === 单次模式 ===
    echo "[1/1] 启动训练 ..."
    if [ -n "${TARGET_SUBJECT_EMOTION}" ]; then
        python main.py "${COMMON_ARGS[@]}" --target_subject "${TARGET_SUBJECT_EMOTION}"
    else
        python main.py "${COMMON_ARGS[@]}"
    fi
    echo "完成。"
fi
