#!/bin/bash
# DailyEEG 情绪回归 + CADT 域适应测试脚本
#
# 用法:
#   bash /root/cadt/script/dailyeeg_run.sh
#
# 修改参数: 直接编辑下方的参数设置即可
# 结果保存: 每次训练结果自动保存到 /root/zjm/cadt/results3/ 目录

# ==================== 参数设置 ====================
SOURCE_DOMAIN='6 23 30 35'       # 源域 ID 
TARGET_DOMAIN=26       # 目标域 ID 
DEVICE=0                # GPU 编号
DATA_DIR="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"  # NPY 数据目录
OUTPUT_DIR="/root/zjm/cadt/results3"  # 结果保存目录
SKIP_ALIGN=1            # 是否跳过数据对齐 (1=跳过, 0=执行)

SEQ_LEN=2000            # 时序长度 (2000=10秒@200Hz)
BATCH_SIZE=128           # 批大小 (2000点数据建议调小)
N_EPOCH=351             # 训练轮数
# ==================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_CODE_DIR="/root/cadt_test_code"
H5_DIR="${TEST_CODE_DIR}/de_output"
CONDA_ENV="cadt_env"

echo "========================================"
echo " DailyEEG -> CADT 回归测试"
echo "========================================"
echo " 源域: ${SOURCE_DOMAIN}, 目标域: ${TARGET_DOMAIN}"
echo " 时序长度: ${SEQ_LEN}, 批大小: ${BATCH_SIZE}, 训练轮数: ${N_EPOCH}"
echo "========================================"

# 计算源域数量
NUM_SOURCES=$(echo "${SOURCE_DOMAIN}" | wc -w)
PER_DOMAIN_BATCH=$((BATCH_SIZE / NUM_SOURCES))
ACTUAL_BATCH=$((PER_DOMAIN_BATCH * NUM_SOURCES))

if [ "${PER_DOMAIN_BATCH}" -eq 0 ]; then
    echo "ERROR: batch_size=${BATCH_SIZE} 小于源域数量=${NUM_SOURCES}"
    echo "请增大batch_size或减少源域数量"
    exit 1
fi

if [ "${ACTUAL_BATCH}" -ne "${BATCH_SIZE}" ]; then
    echo "WARNING: batch_size=${BATCH_SIZE} 不能被源域数量=${NUM_SOURCES}整除"
    echo "实际batch_size=${ACTUAL_BATCH} (每域 ${PER_DOMAIN_BATCH} 个样本)"
else
    echo "INFO: 每域分配 ${PER_DOMAIN_BATCH} 个样本，实际batch_size=${ACTUAL_BATCH}"
fi

# 计算源域数量
NUM_SOURCES=$(echo "${SOURCE_DOMAIN}" | wc -w)
PER_DOMAIN_BATCH=$((BATCH_SIZE / NUM_SOURCES))
ACTUAL_BATCH=$((PER_DOMAIN_BATCH * NUM_SOURCES))

if [ "${PER_DOMAIN_BATCH}" -eq 0 ]; then
    echo "ERROR: batch_size=${BATCH_SIZE} 小于源域数量=${NUM_SOURCES}"
    echo "请增大batch_size或减少源域数量"
    exit 1
fi

if [ "${ACTUAL_BATCH}" -ne "${BATCH_SIZE}" ]; then
    echo "WARNING: batch_size=${BATCH_SIZE} 不能被源域数量=${NUM_SOURCES}整除"
    echo "实际batch_size=${ACTUAL_BATCH} (每域 ${PER_DOMAIN_BATCH} 个样本)"
else
    echo "INFO: 每域分配 ${PER_DOMAIN_BATCH} 个样本，实际batch_size=${ACTUAL_BATCH}"
fi

if [ "${SKIP_ALIGN}" != "1" ]; then
  echo "[1/2] 对齐 H5 数据到 CADT 输入格式 ..."
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
  python "${TEST_CODE_DIR}/align_cadt_input.py" \
    --input-dir "${H5_DIR}" \
    --output-dir "${DATA_DIR}" \
    --seq-len "${SEQ_LEN}" \
    --n-channels 59 \
    --n-labels 11 \
    --use-all-segments
else
  echo "[1/2] 跳过数据对齐"
fi

echo "[2/2] 启动 CADT 训练 (11 维回归) ..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${CADT_ROOT}"

python main.py \
  --dataset dailyeeg \
  --task_type regression \
  --x_dim 59 \
  --n_class 11 \
  --seq_len "${SEQ_LEN}" \
  --n_epoch "${N_EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --ball_loss_weight 10 \
  --discriminator_loss 1 \
  --source_domains ${SOURCE_DOMAIN} \
  --target_domain "${TARGET_DOMAIN}"
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --n_clusters 6

echo "完成。结果保存目录: ${OUTPUT_DIR}"