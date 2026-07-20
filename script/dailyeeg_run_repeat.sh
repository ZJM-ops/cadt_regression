#!/bin/bash
# DailyEEG 重复实验脚本
# 串行运行5次实验（一张卡），一个跑完自动接下一个

# ==================== 参数设置 ====================
SOURCE_DOMAIN=0        # 源域 ID 
TARGET_DOMAIN=42       # 目标域 ID 
DEVICE=0                # GPU 编号
REPEAT=5               # 重复次数
# ==================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/dailyeeg_run.sh"
LOG_DIR="/root/cadt/results"

echo "========================================"
echo " 串行运行 DailyEEG 实验 ${REPEAT} 次"
echo " 源域: ${SOURCE_DOMAIN}, 目标域: ${TARGET_DOMAIN}"
echo " GPU: ${DEVICE}"
echo "========================================"

START_TIME=$(date +%s)

for i in $(seq 1 ${REPEAT}); do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/run_${i}_${SOURCE_DOMAIN}_${TARGET_DOMAIN}_${TIMESTAMP}.log"
    
    echo ""
    echo "========================================"
    echo " 启动第 ${i}/${REPEAT} 次实验"
    echo " 时间: $(date)"
    echo " 日志: ${LOG_FILE}"
    echo "========================================"
    
    # 设置不同的随机种子
    export PYTHONHASHSEED=$((RANDOM))
    export SEED=$((i * 1000 + RANDOM))
    
    # 串行运行，等待上一次完成后再开始
    bash "${RUN_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
    
    # 检查上一次命令是否成功
    if [ $? -eq 0 ]; then
        echo "第 ${i}/${REPEAT} 次实验完成"
    else
        echo "第 ${i}/${REPEAT} 次实验失败，退出循环"
        exit 1
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo " 所有 ${REPEAT} 次实验完成！"
echo " 总耗时: ${HOURS}小时${MINUTES}分钟${SECONDS}秒"
echo " 结果目录: ${LOG_DIR}"
echo "========================================"