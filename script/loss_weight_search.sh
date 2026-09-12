#!/bin/bash
# Loss 权重搜索脚本（多 seed 重复实验）
#
# Stage 1: 单变量扫描 — 每次只变一个权重，其余固定为默认值
# Stage 2: 组合最优 — 用 Stage 1 找到的最优值组合跑
#
# 多 seed 策略：
#   - baseline (sphere=10, disc=1, dom=1, tau=1) 不在本脚本训练，
#     复用 result_day_td 的历史结果作为额外 seed（见 loss_weight_compare.py）
#   - 非 baseline 候选值每个跑 NUM_SEEDS 次（不固定 seed，纯重复）
#
# 用法:
#   bash /vePFS-0x0d/home/zjm/cadt_regression/script/loss_weight_search.sh             # Stage 1 全部 split_mode
#   bash /vePFS-0x0d/home/zjm/cadt_regression/script/loss_weight_search.sh stage2      # 仅 Stage 2
#   bash /vePFS-0x0d/home/zjm/cadt_regression/script/loss_weight_search.sh all         # 两阶段
#   bash /vePFS-0x0d/home/zjm/cadt_regression/script/loss_weight_search.sh stage1 cross_subject  # 指定单个 split_mode

# ==================== 基础配置 ====================
# 要运行的划分模式（空格分隔；可用命令行第 2 个参数覆盖）
DEFAULT_SPLIT_MODES="cross_subject cross_day within_subject_day"
SPLIT_MODES="${2:-${DEFAULT_SPLIT_MODES}}"

SPLIT_DIR="/vePFS-0x0d/DailyEEG/splits_new"
TARGET_SUBJECT_EMOTION=""

ENCODER_TYPE="cnn"                   # cnn | mlp
T_SIZE=8                             # 编码器嵌入维度
X_DIM=59                             # CNN=59 通道，MLP=295 DE 特征维
SEQ_LEN=2000                         # CNN=2000（10s @200Hz），MLP=1
DATA_DIR="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"

DEVICE=0
NUM_WORKERS=4
BATCH_SIZE=256
N_EPOCH=351
NORM_MODE="day"    # sample（逐样本）/ day（按天）/ subject（按被试）
ABLATION="none"    # 调权重时用 none（完整模型）

# 搜索结果单独存放的目录名（在 cadt/ 下）
RESULT_BASE="results_cnn_search"

# 每个（非 baseline）参数组合重复跑多少次（不固定 seed，纯重复实验）
NUM_SEEDS=5

# 默认权重值（Stage 1 的基线）
DEFAULT_SPHERE=10.0
DEFAULT_DISC=1.0
DEFAULT_DOM=1.0
DEFAULT_TAU=1.0

# ==================== 搜索空间 ====================
# Stage 1: 每个权重的候选值（不含各自默认值，baseline 复用 result_day_td 历史结果，不再训练）
# SPHERE_CANDIDATES="0.0 1.0 5.0 20.0 50.0"      # 默认 10.0 由 baseline 复用
# DISC_CANDIDATES="0.0 0.1 0.5 2.0 5.0"          # 默认 1.0  由 baseline 复用
# DOM_CANDIDATES="0.0 0.1 0.5 2.0 5.0"           # 默认 1.0  由 baseline 复用
# # contrast_tau 不再扫描，保持默认值 1.0（已由 baseline 覆盖，无需额外实验）
# TAU_CANDIDATES=""

SPHERE_CANDIDATES="0.0"      # 默认 10.0 由 baseline 复用
DISC_CANDIDATES="0.0"          # 默认 1.0  由 baseline 复用
DOM_CANDIDATES="0.0"           # 默认 1.0  由 baseline 复用
TAU_CANDIDATES=""

# Stage 2: 组合搜索（手动填入 Stage 1 最优值附近的小网格）
# 下面这些值在 Stage 1 跑完后手动调整
STAGE2_SPHERE="5.0 10.0"
STAGE2_DISC="0.5 1.0"
STAGE2_DOM="0.5 1.0"
STAGE2_TAU="1.0 2.0"

# ==================== 逻辑 ====================
STAGE="${1:-stage1}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="cadt_env"

mkdir -p /vePFS-0x0d/home/zjm/tmp
export TMPDIR=/vePFS-0x0d/home/zjm/tmp

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${CADT_ROOT}"

# 统计某个参数组合在指定 split_mode 下已成功跑过多少次（在 ${RESULT_BASE}/ 下匹配 params.txt 且含 results.txt）
# 用法: count_done <sphere> <disc> <dom> <tau> <split_mode>
# 输出: 已成功次数（整数，0 表示没跑过）
count_done() {
    python3 - "$1" "$2" "$3" "$4" "$5" "$RESULT_BASE" <<'PYEOF'
import sys, os, re
sphere, disc, dom, tau, split_mode, base = sys.argv[1:7]
def norm(x):
    try: return f"{float(x):g}"
    except: return x
target = {
    "sphere_loss_weight": norm(sphere),
    "discriminator_loss_weight": norm(disc),
    "domain_cls_weight": norm(dom),
    "contrast_tau": norm(tau),
    "split_mode": split_mode,
}
count = 0
if not os.path.isdir(base):
    print(0); sys.exit()
for d in os.listdir(base):
    p = os.path.join(base, d, "params.txt")
    if not os.path.isfile(p): continue
    pm = {}
    for line in open(p):
        m = re.match(r"\s*(\w+):\s*(\S+)", line)
        if m: pm[m.group(1)] = norm(m.group(2))
    if all(pm.get(k) == v for k, v in target.items()) and \
       os.path.isfile(os.path.join(base, d, "results.txt")):
        count += 1
print(count)
PYEOF
}

# 用法: run_single_seed <sphere> <disc> <dom> <tau> <split_mode> <label>
# 单次训练（不检查是否已跑过）
run_single_seed() {
    local sphere_w="$1"
    local disc_w="$2"
    local dom_w="$3"
    local tau="$4"
    local split_mode="$5"
    local label="$6"

    local args=(
        --x_dim "${X_DIM}"
        --n_targets 11
        --seq_len "${SEQ_LEN}"
        --t_size "${T_SIZE}"
        --encoder_type "${ENCODER_TYPE}"
        --n_epoch "${N_EPOCH}"
        --batch_size "${BATCH_SIZE}"
        --device "${DEVICE}"
        --num_workers "${NUM_WORKERS}"
        --sphere_loss_weight "${sphere_w}"
        --discriminator_loss_weight "${disc_w}"
        --domain_cls_weight "${dom_w}"
        --contrast_tau "${tau}"
        --split_mode "${split_mode}"
        --split_dir "${SPLIT_DIR}"
        --data_dir "${DATA_DIR}"
        --norm_mode "${NORM_MODE}"
        --ablation "${ABLATION}"
        --result_base "${RESULT_BASE}"
    )
    if [ -n "${TARGET_SUBJECT_EMOTION}" ]; then
        args+=(--target_subject "${TARGET_SUBJECT_EMOTION}")
    fi

    python main.py "${args[@]}"
}

# 用法: run_multi_seed <sphere> <disc> <dom> <tau> <split_mode> <label>
# 重复训练直到达到 NUM_SEEDS 次（已成功的次数计入，不足的补齐）
run_multi_seed() {
    local sphere_w="$1"
    local disc_w="$2"
    local dom_w="$3"
    local tau="$4"
    local split_mode="$5"
    local label="$6"

    local done_count
    done_count=$(count_done "${sphere_w}" "${disc_w}" "${dom_w}" "${tau}" "${split_mode}")
    done_count=$(echo "${done_count}" | tr -d '[:space:]')

    local remaining=$((NUM_SEEDS - done_count))
    echo ""
    echo ">>> [${label}] split=${split_mode} sphere=${sphere_w} disc=${disc_w} dom=${dom_w} tau=${tau}"
    echo ">>> 已成功 ${done_count}/${NUM_SEEDS}，补跑 ${remaining} 次"

    if [ "${remaining}" -le 0 ]; then
        echo ">>> [${label}] 已达 NUM_SEEDS=${NUM_SEEDS}，跳过"
        return 0
    fi

    local i
    for i in $(seq 1 "${remaining}"); do
        echo ""
        echo ">>> [${label}] seed ${i}/${remaining} (补跑第 $((done_count + i)) 次)"
        run_single_seed "${sphere_w}" "${disc_w}" "${dom_w}" "${tau}" "${split_mode}" "${label}"
        echo ">>> [${label}] seed ${i}/${remaining} done"
    done
}

# ==================== Stage 1: 单变量扫描 ====================
run_stage1() {
    echo "========================================"
    echo " Stage 1: Single-Variable Sweep (NUM_SEEDS=${NUM_SEEDS})"
    echo " SPLIT_MODES: ${SPLIT_MODES}"
    echo " Baseline (sphere=10,disc=1,dom=1,tau=1) 不训练，复用 result_day_td"
    echo "========================================"

    local count=0

    for SPLIT_MODE in ${SPLIT_MODES}; do
        echo ""
        echo "########## split_mode = ${SPLIT_MODE} ##########"

        # sphere_loss_weight
        for w in ${SPHERE_CANDIDATES}; do
            count=$((count + 1))
            run_multi_seed "${w}" "${DEFAULT_DISC}" "${DEFAULT_DOM}" "${DEFAULT_TAU}" \
                "${SPLIT_MODE}" "s1_sphere_${w}"
        done

        # discriminator_loss_weight
        for w in ${DISC_CANDIDATES}; do
            count=$((count + 1))
            run_multi_seed "${DEFAULT_SPHERE}" "${w}" "${DEFAULT_DOM}" "${DEFAULT_TAU}" \
                "${SPLIT_MODE}" "s1_disc_${w}"
        done

        # domain_cls_weight
        for w in ${DOM_CANDIDATES}; do
            count=$((count + 1))
            run_multi_seed "${DEFAULT_SPHERE}" "${DEFAULT_DISC}" "${w}" "${DEFAULT_TAU}" \
                "${SPLIT_MODE}" "s1_dom_${w}"
        done

        # contrast_tau
        for w in ${TAU_CANDIDATES}; do
            count=$((count + 1))
            run_multi_seed "${DEFAULT_SPHERE}" "${DEFAULT_DISC}" "${DEFAULT_DOM}" "${w}" \
                "${SPLIT_MODE}" "s1_tau_${w}"
        done
    done

    echo ""
    echo "========================================"
    echo " Stage 1 done! ${count} 个参数组合 × ${NUM_SEEDS} seeds"
    echo " (baseline 复用 result_day_td，未训练)"
    echo " 用 loss_weight_compare.py 查看结果，"
    echo " 然后把最优值填到 Stage 2 的搜索空间中。"
    echo "========================================"
}

# ==================== Stage 2: 组合搜索 ====================
run_stage2() {
    echo "========================================"
    echo " Stage 2: Grid Combination (NUM_SEEDS=${NUM_SEEDS})"
    echo " SPLIT_MODES: ${SPLIT_MODES}"
    echo "========================================"

    local count=0
    for SPLIT_MODE in ${SPLIT_MODES}; do
        echo ""
        echo "########## split_mode = ${SPLIT_MODE} ##########"
        for s in ${STAGE2_SPHERE}; do
            for d in ${STAGE2_DISC}; do
                for m in ${STAGE2_DOM}; do
                    for t in ${STAGE2_TAU}; do
                        count=$((count + 1))
                        run_multi_seed "${s}" "${d}" "${m}" "${t}" \
                            "${SPLIT_MODE}" "s2_s${s}_d${d}_m${m}_t${t}"
                    done
                done
            done
        done
    done

    echo ""
    echo "========================================"
    echo " Stage 2 done! ${count} 个参数组合 × ${NUM_SEEDS} seeds"
    echo "========================================"
}

case "${STAGE}" in
    stage1) run_stage1 ;;
    stage2) run_stage2 ;;
    all)    run_stage1; run_stage2 ;;
    *) echo "Usage: $0 [stage1|stage2|all] [split_mode]"; exit 1 ;;
esac
