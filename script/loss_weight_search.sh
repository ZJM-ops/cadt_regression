#!/bin/bash
# Loss 权重搜索脚本（两阶段策略）
#
# Stage 1: 单变量扫描 — 每次只变一个权重，其余固定为默认值
# Stage 2: 组合最优 — 用 Stage 1 找到的最优值组合跑
#
# 用法:
#   bash /root/zjm/cadt/script/loss_weight_search.sh          # 仅 Stage 1
#   bash /root/zjm/cadt/script/loss_weight_search.sh stage2   # 仅 Stage 2（需先手动填入 Stage 1 最优值）
#   bash /root/zjm/cadt/script/loss_weight_search.sh all      # 两阶段都跑

# ==================== 基础配置 ====================
SPLIT_MODE="cross_day"
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

# 默认权重值（Stage 1 的基线）
DEFAULT_SPHERE=10.0
DEFAULT_DISC=1.0
DEFAULT_DOM=1.0
DEFAULT_TAU=1.0

# ==================== 搜索空间 ====================
# Stage 1: 每个权重的候选值
SPHERE_CANDIDATES="1.0 5.0 10.0 20.0 50.0"
DISC_CANDIDATES="0.1 0.5 1.0 2.0 5.0"
DOM_CANDIDATES="0.1 0.5 1.0 2.0 5.0"
TAU_CANDIDATES="0.5 1.0 2.0 5.0"

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

# 检查某个参数组合是否已成功跑过（在 ${RESULT_BASE}/ 下存在匹配 params.txt 且含 results.txt 的目录）
# 用法: already_done <sphere> <disc> <dom> <tau>
# 返回 0=已完成（跳过），1=未完成（需跑）
already_done() {
    local result
    result=$(python3 - "$1" "$2" "$3" "$4" "$SPLIT_MODE" "$RESULT_BASE" <<'PYEOF'
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
if not os.path.isdir(base):
    print("no"); sys.exit()
for d in os.listdir(base):
    p = os.path.join(base, d, "params.txt")
    if not os.path.isfile(p): continue
    pm = {}
    for line in open(p):
        m = re.match(r"\s*(\w+):\s*(\S+)", line)
        if m: pm[m.group(1)] = norm(m.group(2))
    if all(pm.get(k) == v for k, v in target.items()) and \
       os.path.isfile(os.path.join(base, d, "results.txt")):
        print("yes"); sys.exit()
print("no")
PYEOF
)
    [ "${result}" = "yes" ]
}

run_single() {
    local sphere_w="$1"
    local disc_w="$2"
    local dom_w="$3"
    local tau="$4"
    local label="$5"

    echo ""
    echo ">>> [${label}] sphere=${sphere_w} disc=${disc_w} dom=${dom_w} tau=${tau}"

    # 跳过已成功跑过的参数组合（results.txt 存在）
    if already_done "${sphere_w}" "${disc_w}" "${dom_w}" "${tau}"; then
        echo ">>> [${label}] 已跑过（results.txt 存在），跳过"
        return 0
    fi

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
        --split_mode "${SPLIT_MODE}"
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
    echo ">>> [${label}] done"
}

# ==================== Stage 1: 单变量扫描 ====================
run_stage1() {
    echo "========================================"
    echo " Stage 1: Single-Variable Sweep"
    echo "========================================"

    local count=0

    # sphere_loss_weight
    for w in ${SPHERE_CANDIDATES}; do
        count=$((count + 1))
        run_single "${w}" "${DEFAULT_DISC}" "${DEFAULT_DOM}" "${DEFAULT_TAU}" \
            "s1_sphere_${w}"
    done

    # discriminator_loss_weight
    for w in ${DISC_CANDIDATES}; do
        count=$((count + 1))
        run_single "${DEFAULT_SPHERE}" "${w}" "${DEFAULT_DOM}" "${DEFAULT_TAU}" \
            "s1_disc_${w}"
    done

    # domain_cls_weight
    for w in ${DOM_CANDIDATES}; do
        count=$((count + 1))
        run_single "${DEFAULT_SPHERE}" "${DEFAULT_DISC}" "${w}" "${DEFAULT_TAU}" \
            "s1_dom_${w}"
    done

    # contrast_tau
    for w in ${TAU_CANDIDATES}; do
        count=$((count + 1))
        run_single "${DEFAULT_SPHERE}" "${DEFAULT_DISC}" "${DEFAULT_DOM}" "${w}" \
            "s1_tau_${w}"
    done

    echo ""
    echo "========================================"
    echo " Stage 1 done! ${count} runs total."
    echo " 用 loss_weight_compare.py 查看结果，"
    echo " 然后把最优值填到 Stage 2 的搜索空间中。"
    echo "========================================"
}

# ==================== Stage 2: 组合搜索 ====================
run_stage2() {
    echo "========================================"
    echo " Stage 2: Grid Combination"
    echo "========================================"

    local count=0
    for s in ${STAGE2_SPHERE}; do
        for d in ${STAGE2_DISC}; do
            for m in ${STAGE2_DOM}; do
                for t in ${STAGE2_TAU}; do
                    count=$((count + 1))
                    run_single "${s}" "${d}" "${m}" "${t}" \
                        "s2_s${s}_d${d}_m${m}_t${t}"
                done
            done
        done
    done

    echo ""
    echo "========================================"
    echo " Stage 2 done! ${count} runs total."
    echo "========================================"
}

case "${STAGE}" in
    stage1) run_stage1 ;;
    stage2) run_stage2 ;;
    all)    run_stage1; run_stage2 ;;
    *) echo "Usage: $0 [stage1|stage2|all]"; exit 1 ;;
esac
