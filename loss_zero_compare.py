"""
Loss 置零消融对比可视化
=========================
聚焦对比：把每个 loss 权重单独降到 0 时模型效果的变化，外加 no_detach 消融。

5 个条件：
  - baseline   : sphere=10, disc=1, dom=1, tau=1, ablation=none  (完整模型)
  - sphere=0   : sphere_loss_weight=0       (其余=默认)
  - disc=0     : discriminator_loss_weight=0 (其余=默认)
  - dom=0      : domain_cls_weight=0         (其余=默认)
  - no_detach  : 权重=baseline，但 ablation=no_detach (只移除判别器自身 detach 更新路径)

数据来源：
  - /vePFS-0x0d/home/zjm/cadt_regression/results_cnn_search  (sphere/disc/dom sweep，ablation=none)
  - /vePFS-0x0d/home/zjm/cadt_regression/result_day_td       (历史 baseline，只取 sphere=10/disc=1/dom=1/tau=1 且 t_size=8)
  - /vePFS-0x0d/home/zjm/cadt_regression/results_no_detach   (ablation=no_detach，权重=baseline)

输出：
  - /vePFS-0x0d/home/zjm/cadt_regression/report/08.loss_zero_compare.png        (分组柱状图)
  - /vePFS-0x0d/home/zjm/cadt_regression/report/08.loss_zero_compare_summary.csv (数值表)
  - stdout 打印 delta-from-baseline 表，便于直接判断哪个 loss 影响最大

约定：
  - std 用 ddof=1；单样本 (n<=1) std=0.0
  - 输出图存到 /vePFS-0x0d/home/zjm/cadt_regression/report/，前缀 08
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================== 配置 ========================
SEARCH_DIRS = [
    ("/vePFS-0x0d/home/zjm/cadt_regression/results_cnn_search", False, "none"),       # 主 sweep，权重任意
    ("/vePFS-0x0d/home/zjm/cadt_regression/result_day_td",      True,  "none"),       # 只接受 baseline 权重作为额外 seed
    ("/vePFS-0x0d/home/zjm/cadt_regression/results_no_detach",  False, "no_detach"), # no_detach 消融
]

# result_day_td 跳过 t_size 不一致的子目录
SKIP_SUBDIRS_RESULT_DAY_TD = {"tsize32", "tsize128"}

BASELINE_WEIGHTS = {
    "sphere_loss_weight": 10.0,
    "discriminator_loss_weight": 1.0,
    "domain_cls_weight": 1.0,
    "contrast_tau": 1.0,
}
WEIGHT_KEYS = list(BASELINE_WEIGHTS)

# 条件定义（按展示顺序）
CONDITIONS = ["baseline", "sphere=0", "disc=0", "dom=0", "no_detach"]

# 拆分模式（按展示顺序）
SPLIT_MODES = ["cross_subject", "cross_day", "within_subject_day"]

# 重点展示的测试集指标（回归任务核心）
METRICS = ["Test Pearson", "Test Spearman", "Test R2", "Test RMSE"]
# 额外记录的指标（写进 CSV，但不进主图）
EXTRA_METRICS = ["Test MSE", "Test MAE", "Test Avg R2 (per-dim)",
                 "Val R2", "Val Pearson", "Val Spearman"]

# 条件配色：baseline 灰色，三个置零条件用区分色，no_detach 红色
COND_COLORS = {
    "baseline":  "#7f7f7f",  # 灰
    "sphere=0":  "#1f77b4",  # 蓝
    "disc=0":    "#ff7f0e",  # 橙
    "dom=0":     "#2ca02c",  # 绿
    "no_detach": "#d62728",  # 红
}

OUTPUT_PNG  = "/vePFS-0x0d/home/zjm/cadt_regression/report/08.loss_zero_compare.png"
OUTPUT_CSV  = "/vePFS-0x0d/home/zjm/cadt_regression/report/08.loss_zero_compare_summary.csv"


# ======================== 解析函数 ========================
def parse_params(filepath):
    p = {}
    if not os.path.exists(filepath):
        return p
    for line in open(filepath):
        m = re.match(r"(\w+)\s*:\s*(.+)", line.strip())
        if m:
            p[m.group(1)] = m.group(2).strip()
    return p


def parse_results(filepath):
    if not os.path.exists(filepath):
        return {}
    t = open(filepath).read()
    out = {}
    for k in METRICS + EXTRA_METRICS:
        m = re.search(re.escape(k) + r":\s*([-+]?\d*\.?\d+)", t)
        if m:
            out[k] = float(m.group(1))
    return out


def _iter_exp_dirs(base_dir):
    """枚举一个结果根目录下所有实验目录（支持直接或一层嵌套）。"""
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "params.txt")):
            yield entry, full
            continue
        for sub_entry in sorted(os.listdir(full)):
            sub_full = os.path.join(full, sub_entry)
            if os.path.isdir(sub_full) and os.path.isfile(os.path.join(sub_full, "params.txt")):
                yield f"{entry}/{sub_entry}", sub_full


def classify(weights, ablation):
    """把一次实验归到 5 个条件之一（或 'other'）。"""
    try:
        w = {k: float(weights[k]) for k in WEIGHT_KEYS}
    except (KeyError, ValueError):
        return "other"

    is_base_w = all(abs(w[k] - BASELINE_WEIGHTS[k]) < 1e-9 for k in WEIGHT_KEYS)

    if ablation == "no_detach":
        # no_detach 条件要求权重=baseline
        return "no_detach" if is_base_w else "other"

    if ablation != "none":
        return "other"

    if is_base_w:
        return "baseline"
    # 其余权重=默认，只 sphere=0
    if (abs(w["sphere_loss_weight"]) < 1e-9
            and abs(w["discriminator_loss_weight"] - 1.0) < 1e-9
            and abs(w["domain_cls_weight"] - 1.0) < 1e-9
            and abs(w["contrast_tau"] - 1.0) < 1e-9):
        return "sphere=0"
    if (abs(w["discriminator_loss_weight"]) < 1e-9
            and abs(w["sphere_loss_weight"] - 10.0) < 1e-9
            and abs(w["domain_cls_weight"] - 1.0) < 1e-9
            and abs(w["contrast_tau"] - 1.0) < 1e-9):
        return "disc=0"
    if (abs(w["domain_cls_weight"]) < 1e-9
            and abs(w["sphere_loss_weight"] - 10.0) < 1e-9
            and abs(w["discriminator_loss_weight"] - 1.0) < 1e-9
            and abs(w["contrast_tau"] - 1.0) < 1e-9):
        return "dom=0"
    return "other"


def collect_runs():
    runs = []
    for base_dir, baseline_only, _expected_abl in SEARCH_DIRS:
        if not os.path.isdir(base_dir):
            print(f"[skip] {base_dir} 不存在")
            continue
        for exp_dir, full_path in _iter_exp_dirs(base_dir):
            top_dir = exp_dir.split("/", 1)[0]
            if base_dir.endswith("result_day_td") and top_dir in SKIP_SUBDIRS_RESULT_DAY_TD:
                continue

            p = parse_params(os.path.join(full_path, "params.txt"))
            r = parse_results(os.path.join(full_path, "results.txt"))
            if not r:
                continue

            weights = {k: p.get(k) for k in WEIGHT_KEYS}
            if any(v is None for v in weights.values()):
                continue

            # baseline_only 目录：只接受 baseline 权重
            if baseline_only:
                try:
                    if not all(abs(float(weights[k]) - BASELINE_WEIGHTS[k]) < 1e-9 for k in WEIGHT_KEYS):
                        continue
                except ValueError:
                    continue

            # result_day_td 早于 ablation 特性，params.txt 无 ablation 字段，
            # 这些 run 都是完整模型 (ablation=none)，故缺失时按 "none" 处理
            abl = p.get("ablation", "none")
            split = p.get("split_mode", "?")
            cond = classify(weights, abl)
            if cond == "other":
                continue

            runs.append({
                "dir": exp_dir,
                "base_dir": os.path.basename(base_dir),
                "split_mode": split,
                "ablation": abl,
                "condition": cond,
                "metrics": r,
            })
    return runs


def mean_sd(values):
    """mean±sd，ddof=1；单样本 sd=0。"""
    if len(values) == 0:
        return (np.nan, np.nan)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return (m, s)


# ======================== 主流程 ========================
print("Collecting runs...")
runs = collect_runs()
print(f"Found {len(runs)} runs matching the 5 target conditions\n")

# 统计每个 (split, condition) 的 seed 数
print("Seed counts (split × condition):")
print(f"{'split':<22} " + " ".join(f"{c:<11}" for c in CONDITIONS))
for split in SPLIT_MODES:
    row = [sum(1 for r in runs if r["split_mode"] == split and r["condition"] == c) for c in CONDITIONS]
    print(f"{split:<22} " + " ".join(f"{n:<11}" for n in row))
print()

# 聚合：每个 (split, condition, metric) 算 mean±sd
agg = {}  # agg[(split, condition)][metric] = (mean, sd, n)
for split in SPLIT_MODES:
    for cond in CONDITIONS:
        sub = [r for r in runs if r["split_mode"] == split and r["condition"] == cond]
        entry = {}
        for mk in METRICS + EXTRA_METRICS:
            vals = [r["metrics"][mk] for r in sub if mk in r["metrics"]]
            m, s = mean_sd(vals)
            entry[mk] = (m, s, len(vals))
        agg[(split, cond)] = entry

# ---- 数值表 (CSV) ----
csv_rows = []
for split in SPLIT_MODES:
    for cond in CONDITIONS:
        e = agg[(split, cond)]
        row = {"split_mode": split, "condition": cond, "n_seeds": e[METRICS[0]][2]}
        for mk in METRICS + EXTRA_METRICS:
            m, s, _ = e[mk]
            row[f"{mk}_mean"] = m
            row[f"{mk}_sd"] = s
        csv_rows.append(row)
df_csv = pd.DataFrame(csv_rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_csv.to_csv(OUTPUT_CSV, index=False)
print(f"Saved summary CSV: {OUTPUT_CSV}\n")

# ---- delta-from-baseline 表（直接判断哪个 loss 影响最大）----
print("=" * 100)
print("  Delta from baseline (condition - baseline)  ——  负值=变差，正值=变好")
print("  指标方向：Pearson/Spearman/R2 越大越好；RMSE 越小越好")
print("=" * 100)
delta_conds = ["sphere=0", "disc=0", "dom=0", "no_detach"]
for split in SPLIT_MODES:
    print(f"\n### split_mode = {split}")
    base = agg[(split, "baseline")]
    header = f"{'condition':<12} " + " ".join(f"{m:<22}" for m in METRICS)
    print(header)
    for cond in delta_conds:
        e = agg[(split, cond)]
        if e[METRICS[0]][2] == 0:
            print(f"{cond:<12} " + " ".join(f"{'(no data)':<22}" for _ in METRICS))
            continue
        cells = []
        for m in METRICS:
            bm, _, _ = base[m]
            cm, _, cn = e[m]
            if np.isnan(bm) or np.isnan(cm):
                cells.append(f"{'N/A':<22}")
            else:
                d = cm - bm
                cells.append(f"{d:+.4f} (n={cn}){'':<6}"[:22])
        print(f"{cond:<12} " + " ".join(cells))

# ---- baseline 绝对值参考 ----
print("\n" + "=" * 100)
print("  Baseline absolute values (mean±sd, ddof=1)")
print("=" * 100)
print(f"{'split':<22} " + " ".join(f"{m:<22}" for m in METRICS))
for split in SPLIT_MODES:
    base = agg[(split, "baseline")]
    cells = []
    for m in METRICS:
        bm, bs, bn = base[m]
        if np.isnan(bm):
            cells.append(f"{'N/A':<22}")
        else:
            cells.append(f"{bm:.4f}±{bs:.4f}(n={bn})"[:22])
    print(f"{split:<22} " + " ".join(cells))

# ======================== 绘图 ========================
# 注册 CJK 字体以正确显示中文（Noto Sans CJK SC）
import matplotlib.font_manager as _fm
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if os.path.isfile(_fp):
        _fm.fontManager.addfont(_fp)
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

n_splits = len(SPLIT_MODES)
n_metrics = len(METRICS)
fig, axes = plt.subplots(n_splits, n_metrics, figsize=(5 * n_metrics, 4.2 * n_splits),
                         sharex=True, squeeze=False)

x = np.arange(len(CONDITIONS))
bar_w = 0.62

for i, split in enumerate(SPLIT_MODES):
    for j, mk in enumerate(METRICS):
        ax = axes[i][j]
        means, sds, ns, colors = [], [], [], []
        for cond in CONDITIONS:
            m, s, n = agg[(split, cond)][mk]
            means.append(0.0 if np.isnan(m) else m)
            sds.append(0.0 if np.isnan(s) else s)
            ns.append(n)
            colors.append(COND_COLORS[cond])

        bars = ax.bar(x, means, bar_w, yerr=sds, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.6,
                      error_kw={"ecolor": "black", "elinewidth": 1, "alpha": 0.8})

        # 在柱顶标注 n（含 N/A 标记）
        for xi, (b, n) in enumerate(zip(bars, ns)):
            h = b.get_height()
            label = f"n={n}" if n > 0 else "N/A"
            ax.text(xi, h + (max(means) - min(means)) * 0.03 + 1e-9,
                    label, ha="center", va="bottom", fontsize=8, color="dimgray")

        # baseline 参考线
        bm, _, _ = agg[(split, "baseline")][mk]
        if not np.isnan(bm):
            ax.axhline(bm, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8,
                       label=f"baseline={bm:.4f}")

        ax.set_xticks(x)
        # 仅底行显示 x tick 标签
        if i == n_splits - 1:
            ax.set_xticklabels(CONDITIONS, rotation=20, ha="right", fontsize=9)
        else:
            ax.set_xticklabels([""] * len(CONDITIONS))

        if j == 0:
            ax.set_ylabel(split, fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_title(mk, fontsize=12, fontweight="bold")

        ax.legend(fontsize=8, loc="best")
        # 留出顶部空间给 n 标注
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(top=ymax + (ymax - ymin) * 0.12)

fig.suptitle("Loss 置零消融对比 — 各 loss 权重降到 0 / no_detach 相对 baseline 的影响\n"
             "(bars=mean, errorbar=std ddof=1, 虚线=baseline mean; Pearson/Spearman/R2 越大越好, RMSE 越小越好)",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {OUTPUT_PNG}")
