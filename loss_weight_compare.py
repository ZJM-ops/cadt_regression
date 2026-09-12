"""
Loss 权重搜索结果对比脚本
扫描 results_cnn_search / result_day_td 目录，从 params.txt 提取 4 个 loss 权重参数，
从 results.txt 提取 Test 指标，按权重组组装分组，展示 mean±sd（多 seed）和单次结果。

说明：
- results_cnn_search: 主 sweep 结果，所有权重组合都接受
- result_day_td: 历史结果，只接受 baseline 权重 (sphere=10, disc=1, dom=1, tau=1)
  且必须 t_size=8（用模型 fc11 层 shape 验证，跳过 tsize32/tsize128 子目录）
  作为 baseline 的额外 seed 复用，无需重新训练
"""

import os
import re
import numpy as np
import pandas as pd

# ======================== 配置 ========================
# 要扫描的结果目录，每项格式: (目录路径, 是否只接受 baseline 权重)
# result_day_td 只接受 baseline 权重作为额外 seed 复用，无需重新训练
SEARCH_DIRS = [
    ("/vePFS-0x0d/home/zjm/cadt_regression/results_cnn_search", False),
    ("/vePFS-0x0d/home/zjm/cadt_regression/result_day_td",      True),
]

# result_day_td 中跳过这些子目录（t_size 与 sweep 不一致，已通过模型 shape 验证）
SKIP_SUBDIRS_RESULT_DAY_TD = {"tsize32", "tsize128"}

# baseline 权重定义
BASELINE_WEIGHTS = {
    "sphere_loss_weight": 10.0,
    "discriminator_loss_weight": 1.0,
    "domain_cls_weight": 1.0,
    "contrast_tau": 1.0,
}

# 要过滤的 split_mode（None=所有）
FILTER_SPLIT_MODE = None  # "cross_day" or "cross_subject" or "within_subject_day" or None

# 从 params.txt 提取的权重参数
WEIGHT_KEYS = [
    "sphere_loss_weight",
    "discriminator_loss_weight",
    "domain_cls_weight",
    "contrast_tau",
]

# 从 results.txt 提取的指标
METRIC_KEYS = [
    "Test MSE", "Test MAE", "Test RMSE",
    "Test R2", "Test Avg R2 (per-dim)",
    "Test Pearson", "Test Spearman",
    "Val R2", "Val Pearson", "Val Spearman",
]


# ======================== 解析函数 ========================
def parse_params(filepath):
    """从 params.txt 解析 loss 权重"""
    if not os.path.exists(filepath):
        return {}
    params = {}
    with open(filepath, "r") as f:
        for line in f:
            m = re.match(r"(\w+)\s*:\s*(.+)", line.strip())
            if m:
                key, val = m.group(1), m.group(2).strip()
                params[key] = val
    return params


def parse_results(filepath):
    """从 results.txt 解析指标"""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        text = f.read()
    metrics = {}
    for key in METRIC_KEYS:
        pattern = re.escape(key) + r":\s*([-+]?\d*\.?\d+)"
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def _iter_exp_dirs(base_dir):
    """枚举一个结果根目录下的所有实验目录。
    支持两种布局：
    - 顶层直接是实验目录: {base_dir}/{exp_dir}/params.txt
    - 一层中间子目录: {base_dir}/{sub}/{exp_dir}/params.txt（如 result_day_td/tsize8/...）
    """
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue
        # 直接是实验目录
        if os.path.isfile(os.path.join(full, "params.txt")):
            yield entry, full
            continue
        # 一层中间子目录（如 tsize8/）
        for sub_entry in sorted(os.listdir(full)):
            sub_full = os.path.join(full, sub_entry)
            if os.path.isdir(sub_full) and os.path.isfile(os.path.join(sub_full, "params.txt")):
                yield f"{entry}/{sub_entry}", sub_full


def collect_runs():
    """收集所有实验运行"""
    runs = []
    for entry in SEARCH_DIRS:
        # 兼容旧格式（纯字符串）和新格式（元组）
        if isinstance(entry, tuple):
            base_dir, baseline_only = entry
        else:
            base_dir, baseline_only = entry, False

        if not os.path.isdir(base_dir):
            print(f"[skip] {base_dir} not found")
            continue

        for exp_dir, full_path in _iter_exp_dirs(base_dir):
            # 对 result_day_td 跳过 tsize32/tsize128 子目录（t_size 不匹配）
            top_dir = exp_dir.split("/", 1)[0]
            if base_dir.endswith("result_day_td") and top_dir in SKIP_SUBDIRS_RESULT_DAY_TD:
                continue

            params_file = os.path.join(full_path, "params.txt")
            results_file = os.path.join(full_path, "results.txt")

            params = parse_params(params_file)
            metrics = parse_results(results_file)

            # 过滤 split_mode
            split_mode = params.get("split_mode", "")
            if FILTER_SPLIT_MODE and split_mode != FILTER_SPLIT_MODE:
                continue

            # 确保有权重信息
            weights = {}
            has_weights = True
            for wk in WEIGHT_KEYS:
                val = params.get(wk)
                if val is None:
                    has_weights = False
                    break
                try:
                    weights[wk] = float(val)
                except ValueError:
                    has_weights = False
                    break

            if not has_weights:
                continue

            # baseline_only 模式：只接受 baseline 权重组合
            if baseline_only:
                is_baseline = all(
                    abs(weights[wk] - BASELINE_WEIGHTS[wk]) < 1e-9
                    for wk in WEIGHT_KEYS
                )
                if not is_baseline:
                    continue

            if not metrics:
                continue

            runs.append({
                "dir": exp_dir,
                "base_dir": os.path.basename(base_dir),
                "split_mode": split_mode,
                "ablation": params.get("ablation", "unknown"),
                "encoder_type": params.get("encoder_type", "unknown"),
                "weights": weights,
                "metrics": metrics,
            })

    return runs


def group_by_weights(runs):
    """按权重组分组"""
    groups = {}
    for r in runs:
        # 用权重值组合作为 key
        key_parts = []
        for wk in WEIGHT_KEYS:
            v = r["weights"].get(wk)
            key_parts.append(f"{v}")
        group_key = " | ".join(f"{wk}={r['weights'].get(wk)}" for wk in WEIGHT_KEYS)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(r)

    return groups


def fmt_mean_sd(values):
    if len(values) == 0:
        return "N/A"
    m = np.mean(values)
    s = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}"


# ======================== 主流程 ========================
print("Collecting runs...")
runs = collect_runs()
print(f"Found {len(runs)} runs total")

if len(runs) == 0:
    print("No runs found. Check SEARCH_DIRS and FILTER_SPLIT_MODE.")
    exit()

groups = group_by_weights(runs)
print(f"Grouped into {len(groups)} unique weight combinations\n")

# ---- 1. 按权重组展示总体指标 ----
print("=" * 120)
print("  Loss Weight Search Results (mean±sd across seeds)")
print("=" * 120)

rows = []
for group_key, group_runs in sorted(groups.items()):
    row = {"Weight Config": group_key, "n_seeds": len(group_runs)}
    for mk in METRIC_KEYS:
        vals = [r["metrics"][mk] for r in group_runs if mk in r["metrics"]]
        row[mk] = fmt_mean_sd(vals)
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# ---- 2. 按 Test Pearson 排序 ----
print("\n" + "=" * 120)
print("  Sorted by Test Pearson (descending)")
print("=" * 120)

# 提取 mean 值用于排序
sort_rows = []
for group_key, group_runs in groups.items():
    vals = [r["metrics"].get("Test Pearson", 0) for r in group_runs]
    if vals:
        sort_rows.append((group_key, np.mean(vals), len(group_runs)))

sort_rows.sort(key=lambda x: x[1], reverse=True)

for rank, (gk, mean_pearson, n) in enumerate(sort_rows, 1):
    print(f"  #{rank}  Pearson={mean_pearson:.4f}  (n={n})  {gk}")

# ---- 3. Stage 1 单变量分析 ----
print("\n" + "=" * 120)
print("  Stage 1: Single-Variable Sensitivity")
print("=" * 120)

DEFAULTS = {"sphere_loss_weight": 10.0, "discriminator_loss_weight": 1.0,
            "domain_cls_weight": 1.0, "contrast_tau": 1.0}

for var_key in WEIGHT_KEYS:
    print(f"\n--- Varying: {var_key} ---")
    var_rows = []
    for group_key, group_runs in groups.items():
        # 只保留其他权重 = 默认值的组
        weights = group_runs[0]["weights"]
        is_single_var = True
        for wk in WEIGHT_KEYS:
            if wk == var_key:
                continue
            if weights.get(wk) != DEFAULTS[wk]:
                is_single_var = False
                break
        if not is_single_var:
            continue

        vals_pearson = [r["metrics"].get("Test Pearson", 0) for r in group_runs]
        vals_r2 = [r["metrics"].get("Test R2", 0) for r in group_runs]
        vals_spearman = [r["metrics"].get("Test Spearman", 0) for r in group_runs]

        var_rows.append({
            var_key: weights.get(var_key),
            "n": len(group_runs),
            "Test Pearson": fmt_mean_sd(vals_pearson),
            "Test R2": fmt_mean_sd(vals_r2),
            "Test Spearman": fmt_mean_sd(vals_spearman),
        })

    if var_rows:
        df_var = pd.DataFrame(var_rows)
        print(df_var.to_string(index=False))
    else:
        print("  (no single-variable runs found)")

# ---- 4. 保存 CSV（字符串版：mean±sd）----
output_csv = "/vePFS-0x0d/home/zjm/cadt_regression/report/07.loss_weight_search_results.csv"
df.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")

# ---- 5. 保存数值版 CSV（每行一个实验，权重/指标都是数字，供 Jupyter 绘图）----
# 直接从 runs 派生，按 (4 权重 + split_mode) 分组，保留 split_mode 列
DEFAULTS = {"sphere_loss_weight": 10.0, "discriminator_loss_weight": 1.0,
            "domain_cls_weight": 1.0, "contrast_tau": 1.0}

def _classify_sweep(weights):
    """识别单变量 sweep：其余 3 个=默认值时返回变化的那个 key，全默认返回 baseline"""
    all_default = all(
        weights.get(k) is not None and abs(weights[k] - DEFAULTS[k]) < 1e-9
        for k in DEFAULTS
    )
    if all_default:
        return "baseline"
    for var_key in DEFAULTS:
        others_default = all(
            weights.get(k) is not None and abs(weights[k] - DEFAULTS[k]) < 1e-9
            for k in DEFAULTS if k != var_key
        )
        if others_default:
            return var_key
    return "stage2_or_other"

# 按 (4 权重 + split_mode) 分组
groups_with_split = {}
for r in runs:
    key = tuple(r["weights"].get(wk) for wk in WEIGHT_KEYS) + (r["split_mode"],)
    groups_with_split.setdefault(key, []).append(r)

numeric_rows = []
for key, group_runs in groups_with_split.items():
    weights = group_runs[0]["weights"]
    rec = {
        "sphere_loss_weight": weights.get("sphere_loss_weight"),
        "discriminator_loss_weight": weights.get("discriminator_loss_weight"),
        "domain_cls_weight": weights.get("domain_cls_weight"),
        "contrast_tau": weights.get("contrast_tau"),
        "split_mode": group_runs[0]["split_mode"],
        "n_seeds": len(group_runs),
    }
    for mk in METRIC_KEYS:
        vals = [r["metrics"][mk] for r in group_runs if mk in r["metrics"]]
        rec[f"{mk}_mean"] = float(np.mean(vals)) if vals else None
    rec["sweep_type"] = _classify_sweep(weights)
    numeric_rows.append(rec)

df_numeric = pd.DataFrame(numeric_rows)

output_csv_numeric = "/vePFS-0x0d/home/zjm/cadt_regression/report/07.loss_weight_search_results_numeric.csv"
df_numeric.to_csv(output_csv_numeric, index=False)
print(f"Saved (numeric, for plotting): {output_csv_numeric}")
print("sweep_type × split_mode distribution:")
print(pd.crosstab(df_numeric["sweep_type"], df_numeric["split_mode"]).to_string())
