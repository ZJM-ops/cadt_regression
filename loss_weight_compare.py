"""
Loss 权重搜索结果对比脚本
扫描 result_day_td / result_day_td_ablation / results_mlp / results_mlp_ablation 目录，
从 params.txt 提取 4 个 loss 权重参数，从 results.txt 提取 Test 指标，
按权重组装分组，展示 mean±sd（多 seed）和单次结果。
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# ======================== 配置 ========================
# 要扫描的结果目录（根据 encoder_type 和 ablation 选择）
SEARCH_DIRS = [
    "/root/zjm/cadt/results_cnn_search",  # CNN 权重搜索结果
]

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


def collect_runs():
    """收集所有实验运行"""
    runs = []
    for base_dir in SEARCH_DIRS:
        if not os.path.isdir(base_dir):
            print(f"[skip] {base_dir} not found")
            continue

        for exp_dir in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, exp_dir)
            if not os.path.isdir(full_path):
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
            has_weights = False
            for wk in WEIGHT_KEYS:
                val = params.get(wk)
                if val is not None:
                    try:
                        weights[wk] = float(val)
                        has_weights = True
                    except ValueError:
                        weights[wk] = None
                else:
                    weights[wk] = None

            if not has_weights:
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
    s = np.std(values, ddof=0)
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

# ---- 4. 保存 CSV ----
output_csv = "/root/zjm/cadt_test_code/loss_weight_search_results.csv"
df.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")
