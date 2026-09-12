"""
对比4组实验的 within_subject_day 结果（mean±sd）：
  1. CNN Full      — result_day_td/tsize8
  2. CNN Ablation  — result_day_td_ablation/tsize8
  3. MLP Full      — results_mlp
  4. MLP Ablation  — results_mlp_ablation
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# ======================== 配置 ========================
EXPERIMENTS = {
    "CNN-Full":     "/vePFS-0x0d/home/zjm/cadt_regression/result_day_td/tsize8",
    "CNN-Ablation": "/vePFS-0x0d/home/zjm/cadt_regression/result_day_td_ablation/tsize8",
    "MLP-Full":     "/vePFS-0x0d/home/zjm/cadt_regression/results_mlp",
    "MLP-Ablation": "/vePFS-0x0d/home/zjm/cadt_regression/results_mlp_ablation",
}
SPLIT_MODE = "within_subject_day"

# 要展示的总体指标（results.txt 中的键名）
OVERALL_METRICS = [
    ("Test MSE",              "Test MSE"),
    ("Test MAE",              "Test MAE"),
    ("Test RMSE",             "Test RMSE"),
    ("Test R2",               "Test R2"),
    ("Test Avg R2 (per-dim)", "Test Avg R2"),
    ("Test Pearson",          "Test Pearson"),
    ("Test Spearman",         "Test Spearman"),
    ("Val MSE",               "Val MSE"),
    ("Val MAE",               "Val MAE"),
    ("Val RMSE",              "Val RMSE"),
    ("Val R2",                "Val R2"),
    ("Val Avg R2 (per-dim)",  "Val Avg R2"),
    ("Val Pearson",           "Val Pearson"),
    ("Val Spearman",          "Val Spearman"),
]

EMOTIONS = [
    "inspired", "alert", "determined", "attentive", "active",
    "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue"
]

PER_EMOTION_METRICS = ["MSE", "MAE", "R2", "Pearson", "Spearman"]


# ======================== 解析函数 ========================
def parse_results_txt(filepath):
    """解析 results.txt，返回总体指标 dict + 每情绪指标 dict"""
    with open(filepath, "r") as f:
        text = f.read()

    overall = {}
    for key in [m[0] for m in OVERALL_METRICS]:
        # 匹配 "Key: value" 或 "Key: value" (带括号)
        pattern = re.escape(key) + r":\s*([-+]?\d*\.?\d+)"
        m = re.search(pattern, text)
        if m:
            overall[key] = float(m.group(1))

    # 解析 per-emotion（Test Set）
    per_emotion_test = {}
    per_emotion_val = {}

    # 找 Test Set per-emotion 表（跳过分隔线 ------，匹配到下一个 --- 标题或文件末尾）
    test_section = re.search(
        r"--- Per-Emotion Metrics \(Test Set\) ---\s*\n.*?\n-+\n(.*?)(?:\n---|\Z)", text, re.DOTALL
    )
    if test_section:
        for emo in EMOTIONS:
            pattern = rf"^{emo}\s+([\d.]+)\s+([\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)"
            m = re.search(pattern, test_section.group(1), re.MULTILINE)
            if m:
                per_emotion_test[emo] = {
                    "MSE": float(m.group(1)),
                    "MAE": float(m.group(2)),
                    "R2": float(m.group(3)),
                    "Pearson": float(m.group(4)),
                    "Spearman": float(m.group(5)),
                }

    # 找 Validation Set per-emotion 表
    val_section = re.search(
        r"--- Per-Emotion Metrics \(Validation Set\) ---\s*\n.*?\n-+\n(.*?)(?:\n---|\Z)", text, re.DOTALL
    )
    if val_section:
        for emo in EMOTIONS:
            pattern = rf"^{emo}\s+([\d.]+)\s+([\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)"
            m = re.search(pattern, val_section.group(1), re.MULTILINE)
            if m:
                per_emotion_val[emo] = {
                    "MSE": float(m.group(1)),
                    "MAE": float(m.group(2)),
                    "R2": float(m.group(3)),
                    "Pearson": float(m.group(4)),
                    "Spearman": float(m.group(5)),
                }

    return overall, per_emotion_test, per_emotion_val


def collect_experiment(base_dir):
    """收集一个实验组下所有 within_subject_day 的结果"""
    pattern = os.path.join(base_dir, f"*{SPLIT_MODE}*")
    exp_dirs = sorted(glob.glob(pattern))

    results = []
    for d in exp_dirs:
        rfile = os.path.join(d, "results.txt")
        if not os.path.exists(rfile):
            print(f"  [skip] No results.txt in {d}")
            continue
        overall, pe_test, pe_val = parse_results_txt(rfile)
        results.append({
            "dir": os.path.basename(d),
            "overall": overall,
            "per_emotion_test": pe_test,
            "per_emotion_val": pe_val,
        })

    return results


def fmt_mean_sd(values):
    """格式化为 mean±sd"""
    if len(values) == 0:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values, ddof=0)  # 总体标准差
    return f"{mean:.4f}±{std:.4f}"


# ======================== 主流程 ========================
all_data = {}
for exp_name, exp_dir in EXPERIMENTS.items():
    print(f"\n{'='*60}")
    print(f"Scanning: {exp_name} -> {exp_dir}")
    results = collect_experiment(exp_dir)
    all_data[exp_name] = results
    print(f"  Found {len(results)} runs")

# ---- 1. 总体指标对比表 ----
print("\n" + "=" * 80)
print("  Overall Metrics Comparison (mean±sd)")
print("=" * 80)

rows = []
for metric_key, display_name in OVERALL_METRICS:
    row = {"Metric": display_name}
    for exp_name in EXPERIMENTS:
        values = []
        for r in all_data[exp_name]:
            if metric_key in r["overall"]:
                values.append(r["overall"][metric_key])
        row[exp_name] = fmt_mean_sd(values)
    rows.append(row)

df_overall = pd.DataFrame(rows)
print(df_overall.to_string(index=False))

# ---- 2. Per-Emotion Test Set 对比 ----
for metric in PER_EMOTION_METRICS:
    print(f"\n{'='*80}")
    print(f"  Per-Emotion Test {metric} (mean±sd)")
    print("=" * 80)

    rows = []
    for emo in EMOTIONS + ["Average"]:
        row = {"Emotion": emo}
        for exp_name in EXPERIMENTS:
            values = []
            for r in all_data[exp_name]:
                pe = r["per_emotion_test"]
                if emo == "Average":
                    # 用 overall 里对应指标
                    overall_key_map = {
                        "MSE": "Test MSE", "MAE": "Test MAE",
                        "R2": "Test R2", "Pearson": "Test Pearson",
                        "Spearman": "Test Spearman",
                    }
                    key = overall_key_map.get(metric)
                    if key and key in r["overall"]:
                        values.append(r["overall"][key])
                else:
                    if emo in pe and metric in pe[emo]:
                        values.append(pe[emo][metric])
            row[exp_name] = fmt_mean_sd(values)
        rows.append(row)

    df_emo = pd.DataFrame(rows)
    print(df_emo.to_string(index=False))

# ---- 3. 保存到 CSV ----
output_dir = "/vePFS-0x0d/home/zjm/cadt_regression"
csv_overall = os.path.join(output_dir, "compare_ablation_overall.csv")
df_overall.to_csv(csv_overall, index=False)
print(f"\nSaved: {csv_overall}")

for metric in PER_EMOTION_METRICS:
    rows = []
    for emo in EMOTIONS + ["Average"]:
        row = {"Emotion": emo}
        for exp_name in EXPERIMENTS:
            values = []
            for r in all_data[exp_name]:
                pe = r["per_emotion_test"]
                if emo == "Average":
                    overall_key_map = {
                        "MSE": "Test MSE", "MAE": "Test MAE",
                        "R2": "Test R2", "Pearson": "Test Pearson",
                        "Spearman": "Test Spearman",
                    }
                    key = overall_key_map.get(metric)
                    if key and key in r["overall"]:
                        values.append(r["overall"][key])
                else:
                    if emo in pe and metric in pe[emo]:
                        values.append(pe[emo][metric])
            row[exp_name] = fmt_mean_sd(values)
        rows.append(row)
    df_m = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"compare_ablation_test_{metric}.csv")
    df_m.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

print("\nDone!")
