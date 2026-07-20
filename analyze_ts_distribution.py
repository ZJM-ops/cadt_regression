import numpy as np
import pandas as pd
import os
import glob

DATA_ROOT = "/vePFS-0x0d/DailyEEG"
LABELED_DIR = "/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"

# 加载已标注数据
ts_arr = np.load(os.path.join(LABELED_DIR, "ts.npy"))
sub_arr = np.load(os.path.join(LABELED_DIR, "sub.npy"))

print(f"已标注样本总数: {len(ts_arr)}")
print(f"时间戳范围: [{ts_arr.min():.1f}, {ts_arr.max():.1f}] 秒")

# 按被试分组统计
unique_subs = np.unique(sub_arr)
print(f"\n覆盖被试: {unique_subs}")

# 检查每个被试的数据
label_cols = [
    "inspired", "alert", "determined", "attentive", "active",
    "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue"
]

for sub_idx in sorted(unique_subs):
    sub_ts = ts_arr[sub_arr == sub_idx]
    sub_str = f"sub-{int(sub_idx):02d}"
    print(f"\n{'='*50}")
    print(f"被试 {sub_str}: {len(sub_ts)} 个样本")
    print(f"  时间戳范围: [{sub_ts.min():.1f}, {sub_ts.max():.1f}] 秒")
    
    # 读取行为数据获取onset
    beh_root = os.path.join(DATA_ROOT, sub_str)
    ses_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "derivatives", "eeg-prep", sub_str, "ses-*")))
    
    all_onsets = []
    for ses_dir in ses_dirs:
        ses_id = os.path.basename(ses_dir)
        beh_pattern = os.path.join(beh_root, ses_id, "beh", f"{sub_str}_{ses_id}_task-dailylife_emotion_beh*.tsv")
        beh_files = sorted(glob.glob(beh_pattern))
        if len(beh_files) > 0:
            df_list = []
            for bf in beh_files:
                df_part = pd.read_csv(bf, sep="\t")
                df_list.append(df_part)
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                if len(df) > 0:
                    all_onsets.extend(df["onset"].tolist())
    
    if all_onsets:
        all_onsets = np.array(all_onsets)
        print(f"  问卷onset范围: [{all_onsets.min():.1f}, {all_onsets.max():.1f}] 秒")
        print(f"  问卷前120秒区间范围: [{all_onsets.min()-120:.1f}, {all_onsets.max():.1f}] 秒")
        
        # 找出超出问卷前120秒区间的样本
        # 样本时间戳应该在 [onset-120, onset] 范围内
        # 检查是否有样本时间戳 > onset（超出问卷时间点）
        # 或者样本时间戳 < onset-120（早于120秒窗口）
        
        # 对于每个样本，检查是否在某个问卷的120秒窗口内
        outside_count = 0
        outside_samples = []
        for ts in sub_ts:
            in_window = False
            for onset in all_onsets:
                if onset - 120 <= ts <= onset:
                    in_window = True
                    break
            if not in_window:
                outside_count += 1
                outside_samples.append(ts)
        
        if outside_count > 0:
            outside_samples = sorted(outside_samples)
            print(f"  ❌ 超出120秒窗口的样本数: {outside_count}")
            print(f"  超出窗口的时间戳示例: {outside_samples[:10]}")
            # 分析超出原因
            print(f"  样本时间戳 > onset 的数量: {sum(ts > all_onsets.max() for ts in outside_samples)}")
            print(f"  样本时间戳 < onset-120 的数量: {sum(ts < all_onsets.min()-120 for ts in outside_samples)}")
        else:
            print(f"  ✅ 所有样本都在问卷前120秒窗口内")
    else:
        print(f"  ⚠️ 未找到行为数据")

print(f"\n{'='*50}")
print("分析完成")