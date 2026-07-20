#!/usr/bin/env python3
"""
验证生成的unlabeled数据是否正确（支持完整5文件格式）：
修正：将验证条件从 t <= onset 改为 t < onset，允许窗口起始时间恰好等于onset的样本
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Validate unlabeled raw EEG data")
    parser.add_argument(
        "--unlabeled-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_raw_unlabeled",
        help="生成的未标注数据目录",
    )
    parser.add_argument(
        "--labeled-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new",
        help="已标注数据目录",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/vePFS-0x0d/DailyEEG",
        help="DailyEEG数据集根路径",
    )
    parser.add_argument("--label-window", type=int, default=120, help="问卷前多少秒为已标注区间")
    parser.add_argument("--fs", type=int, default=200, help="采样率")
    parser.add_argument("--window-size", type=int, default=10, help="窗口大小（秒）")
    parser.add_argument("--step-size", type=int, default=5, help="滑动步长（秒）")
    parser.add_argument("--sample-fraction", type=float, default=0.1, help="X数据采样比例")
    return parser.parse_args()


def load_data(args):
    """加载所有数据"""
    print("=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)
    
    required_files = ["X.npy", "y.npy", "d.npy", "ts.npy", "sub.npy"]
    for f in required_files:
        path = os.path.join(args.unlabeled_dir, f)
        if not os.path.exists(path):
            raise RuntimeError(f"未找到必需文件: {path}")
    
    y = np.load(os.path.join(args.unlabeled_dir, "y.npy"))
    d = np.load(os.path.join(args.unlabeled_dir, "d.npy"))
    ts = np.load(os.path.join(args.unlabeled_dir, "ts.npy"))
    sub = np.load(os.path.join(args.unlabeled_dir, "sub.npy"))
    
    print(f"Unlabeled data loaded:")
    print(f"  y shape: {y.shape}")
    print(f"  d shape: {d.shape}")
    print(f"  ts shape: {ts.shape}")
    print(f"  sub shape: {sub.shape}")
    print(f"  ts range: [{ts.min():.1f}, {ts.max():.1f}]")
    print(f"  d range: [{d.min()}, {d.max()}]")
    print(f"  sub range: [{sub.min()}, {sub.max()}]")
    
    labeled_ts = np.load(os.path.join(args.labeled_dir, "ts.npy"))
    labeled_sub = np.load(os.path.join(args.labeled_dir, "sub.npy"))
    labeled_d = np.load(os.path.join(args.labeled_dir, "d.npy"))
    
    print(f"\nLabeled data loaded:")
    print(f"  ts shape: {labeled_ts.shape}")
    print(f"  sub shape: {labeled_sub.shape}")
    print(f"  d shape: {labeled_d.shape}")
    
    meta_path = os.path.join(args.labeled_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        labeled_meta = json.load(f)
    
    domain_offset_map = {int(k): v for k, v in labeled_meta["domain_offset_map"].items()}
    domains_per_subject_map = {int(k): v for k, v in labeled_meta["domains_per_subject_map"].items()}
    
    return y, d, ts, sub, labeled_ts, labeled_sub, labeled_d, domain_offset_map, domains_per_subject_map


def check_dimensions(y, d, ts, sub):
    """检查所有文件的维度一致性"""
    print("\n" + "=" * 60)
    print("Step 2: Dimension check...")
    print("=" * 60)
    
    n_samples = y.shape[0]
    
    checks = [
        ("y samples == d samples", n_samples == d.shape[0]),
        ("y samples == ts samples", n_samples == ts.shape[0]),
        ("y samples == sub samples", n_samples == sub.shape[0]),
        ("y labels == 11", y.shape[1] == 11),
        ("d is 1D", d.ndim == 1),
        ("ts is 1D", ts.ndim == 1),
        ("sub is 1D", sub.ndim == 1),
    ]
    
    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}: {result}")
        if not result:
            all_pass = False
    
    return all_pass


def check_data_types(y, d, ts, sub):
    """检查数据类型是否正确"""
    print("\n" + "=" * 60)
    print("Step 3: Data type check...")
    print("=" * 60)
    
    checks = [
        ("y dtype == float32", y.dtype == np.float32),
        ("d dtype == int64", d.dtype == np.int64),
        ("ts dtype == float32", ts.dtype == np.float32),
        ("sub dtype == int64", sub.dtype == np.int64),
    ]
    
    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}: {result}")
        if not result:
            all_pass = False
    
    return all_pass


def check_data_quality(y, d, ts, sub):
    """检查数据质量"""
    print("\n" + "=" * 60)
    print("Step 4: Data quality check...")
    print("=" * 60)
    
    checks = [
        ("y has NaN", np.any(np.isnan(y))),
        ("y has Inf", np.any(np.isinf(y))),
        ("d has NaN", np.any(np.isnan(d))),
        ("d has Inf", np.any(np.isinf(d))),
        ("ts has NaN", np.any(np.isnan(ts))),
        ("ts has Inf", np.any(np.isinf(ts))),
        ("sub has NaN", np.any(np.isnan(sub))),
        ("sub has Inf", np.any(np.isinf(sub))),
    ]
    
    all_good = True
    for name, has_issue in checks:
        status = "OK" if not has_issue else "PROBLEM"
        print(f"  [{status}] {name}: {has_issue}")
        if has_issue:
            all_good = False
    
    print(f"\ny statistics (should be ~1-5):")
    print(f"  min: {y.min():.2f}, max: {y.max():.2f}")
    print(f"  mean: {np.mean(y, axis=0)}")
    
    print(f"\nsub statistics (should be 1-15):")
    print(f"  unique values: {np.unique(sub)}")
    
    return all_good


def check_x_quality(args, n_samples):
    """使用mmap检查X数据质量"""
    print("\n" + "=" * 60)
    print("Step 5: X data quality check (mmap mode)...")
    print("=" * 60)
    
    x_path = os.path.join(args.unlabeled_dir, "X.npy")
    X = np.load(x_path, mmap_mode='r')
    
    print(f"  X shape: {X.shape}")
    
    if X.shape[0] != n_samples:
        print(f"  [FAIL] X samples ({X.shape[0]}) != y samples ({n_samples})")
        return False
    
    if X.shape[1] != 2000 or X.shape[2] != 59:
        print(f"  [FAIL] X dimensions wrong: {X.shape[1]}x{X.shape[2]} != 2000x59")
        return False
    
    if X.dtype != np.float32:
        print(f"  [FAIL] X dtype {X.dtype} != float32")
        return False
    
    n_sample_check = max(100, int(n_samples * args.sample_fraction))
    indices = np.random.choice(n_samples, n_sample_check, replace=False)
    
    has_nan = False
    has_inf = False
    sample_values = []
    
    for i in indices:
        row = X[i]
        if np.any(np.isnan(row)):
            has_nan = True
        if np.any(np.isinf(row)):
            has_inf = True
        sample_values.append(row.flatten())
    
    sample_values = np.concatenate(sample_values)
    
    print(f"  [{'OK' if not has_nan else 'PROBLEM'}] X has NaN: {has_nan}")
    print(f"  [{'OK' if not has_inf else 'PROBLEM'}] X has Inf: {has_inf}")
    
    print(f"\n  X sample statistics (from {n_sample_check} samples):")
    print(f"    min: {sample_values.min():.6f}")
    print(f"    max: {sample_values.max():.6f}")
    print(f"    mean: {sample_values.mean():.6f}")
    print(f"    std: {sample_values.std():.6f}")
    
    return not (has_nan or has_inf)


def check_domain_mapping(sub, d, domain_offset_map, domains_per_subject_map):
    """验证域ID与被试编号的映射是否正确"""
    print("\n" + "=" * 60)
    print("Step 6: Domain mapping validation...")
    print("=" * 60)
    
    all_valid = True
    
    for sub_id in sorted(np.unique(sub)):
        mask = sub == sub_id
        d_values = d[mask]
        
        expected_offset = domain_offset_map.get(sub_id, -1)
        expected_domains = domains_per_subject_map.get(sub_id, 0)
        
        if expected_offset < 0:
            print(f"  [WARNING] sub-{sub_id} not found in domain_offset_map")
            all_valid = False
            continue
        
        expected_min_d = expected_offset
        expected_max_d = expected_offset + expected_domains - 1
        
        actual_min_d = d_values.min()
        actual_max_d = d_values.max()
        
        is_valid = (actual_min_d >= expected_min_d) and (actual_max_d <= expected_max_d)
        
        status = "OK" if is_valid else "PROBLEM"
        print(f"  [{status}] sub-{sub_id}: domain range [{actual_min_d}, {actual_max_d}], "
              f"expected [{expected_min_d}, {expected_max_d}]")
        
        if not is_valid:
            all_valid = False
    
    return all_valid


def check_timestamp_coverage(ts, fs=200, window_size=10, step_size=5):
    """检查时间戳覆盖范围"""
    print("\n" + "=" * 60)
    print("Step 7: Timestamp coverage analysis...")
    print("=" * 60)
    
    print(f"  Timestamp statistics:")
    print(f"    min: {ts.min():.1f}s")
    print(f"    max: {ts.max():.1f}s")
    print(f"    unique timestamps: {len(np.unique(ts))}/{len(ts)}")
    
    ts_sorted = np.sort(ts)
    diffs = np.diff(ts_sorted)
    positive_diffs = diffs[diffs > 0]
    
    if len(positive_diffs) > 0:
        print(f"\n  Positive interval statistics (expected ~{step_size}s):")
        print(f"    mean: {np.mean(positive_diffs):.3f}s")
        print(f"    median: {np.median(positive_diffs):.3f}s")
        print(f"    min: {np.min(positive_diffs):.3f}s")
        print(f"    max: {np.max(positive_diffs):.3f}s")
    
    hours = ts // 3600
    unique_hours = np.unique(hours)
    print(f"\n  Samples distribution by hour:")
    for hour in sorted(unique_hours):
        count = np.sum(hours == hour)
        print(f"    Hour {int(hour)}: {count} samples")
    
    too_early = np.sum(ts < window_size)
    print(f"\n  Samples with timestamp < window_size ({window_size}s): {too_early}")
    
    return True


def load_session_onsets(args, domain_offset_map):
    """加载每个session的问卷onset"""
    domain_onset_map = {}
    
    for sub_idx in range(1, 16):
        sub_str = f"sub-{sub_idx:02d}"
        eeg_deriv_root = os.path.join(args.data_root, "derivatives", "eeg-prep", sub_str)
        
        if not os.path.exists(eeg_deriv_root):
            continue
        
        ses_dirs = sorted(glob.glob(os.path.join(eeg_deriv_root, "ses-*")))
        
        for ses_dir in ses_dirs:
            ses_id = os.path.basename(ses_dir)
            
            try:
                day_id = int(ses_id.replace("ses-", "")) - 1
                offset = domain_offset_map.get(sub_idx, 0)
                domain_id = offset + day_id
            except ValueError:
                continue
            
            beh_pattern = os.path.join(args.data_root, sub_str, ses_id, "beh", 
                                      f"{sub_str}_{ses_id}_task-dailylife_emotion_beh*.tsv")
            beh_files = sorted(glob.glob(beh_pattern))
            
            onsets = []
            if len(beh_files) > 0:
                df_list = []
                for bf in beh_files:
                    df_part = pd.read_csv(bf, sep="\t")
                    df_list.append(df_part)
                df = pd.concat(df_list, ignore_index=True)
                
                if "onset" in df.columns and len(df) > 0:
                    onsets = df["onset"].values.tolist()
            
            domain_onset_map[domain_id] = onsets
    
    return domain_onset_map


def check_labeled_zone_overlap(args, ts, sub, d, domain_offset_map):
    """
    精确检查未标注数据是否落在问卷前120秒区间内。
    修正：将条件从 t <= onset 改为 t < onset
    原因：窗口起始时间为onset时，窗口范围是[onset, onset+10)，完全不在问卷前120秒内
    """
    print("\n" + "=" * 60)
    print("Step 8: Precise overlap check with labeled zones...")
    print("=" * 60)
    
    domain_onset_map = load_session_onsets(args, domain_offset_map)
    print(f"  Loaded onsets for {len(domain_onset_map)} domains/sessions")
    
    in_labeled_zone_count = 0
    problematic_samples = []
    
    for domain_id in sorted(np.unique(d)):
        mask = d == domain_id
        domain_ts = ts[mask]
        domain_sub = sub[mask][0] if len(sub[mask]) > 0 else -1
        
        onsets = domain_onset_map.get(domain_id, [])
        
        for t in domain_ts:
            is_in_zone = False
            for onset in onsets:
                # 修正：将条件从 t <= onset 改为 t < onset
                # 窗口起始时间为onset时，窗口范围是[onset, onset+10)，不在问卷前120秒内
                if (onset - args.label_window) <= t < onset:
                    is_in_zone = True
                    break
            
            if is_in_zone:
                in_labeled_zone_count += 1
                if len(problematic_samples) < 10:
                    problematic_samples.append((domain_sub, domain_id, t))
    
    print(f"\n  Total unlabeled samples: {len(ts)}")
    print(f"  Samples in labeled zones (questionnaire -120s): {in_labeled_zone_count}")
    
    if in_labeled_zone_count > 0:
        print(f"\n  First 10 problematic samples:")
        for sid, did, t in problematic_samples:
            print(f"    sub-{sid}, domain-{did}, ts={t:.1f}s")
    
    if in_labeled_zone_count == 0:
        print(f"  [PASS] No samples in labeled zones")
        return True
    else:
        print(f"  [WARNING] {in_labeled_zone_count} samples are in labeled zones!")
        return False


def check_domain_coverage(d, domain_offset_map, domains_per_subject_map):
    """检查域覆盖情况"""
    print("\n" + "=" * 60)
    print("Step 9: Domain coverage analysis...")
    print("=" * 60)
    
    unique_domains = np.unique(d)
    print(f"  Total unique domains: {len(unique_domains)}")
    print(f"  Domain range: [{unique_domains.min()}, {unique_domains.max()}]")
    
    print("\n  Domain coverage by subject:")
    for sub_id in sorted(domain_offset_map.keys()):
        offset = domain_offset_map[sub_id]
        n_domains = domains_per_subject_map[sub_id]
        
        expected_domains = set(range(offset, offset + n_domains))
        actual_domains = set(unique_domains) & expected_domains
        
        coverage = len(actual_domains) / n_domains * 100
        missing = expected_domains - actual_domains
        
        status = "OK" if coverage > 0 else "EMPTY"
        print(f"  [{status}] sub-{sub_id:02d}: {len(actual_domains)}/{n_domains} domains covered ({coverage:.1f}%)")
        if missing:
            print(f"              Missing domains: {sorted(missing)}")
    
    return True


def main():
    args = parse_args()
    
    y, d, ts, sub, labeled_ts, labeled_sub, labeled_d, domain_offset_map, domains_per_subject_map = load_data(args)
    
    dim_ok = check_dimensions(y, d, ts, sub)
    type_ok = check_data_types(y, d, ts, sub)
    quality_ok = check_data_quality(y, d, ts, sub)
    x_ok = check_x_quality(args, len(ts))
    domain_map_ok = check_domain_mapping(sub, d, domain_offset_map, domains_per_subject_map)
    coverage_ok = check_timestamp_coverage(ts, args.fs, args.window_size, args.step_size)
    zone_ok = check_labeled_zone_overlap(args, ts, sub, d, domain_offset_map)
    domain_coverage_ok = check_domain_coverage(d, domain_offset_map, domains_per_subject_map)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_checks = {
        "Dimensions": dim_ok,
        "Data Types": type_ok,
        "Data Quality": quality_ok,
        "X Data Quality": x_ok,
        "Domain Mapping": domain_map_ok,
        "Timestamp Coverage": coverage_ok,
        "Labeled Zone Overlap": zone_ok,
    }
    
    passed = sum(all_checks.values())
    total = len(all_checks)
    
    print(f"\n  Checks passed: {passed}/{total}")
    
    for name, result in all_checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {name}")
    
    if passed == total:
        print("\n  [ALL CHECKS PASSED] Unlabeled data looks valid!")
    else:
        print("\n  [SOME CHECKS FAILED] Please review the warnings above.")


if __name__ == "__main__":
    main()