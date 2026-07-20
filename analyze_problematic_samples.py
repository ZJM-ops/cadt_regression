#!/usr/bin/env python3
"""
分析问题样本产生的原因：为什么有些样本会落在问卷前120秒区间内？
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze problematic samples")
    parser.add_argument(
        "--unlabeled-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_raw_unlabeled",
        help="未标注数据目录",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/vePFS-0x0d/DailyEEG",
        help="DailyEEG数据集根路径",
    )
    parser.add_argument("--label-window", type=int, default=120, help="问卷前多少秒为已标注区间")
    parser.add_argument("--labeled-meta",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/meta.json",
        help="已标注数据的meta.json")
    return parser.parse_args()


def load_domain_offset_map(labeled_meta_path: str) -> dict[int, int]:
    """加载domain_offset_map"""
    with open(labeled_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {int(k): v for k, v in meta["domain_offset_map"].items()}


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


def analyze_problematic_samples(args):
    """分析问题样本"""
    # 加载数据
    ts = np.load(os.path.join(args.unlabeled_dir, "ts.npy"))
    d = np.load(os.path.join(args.unlabeled_dir, "d.npy"))
    sub = np.load(os.path.join(args.unlabeled_dir, "sub.npy"))
    
    # 加载域配置和问卷onset
    domain_offset_map = load_domain_offset_map(args.labeled_meta)
    domain_onset_map = load_session_onsets(args, domain_offset_map)
    
    print("=" * 80)
    print("分析问题样本产生的原因")
    print("=" * 80)
    
    # 找出问题样本
    problematic_info = []
    
    for domain_id in sorted(np.unique(d)):
        mask = d == domain_id
        domain_ts = ts[mask]
        domain_sub = sub[mask][0] if len(sub[mask]) > 0 else -1
        
        onsets = domain_onset_map.get(domain_id, [])
        
        for t in domain_ts:
            for onset in onsets:
                if (onset - args.label_window) <= t <= onset:
                    # 计算距离问卷区间的距离
                    dist_to_start = t - (onset - args.label_window)
                    dist_to_end = onset - t
                    
                    problematic_info.append({
                        "sub": domain_sub,
                        "domain": domain_id,
                        "ts": t,
                        "onset": onset,
                        "label_window_start": onset - args.label_window,
                        "dist_to_start": dist_to_start,
                        "dist_to_end": dist_to_end,
                        "position": "start" if dist_to_start < dist_to_end else "end"
                    })
                    break
    
    print(f"\n总共找到 {len(problematic_info)} 个问题样本")
    
    if len(problematic_info) == 0:
        print("没有问题样本")
        return
    
    # 分析问题样本的位置分布
    start_count = sum(1 for p in problematic_info if p["position"] == "start")
    end_count = sum(1 for p in problematic_info if p["position"] == "end")
    
    print(f"\n位置分布：")
    print(f"  靠近区间起点（onset-120）: {start_count} 个")
    print(f"  靠近区间终点（onset）: {end_count} 个")
    
    # 分析距离分布
    distances = [min(p["dist_to_start"], p["dist_to_end"]) for p in problematic_info]
    print(f"\n距离区间边界的距离分布：")
    print(f"  最小值: {min(distances):.6f}s")
    print(f"  最大值: {max(distances):.6f}s")
    print(f"  平均值: {np.mean(distances):.6f}s")
    print(f"  中位数: {np.median(distances):.6f}s")
    
    # 按距离分组统计
    bins = [0, 0.001, 0.01, 0.1, 1, 5, 10, 60, 120]
    counts, _ = np.histogram(distances, bins=bins)
    
    print(f"\n距离区间边界的距离分组统计：")
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.3f}s ~ {bins[i+1]:.3f}s: {counts[i]} 个")
    
    # 显示一些典型的问题样本
    print(f"\n典型问题样本示例（按距离排序）：")
    problematic_info.sort(key=lambda x: min(x["dist_to_start"], x["dist_to_end"]))
    
    for p in problematic_info[:20]:
        print(f"  sub-{p['sub']}, domain-{p['domain']}, ts={p['ts']:.6f}s, "
              f"onset={p['onset']:.6f}s, "
              f"区间=[{p['label_window_start']:.6f}, {p['onset']:.6f}], "
              f"距离边界={min(p['dist_to_start'], p['dist_to_end']):.6f}s")
    
    # 分析是否存在系统性问题
    print(f"\n" + "=" * 80)
    print("问题原因分析")
    print("=" * 80)
    
    # 检查是否有样本恰好落在边界上
    on_boundary_count = sum(1 for p in problematic_info 
                            if abs(p["dist_to_start"]) < 1e-6 or abs(p["dist_to_end"]) < 1e-6)
    print(f"\n1. 恰好落在边界上的样本数: {on_boundary_count}")
    
    # 检查是否有样本非常接近边界（<0.01秒）
    near_boundary_count = sum(1 for p in problematic_info 
                              if min(p["dist_to_start"], p["dist_to_end"]) < 0.01)
    print(f"2. 非常接近边界的样本数（<0.01秒）: {near_boundary_count}")
    
    # 检查是否有样本距离问卷onset很近（<1秒）
    near_onset_count = sum(1 for p in problematic_info if p["dist_to_end"] < 1)
    print(f"3. 距离问卷onset很近的样本数（<1秒）: {near_onset_count}")
    
    # 分析可能的原因
    print(f"\n可能的原因：")
    
    if near_boundary_count > len(problematic_info) * 0.5:
        print("  [可能] 时间戳精度问题：处理脚本和验证脚本使用不同的精度计算")
        print("         例如：处理脚本使用整数采样点计算，验证脚本使用浮点数比较")
    
    if near_onset_count > len(problematic_info) * 0.3:
        print("  [可能] 边界条件处理不当：处理脚本可能没有正确排除问卷前120秒的全部区间")
    
    # 检查是否存在某个被试或域集中了大量问题样本
    sub_counts = {}
    domain_counts = {}
    
    for p in problematic_info:
        sub_counts[p["sub"]] = sub_counts.get(p["sub"], 0) + 1
        domain_counts[p["domain"]] = domain_counts.get(p["domain"], 0) + 1
    
    print(f"\n问题样本按被试分布：")
    for sid in sorted(sub_counts.keys()):
        print(f"  sub-{sid}: {sub_counts[sid]} 个")
    
    print(f"\n问题样本按域分布（前10个）：")
    for did, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  domain-{did}: {count} 个")


if __name__ == "__main__":
    args = parse_args()
    analyze_problematic_samples(args)