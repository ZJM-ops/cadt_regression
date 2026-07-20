#!/usr/bin/env python3
"""
将 test_load.py 生成的 H5 特征文件转换为 CADT 模型可直接加载的 NPY 格式。

使用 raw_eeg_10s_window（原始 10 秒 EEG 波形），不使用 DE 特征。
输出与 CADT 原始数据格式对齐：
  - X.npy: (N, seq_len, n_channels)  float32
  - y.npy: (N, n_labels)             float32  11 维情绪回归标签
  - d.npy: (N,)                      int64    联合域 ID = (sub_id-1)*10 + day_id
  - ts.npy: (N,)                     float32  每段的起始时间戳（相对于session开始的秒数）
  - sub.npy: (N,)                    int64    被试编号，便于留一被试划分
"""

import argparse
import glob
import json
import os
from typing import Iterator, Optional, Tuple

import h5py
import numpy as np

LABEL_NAMES = [
    "inspired", "alert", "determined", "attentive", "active",
    "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align DailyEEG H5 data to CADT input format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/de_output_addtime_new",
        help="test_load.py 输出的 H5 目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new",
        help="CADT 对齐后的 NPY 保存目录",
    )
    parser.add_argument("--seq-len", type=int, default=2000, help="CADT 时序长度，2000=10秒@200Hz")
    parser.add_argument("--n-channels", type=int, default=59, help="EEG 通道数")
    parser.add_argument("--n-labels", type=int, default=11, help="回归标签维度")
    parser.add_argument(
        "--domains-per-subject",
        type=int,
        default=10,
        help="每个被试最多 session 数，用于计算联合域 ID",
    )
    parser.add_argument(
        "--use-all-segments",
        action="store_true",
        default=True,
        help="是否使用完整10秒数据(2000点)，否则使用seq_len长度的片段",
    )
    return parser.parse_args()


def iter_h5_samples(h5_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray, int, int, float]]:
    """
    遍历单个 H5 文件中的所有样本。

    @param {str} h5_path - H5 文件路径
    @yields {(raw_eeg, label, sub_id, day_id, start_sec)} 原始 EEG、标签、被试 ID、日/session ID、时间戳
    """
    with h5py.File(h5_path, "r") as hf:
        trial_keys = sorted(hf.keys(), key=lambda k: int(k.replace("trial", "")))
        for trial_key in trial_keys:
            trial_grp = hf[trial_key]
            sample_keys = sorted(trial_grp.keys(), key=lambda k: int(k.replace("sample", "")))
            for sample_key in sample_keys:
                sample_grp = trial_grp[sample_key]
                if "raw_eeg_10s_window" not in sample_grp:
                    continue
                raw_eeg = sample_grp["raw_eeg_10s_window"][()]
                label = np.asarray(sample_grp.attrs["label"], dtype=np.float32)
                sub_id = int(sample_grp.attrs["sub_id"])
                day_id = int(sample_grp.attrs["day_id"])
                # 获取时间戳（相对于session开始的秒数），默认为0（兼容旧版H5文件）
                start_sec = float(sample_grp.attrs.get("start_sec", 0.0))
                yield raw_eeg, label, sub_id, day_id, start_sec


def raw_window_to_cadt_segments(
    raw_eeg: np.ndarray,
    seq_len: int,
    n_channels: int,
    use_all: bool = False,
) -> list[np.ndarray]:
    """
    从 10s 原始 EEG 窗口提取 CADT 输入片段。

    @param {np.ndarray} raw_eeg - 形状 (n_channels, n_timepoints)，10秒原始EEG
    @param {int} seq_len - 目标时间步数
    @param {int} n_channels - 目标通道数
    @param {bool} use_all - 是否使用全部时间点，True 返回完整2000点，False 返回seq_len长度的片段
    @returns {list[np.ndarray]} 形状 (seq_len, n_channels) 的片段列表
    """
    segments = []
    
    if raw_eeg.ndim != 2:
        return segments

    n_ch, n_time = raw_eeg.shape
    if n_ch != n_channels:
        return segments

    if use_all:
        segment = raw_eeg.astype(np.float32)
        if np.isfinite(segment).all():
            segments.append(segment.T)
    else:
        if n_time < seq_len:
            return segments
        
        start = n_time - seq_len
        segment = raw_eeg[:, start:start+seq_len].astype(np.float32)
        if np.isfinite(segment).all():
            segments.append(segment.T)

    return segments


def build_domain_id(sub_id: int, day_id: int, domains_per_subject: int) -> int:
    """
    计算 CADT 联合域 ID。

    @param {int} sub_id - 被试编号（1-based）
    @param {int} day_id - session/日编号（0-based）
    @param {int} domains_per_subject - 每个被试的域槽位数
    @returns {int} 联合域 ID
    """
    return (sub_id - 1) * domains_per_subject + day_id


def compute_domain_offset_map(
    h5_files: list[str]
) -> tuple[dict[int, int], dict[int, int]]:
    """
    动态计算每个被试的域起始偏移和域槽位数。
    
    使用累积偏移方式确保域ID连续且不重叠：
    - 先统计每个被试拥有的最大 day_id
    - 按被试编号排序后累加得到起始偏移
    - domain_id = domain_offset_map[sub_id] + day_id
    
    例如：
    - sub-12: domains_per_subject=10, offset=110 → 域范围 110-119
    - sub-13: domains_per_subject=11, offset=120 → 域范围 120-130
    - sub-14: domains_per_subject=10, offset=131 → 域范围 131-140

    @param {list[str]} h5_files - H5 文件路径列表
    @returns {tuple[dict, dict]} (domain_offset_map, domains_per_subject_map)
        domain_offset_map: key=sub_id, value=该被试的域起始偏移
        domains_per_subject_map: key=sub_id, value=该被试的域槽位数
    """
    sub_day_map: dict[int, set[int]] = {}  # sub_id -> set of day_ids

    for h5_path in h5_files:
        for _, _, sub_id, day_id, _ in iter_h5_samples(h5_path):
            if sub_id not in sub_day_map:
                sub_day_map[sub_id] = set()
            sub_day_map[sub_id].add(day_id)

    # 计算每个被试的域槽位数 = max_day_id + 1
    domains_per_subject_map: dict[int, int] = {}
    for sub_id, day_ids in sub_day_map.items():
        max_day = max(day_ids) if day_ids else 0
        domains_per_subject_map[sub_id] = max_day + 1

    # 按被试编号排序，计算累积偏移
    sorted_sub_ids = sorted(domains_per_subject_map.keys())
    domain_offset_map: dict[int, int] = {}
    current_offset = 0
    for sub_id in sorted_sub_ids:
        domain_offset_map[sub_id] = current_offset
        current_offset += domains_per_subject_map[sub_id]

    return domain_offset_map, domains_per_subject_map


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "sub_*_de.h5")))
    if not h5_files:
        raise FileNotFoundError(f"未在 {args.input_dir} 找到 sub_*_de.h5 文件")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    d_list: list[int] = []
    sub_list: list[int] = []
    ts_list: list[float] = []  # 新增：时间戳列表（相对于session开始的秒数）
    skipped = 0

    # 第一步：动态计算每个被试的域起始偏移和域槽位数（基于实际session数量）
    domain_offset_map, domains_per_subject_map = compute_domain_offset_map(h5_files)
    print(f"\n动态域配置:")
    for sub_id in sorted(domain_offset_map.keys()):
        start = domain_offset_map[sub_id]
        end = start + domains_per_subject_map[sub_id] - 1
        print(f"  sub-{sub_id:02d}: domains_per_subject={domains_per_subject_map[sub_id]}, 域范围 [{start}, {end}]")

    # 第二步：遍历生成样本
    for h5_path in h5_files:
        for raw_eeg, label, sub_id, day_id, start_sec in iter_h5_samples(h5_path):
            if label.shape[0] != args.n_labels:
                skipped += 1
                continue

            segments = raw_window_to_cadt_segments(
                raw_eeg,
                seq_len=args.seq_len,
                n_channels=args.n_channels,
                use_all=args.use_all_segments,
            )
            if not segments:
                skipped += 1
                continue

            # 使用动态的域起始偏移
            offset = domain_offset_map.get(sub_id, 0)
            domain_id = offset + day_id
            for segment in segments:
                x_list.append(segment)
                y_list.append(label)
                d_list.append(domain_id)
                sub_list.append(sub_id)
                ts_list.append(start_sec)  # 新增：保存时间戳

    if not x_list:
        raise RuntimeError("没有有效样本，请检查 H5 文件与参数配置")

    x_arr = np.stack(x_list, axis=0).astype(np.float32)
    y_arr = np.stack(y_list, axis=0).astype(np.float32)
    d_arr = np.asarray(d_list, dtype=np.int64)
    sub_arr = np.asarray(sub_list, dtype=np.int64)
    ts_arr = np.asarray(ts_list, dtype=np.float32)  # 新增：时间戳数组（相对于session开始的秒数）

    np.save(os.path.join(args.output_dir, "X.npy"), x_arr)
    np.save(os.path.join(args.output_dir, "y.npy"), y_arr)
    np.save(os.path.join(args.output_dir, "d.npy"), d_arr)
    np.save(os.path.join(args.output_dir, "sub.npy"), sub_arr)
    np.save(os.path.join(args.output_dir, "ts.npy"), ts_arr)  # 新增：保存时间戳

    meta = {
        "seq_len": args.seq_len,
        "n_channels": args.n_channels,
        "n_labels": args.n_labels,
        "label_names": LABEL_NAMES,
        "use_all_segments": args.use_all_segments,
        "domains_per_subject": args.domains_per_subject,  # 保留默认值
        "domain_offset_map": {str(k): v for k, v in domain_offset_map.items()},  # 新增：域起始偏移
        "domains_per_subject_map": {str(k): v for k, v in domains_per_subject_map.items()},  # 新增：动态配置
        "num_samples": int(x_arr.shape[0]),
        "num_subjects": int(len(np.unique(sub_arr))),
        "num_domains": int(len(np.unique(d_arr))),
        "skipped_samples": skipped,
        "x_shape": list(x_arr.shape),
        "y_shape": list(y_arr.shape),
        "ts_shape": list(ts_arr.shape),  # 新增：时间戳维度
        "ts_range": [float(ts_arr.min()), float(ts_arr.max())],  # 新增：时间戳范围
        "source_h5_files": [os.path.basename(p) for p in h5_files],
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("DailyEEG -> CADT 对齐完成")
    print(f"  输入 H5: {len(h5_files)} 个文件")
    print(f"  使用全部切片: {args.use_all_segments}")
    print(f"  有效样本: {x_arr.shape[0]}，跳过: {skipped}")
    print(f"  X shape: {x_arr.shape}  (N, seq_len, n_channels)")
    print(f"  y shape: {y_arr.shape}  (N, n_labels)")
    print(f"  d shape: {d_arr.shape}  域 ID 范围 [{d_arr.min()}, {d_arr.max()}]")
    print(f"  ts shape: {ts_arr.shape}  时间戳范围 [{ts_arr.min():.1f}, {ts_arr.max():.1f}] 秒")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
