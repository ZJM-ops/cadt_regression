#!/usr/bin/env python3
"""
处理 DailyEEG 原始数据中未打标签的部分，使用 10s 窗口、5s 滑动步长截取。

优化方案：先排除已打标签的区间（问卷前120秒），再对剩余部分进行切分。
精确时间戳检查作为额外保障，处理极端情况下超出120秒范围的已标注样本。

输出与 CADT 对齐的 NPY 格式（与已标注数据格式一致）：
  - X.npy: (N, seq_len, n_channels)  float32
  - y.npy: (N, n_labels)             float32  11 维情绪回归标签
  - d.npy: (N,)                      int64    联合域 ID = domain_offset_map[sub_id] + day_id
  - ts.npy: (N,)                     float32  每段的起始时间戳（相对于session开始的秒数）
  - sub.npy: (N,)                    int64    被试编号，便于留一被试划分
"""

import argparse
import glob
import json
import os
from typing import Tuple, List

import mne
import numpy as np
import pandas as pd

LABEL_NAMES = [
    "inspired", "alert", "determined", "attentive", "active",
    "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process unlabeled raw EEG data with sliding window")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/vePFS-0x0d/DailyEEG",
        help="DailyEEG 数据集根路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_raw_unlabeled",
        help="NPY 保存目录",
    )
    parser.add_argument(
        "--labeled-data-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new",
        help="已标注数据目录（包含 ts.npy, sub.npy），用于精确排除已使用的样本",
    )
    parser.add_argument("--seq-len", type=int, default=2000, help="时序长度，2000=10秒@200Hz")
    parser.add_argument("--n-channels", type=int, default=59, help="EEG 通道数")
    parser.add_argument("--n-labels", type=int, default=11, help="回归标签维度")
    parser.add_argument("--window-size", type=int, default=10, help="滑动窗口大小（秒）")
    parser.add_argument("--step-size", type=int, default=5, help="滑动步长（秒）")
    parser.add_argument("--fs", type=int, default=200, help="采样率")
    parser.add_argument("--label-window", type=int, default=120, help="已打标签的窗口长度（秒）")
    return parser.parse_args()


def compute_domain_offset_map(data_root: str) -> tuple[dict[int, int], dict[int, int]]:
    """
    动态计算每个被试的域起始偏移和域槽位数（与align_cadt_input.py一致）。
    
    使用累积偏移方式确保域ID连续且不重叠：
    - 遍历每个被试的session目录，从目录名提取day_id
    - 按被试编号排序后累加得到起始偏移
    - domain_id = domain_offset_map[sub_id] + day_id
    
    例如：
    - sub-12: domains_per_subject=10, offset=110 → 域范围 110-119
    - sub-13: domains_per_subject=11, offset=120 → 域范围 120-130
    - sub-14: domains_per_subject=10, offset=131 → 域范围 131-140

    @param {str} data_root - DailyEEG数据集根路径
    @returns {tuple[dict, dict]} (domain_offset_map, domains_per_subject_map)
    """
    sub_day_map: dict[int, set[int]] = {}  # sub_id -> set of day_ids

    for sub_idx in range(1, 16):
        sub_str = f"sub-{sub_idx:02d}"
        eeg_deriv_root = os.path.join(data_root, "derivatives", "eeg-prep", sub_str)
        
        if not os.path.exists(eeg_deriv_root):
            continue
        
        ses_dirs = sorted(glob.glob(os.path.join(eeg_deriv_root, "ses-*")))
        
        for ses_dir in ses_dirs:
            ses_id = os.path.basename(ses_dir)
            # 从session目录名提取day_id（ses-01 -> 0, ses-10 -> 9）
            try:
                day_id = int(ses_id.replace("ses-", "")) - 1
                if sub_idx not in sub_day_map:
                    sub_day_map[sub_idx] = set()
                sub_day_map[sub_idx].add(day_id)
            except ValueError:
                print(f"警告：无法解析session目录名 {ses_id}，跳过")
                continue

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


def _extract_segment_number(fname):
    """从文件名提取段号（用于处理分段数据）"""
    basename = os.path.basename(fname)
    basename_no_ext = basename.split('.')[0]
    parts = basename_no_ext.split('-')
    for i in range(len(parts)-1, 0, -1):
        part = parts[i]
        prev_part = parts[i-1].lower()
        if part.isdigit() and ('eeg' in prev_part or 'beh' in prev_part):
            return int(part)
    return 0


def has_uniform_block(data: np.ndarray, block_len: int, atol: float = 1e-6) -> bool:
    if data is None or data.size == 0:
        return False
    if data.ndim != 2:
        return False
    chan_range = np.max(data, axis=1) - np.min(data, axis=1)
    flat_mask = chan_range < atol
    if block_len <= 1:
        return np.any(flat_mask)
    cnt = 0
    for flag in flat_mask:
        if flag:
            cnt += 1
            if cnt >= block_len:
                return True
        else:
            cnt = 0
    return False


def load_labeled_timestamps(labeled_data_dir: str) -> dict[int, set[float]]:
    """
    加载已标注数据的时间戳，建立 sub_id -> set(timestamp) 的映射。
    作为额外保障，处理极端情况下超出120秒范围的已标注样本。
    """
    ts_path = os.path.join(labeled_data_dir, "ts.npy")
    sub_path = os.path.join(labeled_data_dir, "sub.npy")
    
    if not os.path.exists(ts_path) or not os.path.exists(sub_path):
        print(f"警告：未找到已标注数据 {ts_path} 或 {sub_path}，将仅使用问卷前120秒区间排除")
        return {}
    
    labeled_ts = np.load(ts_path)
    labeled_sub = np.load(sub_path)
    
    sub_ts_map: dict[int, set[float]] = {}
    for sub_id, ts in zip(labeled_sub, labeled_ts):
        sub_id = int(sub_id)
        if sub_id not in sub_ts_map:
            sub_ts_map[sub_id] = set()
        sub_ts_map[sub_id].add(float(ts))
    
    print(f"已加载 {len(labeled_ts)} 个已标注样本时间戳，覆盖 {len(sub_ts_map)} 个被试")
    return sub_ts_map


def get_unlabeled_intervals(
    total_duration_sec: float,
    onset_arr: np.ndarray,
    label_window: int,
) -> List[Tuple[float, float]]:
    """
    计算未打标签的时间区间。
    
    主要逻辑：排除问卷前120秒的区间，返回剩余部分。
    
    已标注区间：[onset - 120, onset)  左闭右开
    原因：窗口起始时间为onset时，窗口范围是[onset, onset+10)，完全不在问卷前120秒内
    
    返回格式: [(start_sec, end_sec), ...]
    """
    if onset_arr is None or len(onset_arr) == 0:
        # 没有标签，整个数据都是未标注的
        return [(0.0, total_duration_sec)]
    
    # 1. 收集所有已打标签的区间（问卷前120秒）
    # 已标注区间：[onset - 120, onset)  左闭右开
    labeled_intervals = []
    for onset in onset_arr:
        label_start = max(0.0, onset - label_window)
        label_end = onset  # 不包含onset本身
        labeled_intervals.append((label_start, label_end))
    
    # 2. 合并重叠的区间
    labeled_intervals.sort(key=lambda x: x[0])
    
    merged = [labeled_intervals[0]]
    for current_start, current_end in labeled_intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start < last_end:
            # 重叠，合并（注意是<而不是<=，因为是左闭右开区间）
            new_end = max(last_end, current_end)
            merged[-1] = (last_start, new_end)
        else:
            merged.append((current_start, current_end))
    
    # 3. 计算未打标签的区间（总范围减去已打标签区间）
    unlabeled_intervals = []
    prev_end = 0.0
    
    for start, end in merged:
        if start > prev_end:
            unlabeled_intervals.append((prev_end, start))
        prev_end = max(prev_end, end)
    
    # 添加最后一个未标注区间
    if prev_end < total_duration_sec:
        unlabeled_intervals.append((prev_end, total_duration_sec))
    
    return unlabeled_intervals


def _load_session_data(
    sub_idx: int,
    ses_dir: str,
    data_root: str,
    fs: int,
    n_channels: int,
    label_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    加载单个session的数据，支持分段数据处理。
    
    返回: (raw_data, onset_arr, scores_all, total_duration_sec)
    """
    sub_str = f"sub-{sub_idx:02d}"
    ses_id = os.path.basename(ses_dir)
    beh_root = os.path.join(data_root, sub_str)
    
    fif_paths = sorted(glob.glob(os.path.join(ses_dir, "*_prep_*eeg*.fif")))
    if len(fif_paths) == 0:
        return None, None, None, 0.0
    
    label_cols = [
        "inspired", "alert", "determined", "attentive", "active",
        "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue"
    ]
    
    # 检测是否为分段数据
    has_segments = any(_extract_segment_number(fp) > 0 for fp in fif_paths)
    
    if has_segments:
        # 处理分段数据（与test_load.py的_process_segmented_session逻辑一致）
        print(f"  {ses_id}: Detected segmented data, processing with time offset...")
        
        fif_segments = {}
        all_meas_dates = []
        
        for fp in fif_paths:
            seg_num = _extract_segment_number(fp)
            if seg_num == 0:
                continue
            r = mne.io.read_raw_fif(fp, preload=True, verbose=False)
            sfreq = r.info["sfreq"]
            if not np.isclose(sfreq, fs):
                print(f"    Skip segment {seg_num}: sampling rate mismatch")
                continue
            meas_date = r.info["meas_date"]
            n_samples = r.get_data().shape[1]
            duration = n_samples / fs
            fif_segments[seg_num] = {
                'raw_data': r.get_data(),
                'meas_date': meas_date,
                'duration': duration,
                'n_samples': n_samples
            }
            all_meas_dates.append((seg_num, meas_date))
        
        if len(fif_segments) == 0:
            print(f"    No valid segments found")
            return None, None, None, 0.0
        
        # 按测量日期排序，计算时间偏移
        all_meas_dates.sort(key=lambda x: x[1])
        earliest_date = all_meas_dates[0][1]
        
        for seg_num, info in fif_segments.items():
            offset_sec = (info['meas_date'] - earliest_date).total_seconds()
            info['offset_sec'] = offset_sec
            info['end_sec'] = offset_sec + info['duration']
        
        # 加载行为数据，加上段偏移
        beh_pattern = os.path.join(beh_root, ses_id, "beh", f"{sub_str}_{ses_id}_task-dailylife_emotion_beh*.tsv")
        beh_files = sorted(glob.glob(beh_pattern))
        
        onset_arr = []
        scores_all = []
        
        for bf in beh_files:
            seg_num = _extract_segment_number(bf)
            if seg_num == 0 or seg_num not in fif_segments:
                continue
            df_part = pd.read_csv(bf, sep="\t")
            if len(df_part) == 0:
                continue
            
            offset_sec = fif_segments[seg_num]['offset_sec']
            for _, row in df_part.iterrows():
                onset_arr.append(row['onset'] + offset_sec)
                scores_all.append(row[label_cols].astype(np.float32).values)
        
        onset_arr = np.array(onset_arr) if onset_arr else None
        scores_all = np.array(scores_all) if scores_all else None
        
        # 合并所有段的EEG数据
        max_end_sec = max(info['end_sec'] for info in fif_segments.values())
        total_samples = int(max_end_sec * fs) + 1
        n_chans = list(fif_segments.values())[0]['raw_data'].shape[0]
        
        raw_data = np.full((n_chans, total_samples), np.nan)
        
        for seg_num, info in fif_segments.items():
            offset_samples = int(info['offset_sec'] * fs)
            seg_raw_data = info['raw_data']
            n_samples = seg_raw_data.shape[1]
            end_sample = min(offset_samples + n_samples, total_samples)
            raw_data[:, offset_samples:end_sample] = seg_raw_data[:, :end_sample - offset_samples]
        
        total_duration_sec = max_end_sec
        
    else:
        # 处理非分段数据
        raw_list = []
        for fp in fif_paths:
            r = mne.io.read_raw_fif(fp, preload=True, verbose=False)
            raw_list.append(r)
        raw = mne.concatenate_raws(raw_list)
        raw_data = raw.get_data()
        sfreq = raw.info["sfreq"]
        
        if not np.isclose(sfreq, fs):
            print(f"  {ses_id}: sampling rate {sfreq} != target fs {fs}, skip")
            return None, None, None, 0.0
        
        if raw_data.shape[0] != n_channels:
            print(f"  {ses_id}: channel count {raw_data.shape[0]} != {n_channels}, skip")
            return None, None, None, 0.0
        
        total_pts = raw_data.shape[1]
        total_duration_sec = total_pts / fs
        
        # 加载行为数据
        beh_pattern = os.path.join(beh_root, ses_id, "beh", f"{sub_str}_{ses_id}_task-dailylife_emotion_beh*.tsv")
        beh_files = sorted(glob.glob(beh_pattern))
        
        onset_arr = None
        scores_all = None
        
        if len(beh_files) > 0:
            df_list = []
            for bf in beh_files:
                df_part = pd.read_csv(bf, sep="\t")
                df_list.append(df_part)
            df = pd.concat(df_list, ignore_index=True)
            if len(df) > 0:
                scores_all = df[label_cols].astype(np.float32).values
                onset_arr = df["onset"].values
    
    return raw_data, onset_arr, scores_all, total_duration_sec


def count_session_samples(
    sub_idx: int,
    ses_dir: str,
    data_root: str,
    fs: int,
    window_size: int,
    step_size: int,
    n_channels: int,
    label_window: int,
    labeled_ts_map: dict[int, set[float]] = None,
) -> int:
    """统计单个 session 的有效未打标签样本数"""
    sub_str = f"sub-{sub_idx:02d}"
    ses_id = os.path.basename(ses_dir)
    
    raw_data, onset_arr, _, total_duration_sec = _load_session_data(
        sub_idx, ses_dir, data_root, fs, n_channels, label_window
    )
    
    if raw_data is None:
        return 0
    
    points_len = int(window_size * fs)
    points_step = int(step_size * fs)
    total_pts = raw_data.shape[1]
    
    # 获取已标注数据的时间戳集合（作为额外保障）
    labeled_ts_set = labeled_ts_map.get(sub_idx, set()) if labeled_ts_map else set()
    
    # 【优化】先计算未打标签的区间（基于问卷前120秒）
    unlabeled_intervals = get_unlabeled_intervals(total_duration_sec, onset_arr, label_window)
    
    count = 0
    flat_block_len = 5 * fs
    
    # 【优化】只在未打标签区间内切分
    for interval_start_sec, interval_end_sec in unlabeled_intervals:
        start_pt = int(interval_start_sec * fs)
        end_pt = int(interval_end_sec * fs)
        
        remainder = start_pt % points_step
        if remainder != 0:
            start_pt = start_pt + points_step - remainder
        
        available_pts = end_pt - start_pt
        if available_pts < points_len:
            continue
        
        n_windows_in_interval = int((available_pts - points_len) / points_step) + 1
        
        for win_idx in range(n_windows_in_interval):
            win_start_pt = start_pt + win_idx * points_step
            win_end_pt = win_start_pt + points_len
            
            if win_end_pt > total_pts:
                continue
            
            # 检查是否有NaN（分段数据可能存在间隙）
            window_data = raw_data[:, win_start_pt:win_end_pt]
            if np.any(np.isnan(window_data)):
                continue
            
            window_start_sec = win_start_pt / fs
            
            # 额外保障：检查时间戳是否已被已标注数据使用
            if labeled_ts_set:
                if any(abs(ts - window_start_sec) < 0.1 for ts in labeled_ts_set):
                    continue
            
            if has_uniform_block(window_data, block_len=flat_block_len):
                chan_range = np.max(window_data, axis=1) - np.min(window_data, axis=1)
                flat_mask = chan_range < 1e-6
                window_data = window_data[~flat_mask, :]
            
            if window_data.shape[1] < points_len:
                continue
            
            if not np.isfinite(window_data).all():
                continue
            
            count += 1
    
    return count


def process_session_to_arrays(
    sub_idx: int,
    ses_dir: str,
    data_root: str,
    fs: int,
    window_size: int,
    step_size: int,
    n_channels: int,
    n_labels: int,
    label_window: int,
    x_memmap: np.memmap,
    y_memmap: np.memmap,
    d_memmap: np.memmap,
    ts_memmap: np.memmap,
    sub_memmap: np.memmap,
    domain_offset_map: dict[int, int],
    start_idx: int,
    labeled_ts_map: dict[int, set[float]] = None,
) -> int:
    """处理单个 session 并写入 memmap，返回写入的样本数"""
    sub_str = f"sub-{sub_idx:02d}"
    ses_id = os.path.basename(ses_dir)
    
    # 加载session数据（支持分段数据处理）
    raw_data, onset_arr, scores_all, total_duration_sec = _load_session_data(
        sub_idx, ses_dir, data_root, fs, n_channels, label_window
    )
    
    if raw_data is None:
        print(f"  {ses_id}: No valid data, skip")
        return 0
    
    total_pts = raw_data.shape[1]
    points_len = int(window_size * fs)
    points_step = int(step_size * fs)
    
    if onset_arr is not None and len(onset_arr) > 0:
        print(f"  {ses_id}: Loaded {len(onset_arr)} emotion labels")
    
    # 获取已标注数据的时间戳集合（作为额外保障）
    labeled_ts_set = labeled_ts_map.get(sub_idx, set()) if labeled_ts_map else set()
    
    # 【优化】先计算未打标签的区间（基于问卷前120秒）
    unlabeled_intervals = get_unlabeled_intervals(total_duration_sec, onset_arr, label_window)
    
    # 从session目录名提取day_id（ses-01 -> 0, ses-10 -> 9）
    try:
        day_id = int(ses_id.replace("ses-", "")) - 1
    except ValueError:
        print(f"  {ses_id}: 无法解析session目录名，跳过")
        return 0
    
    # 计算域ID
    offset = domain_offset_map.get(sub_idx, 0)
    domain_id = offset + day_id
    
    print(f"  {ses_id}: {len(unlabeled_intervals)} unlabeled intervals, day_id={day_id}, domain_id={domain_id}")
    
    # 详细跳过统计
    skipped_invalid = 0           # 数据无效（过短、含NaN等）
    skipped_labeled_rough = 0     # 被120秒区间排除（粗略排除）
    skipped_labeled_exact = 0     # 被精确时间戳匹配排除（额外保障）
    written_count = 0
    flat_block_len = 5 * fs
    
    # 统计被120秒区间排除的窗口数（rough）
    if onset_arr is not None and len(onset_arr) > 0:
        # 计算所有已标注区间的总点数
        labeled_total_pts = 0
        for onset in onset_arr:
            label_start = max(0.0, onset - label_window)
            label_end = onset
            labeled_total_pts += int((label_end - label_start) * fs)
        # 估算被排除的窗口数
        skipped_labeled_rough = int(labeled_total_pts / points_step)
    
    # 【优化】只在未打标签区间内切分
    for interval_start_sec, interval_end_sec in unlabeled_intervals:
        start_pt = int(interval_start_sec * fs)
        end_pt = int(interval_end_sec * fs)
        
        remainder = start_pt % points_step
        if remainder != 0:
            start_pt = start_pt + points_step - remainder
        
        available_pts = end_pt - start_pt
        if available_pts < points_len:
            continue
        
        n_windows_in_interval = int((available_pts - points_len) / points_step) + 1
        
        for win_idx in range(n_windows_in_interval):
            win_start_pt = start_pt + win_idx * points_step
            win_end_pt = win_start_pt + points_len
            
            if win_end_pt > total_pts:
                continue
            
            window_start_sec = win_start_pt / fs
            
            # 额外保障：检查时间戳是否已被已标注数据使用
            # 这是"精确排除"，处理极端情况下超出120秒范围的已标注样本
            if labeled_ts_set:
                if any(abs(ts - window_start_sec) < 0.1 for ts in labeled_ts_set):
                    skipped_labeled_exact += 1
                    continue
            
            window_data = raw_data[:, win_start_pt:win_end_pt]
            
            if has_uniform_block(window_data, block_len=flat_block_len):
                chan_range = np.max(window_data, axis=1) - np.min(window_data, axis=1)
                flat_mask = chan_range < 1e-6
                window_data = window_data[~flat_mask, :]
            
            if window_data.shape[1] < points_len:
                skipped_invalid += 1
                continue
            
            if not np.isfinite(window_data).all():
                skipped_invalid += 1
                continue
            
            segment = window_data.astype(np.float32).T
            
            # 标签分配：使用之后最近的问卷标签
            if onset_arr is not None and scores_all is not None and len(onset_arr) > 0:
                idx = np.searchsorted(onset_arr, window_start_sec, side='right')
                if idx < len(onset_arr):
                    label = scores_all[idx]
                else:
                    label = scores_all[-1]
            else:
                label = np.zeros(n_labels, dtype=np.float32)
            
            current_idx = start_idx + written_count
            x_memmap[current_idx] = segment
            y_memmap[current_idx] = label
            d_memmap[current_idx] = domain_id
            ts_memmap[current_idx] = window_start_sec
            sub_memmap[current_idx] = sub_idx
            
            written_count += 1
    
    print(f"  {ses_id}: {written_count} valid unlabeled windows, "
          f"{skipped_labeled_rough} labeled skipped (rough), "
          f"{skipped_labeled_exact} labeled skipped (exact), "
          f"{skipped_invalid} invalid skipped")
    return written_count


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 第一步：动态计算每个被试的域起始偏移和域槽位数（与align_cadt_input.py一致）
    domain_offset_map, domains_per_subject_map = compute_domain_offset_map(args.data_root)
    print(f"\n动态域配置:")
    for sub_id in sorted(domain_offset_map.keys()):
        start = domain_offset_map[sub_id]
        end = start + domains_per_subject_map[sub_id] - 1
        print(f"  sub-{sub_id:02d}: domains_per_subject={domains_per_subject_map[sub_id]}, 域范围 [{start}, {end}]")
    
    # 加载已标注数据的时间戳，作为额外保障
    labeled_ts_map = load_labeled_timestamps(args.labeled_data_dir)
    
    subjects = list(range(1, 16))
    
    print("\n" + "=" * 60)
    print("Step 1: Counting total samples...")
    print("=" * 60)
    
    total_samples = 0
    total_sessions = 0
    session_counts = []
    
    for sub_idx in subjects:
        sub_str = f"sub-{sub_idx:02d}"
        eeg_deriv_root = os.path.join(args.data_root, "derivatives", "eeg-prep", sub_str)
        
        if not os.path.exists(eeg_deriv_root):
            print(f"Subject {sub_idx:02d}: Directory not found, skip")
            continue
        
        ses_dirs = sorted(glob.glob(os.path.join(eeg_deriv_root, "ses-*")))
        
        if len(ses_dirs) == 0:
            print(f"Subject {sub_idx:02d}: No session directories, skip")
            continue
        
        print(f"\nSubject {sub_idx:02d}:")
        
        for ses_dir in ses_dirs:
            count = count_session_samples(
                sub_idx,
                ses_dir,
                data_root=args.data_root,
                fs=args.fs,
                window_size=args.window_size,
                step_size=args.step_size,
                n_channels=args.n_channels,
                label_window=args.label_window,
                labeled_ts_map=labeled_ts_map,
            )
            session_counts.append(count)
            total_samples += count
            total_sessions += 1
            print(f"  {os.path.basename(ses_dir)}: {count} samples")
    
    print(f"\nTotal estimated samples: {total_samples}")
    print(f"Total sessions: {total_sessions}")
    
    if total_samples == 0:
        raise RuntimeError("No valid unlabeled samples found. Please check data path and parameters.")
    
    print("\n" + "=" * 60)
    print("Step 2: Processing and writing data...")
    print("=" * 60)
    
    x_shape = (total_samples, args.seq_len, args.n_channels)
    y_shape = (total_samples, args.n_labels)
    d_shape = (total_samples,)
    ts_shape = (total_samples,)
    sub_shape = (total_samples,)
    
    x_path = os.path.join(args.output_dir, "X.npy")
    y_path = os.path.join(args.output_dir, "y.npy")
    d_path = os.path.join(args.output_dir, "d.npy")
    ts_path = os.path.join(args.output_dir, "ts.npy")
    sub_path = os.path.join(args.output_dir, "sub.npy")
    
    x_memmap = np.memmap(x_path, dtype=np.float32, mode='w+', shape=x_shape)
    y_memmap = np.memmap(y_path, dtype=np.float32, mode='w+', shape=y_shape)
    d_memmap = np.memmap(d_path, dtype=np.int64, mode='w+', shape=d_shape)
    ts_memmap = np.memmap(ts_path, dtype=np.float32, mode='w+', shape=ts_shape)
    sub_memmap = np.memmap(sub_path, dtype=np.int64, mode='w+', shape=sub_shape)
    
    current_idx = 0
    
    for sub_idx in subjects:
        sub_str = f"sub-{sub_idx:02d}"
        eeg_deriv_root = os.path.join(args.data_root, "derivatives", "eeg-prep", sub_str)
        
        if not os.path.exists(eeg_deriv_root):
            continue
        
        ses_dirs = sorted(glob.glob(os.path.join(eeg_deriv_root, "ses-*")))
        
        if len(ses_dirs) == 0:
            continue
        
        print(f"\n========== Processing Subject {sub_idx:02d} ==========")
        
        for ses_dir in ses_dirs:
            written = process_session_to_arrays(
                sub_idx,
                ses_dir,
                data_root=args.data_root,
                fs=args.fs,
                window_size=args.window_size,
                step_size=args.step_size,
                n_channels=args.n_channels,
                n_labels=args.n_labels,
                label_window=args.label_window,
                x_memmap=x_memmap,
                y_memmap=y_memmap,
                d_memmap=d_memmap,
                ts_memmap=ts_memmap,
                sub_memmap=sub_memmap,
                domain_offset_map=domain_offset_map,
                start_idx=current_idx,
                labeled_ts_map=labeled_ts_map,
            )
            current_idx += written
    
    x_memmap.flush()
    y_memmap.flush()
    d_memmap.flush()
    ts_memmap.flush()
    sub_memmap.flush()
    
    # 使用 np.save 保存标准 .npy 格式
    x_arr = np.array(x_memmap)
    y_arr = np.array(y_memmap)
    d_arr = np.array(d_memmap)
    ts_arr = np.array(ts_memmap)
    sub_arr = np.array(sub_memmap)
    
    del x_memmap
    del y_memmap
    del d_memmap
    del ts_memmap
    del sub_memmap
    
    np.save(x_path, x_arr)
    np.save(y_path, y_arr)
    np.save(d_path, d_arr)
    np.save(ts_path, ts_arr)
    np.save(sub_path, sub_arr)
    
    meta = {
        "seq_len": args.seq_len,
        "n_channels": args.n_channels,
        "n_labels": args.n_labels,
        "label_names": LABEL_NAMES,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "fs": args.fs,
        "label_window": args.label_window,
        "domain_offset_map": {str(k): v for k, v in domain_offset_map.items()},
        "domains_per_subject_map": {str(k): v for k, v in domains_per_subject_map.items()},
        "num_samples": int(x_arr.shape[0]),
        "num_subjects": int(len(np.unique(sub_arr))),
        "num_domains": int(len(np.unique(d_arr))),
        "num_sessions": total_sessions,
        "x_shape": list(x_arr.shape),
        "y_shape": list(y_arr.shape),
        "d_shape": list(d_arr.shape),
        "ts_shape": list(ts_arr.shape),
        "sub_shape": list(sub_arr.shape),
        "ts_range": [float(ts_arr.min()), float(ts_arr.max())],
        "d_range": [int(d_arr.min()), int(d_arr.max())],
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Unlabeled Raw EEG processing complete")
    print(f"  Total sessions processed: {total_sessions}")
    print(f"  Valid unlabeled samples: {x_arr.shape[0]}")
    print(f"  X shape: {x_arr.shape}  (N, seq_len, n_channels)")
    print(f"  y shape: {y_arr.shape}  (N, n_labels)")
    print(f"  d shape: {d_arr.shape}  域 ID 范围 [{d_arr.min()}, {d_arr.max()}]")
    print(f"  ts shape: {ts_arr.shape}  timestamp range [{ts_arr.min():.1f}, {ts_arr.max():.1f}] sec")
    print(f"  sub shape: {sub_arr.shape}  被试范围 [{sub_arr.min()}, {sub_arr.max()}]")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()