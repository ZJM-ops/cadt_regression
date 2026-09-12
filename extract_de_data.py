#!/usr/bin/env python3
"""
从 de_output_addtime_new 下的 H5 提取 DE 特征，输出 NPY 到 /vePFS-0x0d/home/zjm/de_data。

DE 特征单独存放于此目录，仅供 MLPEncoder 使用；其余数据（波形等）仍在
/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/ 读取。

与 align_cadt_input.py 的区别：
  - 使用 H5 中的 "de" dataset (n_chans, 5)，展平成 (295,) 向量
  - 不处理原始波形，不做滑窗（每个 sample 在 H5 中已经是一个 10s 窗的 DE）
  - X.npy 形状: (N, 1, de_dim)  seq_len=1，x_dim=de_dim  保持和 CNN 输入 3 维一致
      也输出额外一份 X_vec.npy 形状 (N, de_dim)，方便 MLP 直接用

注意：DE 已在 test_load.py 阶段按天做过 z-score，本脚本不再额外标准化。
"""

import argparse
import glob
import json
import os
from typing import Iterator, Tuple

import h5py
import numpy as np

LABEL_NAMES = [
    "inspired", "alert", "determined", "attentive", "active",
    "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue",
]

DE_BANDS = 5  # delta / theta / alpha / beta / gamma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DE features from H5 to NPY for MLP baseline")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/vePFS-0x0d/DailyEEG/de_output_addtime_new",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/vePFS-0x0d/home/zjm/de_data",
    )
    parser.add_argument("--n-channels", type=int, default=59)
    parser.add_argument("--n-labels", type=int, default=11)
    return parser.parse_args()


def iter_de_samples(h5_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray, int, int, float]]:
    with h5py.File(h5_path, "r") as hf:
        trial_keys = sorted(hf.keys(), key=lambda k: int(k.replace("trial", "")))
        for trial_key in trial_keys:
            trial_grp = hf[trial_key]
            sample_keys = sorted(trial_grp.keys(), key=lambda k: int(k.replace("sample", "")))
            for sample_key in sample_keys:
                sample_grp = trial_grp[sample_key]
                if "de" not in sample_grp:
                    continue
                de = sample_grp["de"][()]  # (n_chans, bands) or flattened
                label = np.asarray(sample_grp.attrs["label"], dtype=np.float32)
                sub_id = int(sample_grp.attrs["sub_id"])
                day_id = int(sample_grp.attrs["day_id"])
                start_sec = float(sample_grp.attrs.get("start_sec", 0.0))
                yield de, label, sub_id, day_id, start_sec


def compute_domain_offset_map(h5_files):
    sub_day_map = {}
    for h5_path in h5_files:
        for _, _, sub_id, day_id, _ in iter_de_samples(h5_path):
            sub_day_map.setdefault(sub_id, set()).add(day_id)
    domains_per_subject_map = {s: (max(ds) + 1 if ds else 1) for s, ds in sub_day_map.items()}
    domain_offset_map = {}
    offset = 0
    for s in sorted(domains_per_subject_map.keys()):
        domain_offset_map[s] = offset
        offset += domains_per_subject_map[s]
    return domain_offset_map, domains_per_subject_map


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "sub_*_de.h5")))
    if not h5_files:
        raise FileNotFoundError(f"未在 {args.input_dir} 找到 sub_*_de.h5")

    domain_offset_map, domains_per_subject_map = compute_domain_offset_map(h5_files)
    print(f"\n动态域配置（{len(domain_offset_map)} 个被试）:")
    for s in sorted(domain_offset_map.keys()):
        start = domain_offset_map[s]
        end = start + domains_per_subject_map[s] - 1
        print(f"  sub-{s:02d}: day_slots={domains_per_subject_map[s]}  域范围 [{start}, {end}]")

    x_list, y_list, d_list, sub_list, ts_list = [], [], [], [], []
    skipped = 0
    de_dim_expected = args.n_channels * DE_BANDS

    for h5_path in h5_files:
        for de, label, sub_id, day_id, start_sec in iter_de_samples(h5_path):
            if label.shape[0] != args.n_labels:
                skipped += 1
                continue

            de_vec = de.reshape(-1).astype(np.float32)
            if de_vec.shape[0] != de_dim_expected:
                print(f"  [跳过] {os.path.basename(h5_path)}: de dim={de_vec.shape[0]} != expected {de_dim_expected}")
                skipped += 1
                continue
            if not np.isfinite(de_vec).all():
                skipped += 1
                continue

            offset = domain_offset_map.get(sub_id, 0)
            domain_id = offset + day_id
            x_list.append(de_vec)
            y_list.append(label)
            d_list.append(domain_id)
            sub_list.append(sub_id)
            ts_list.append(start_sec)

    if not x_list:
        raise RuntimeError("没有有效样本")

    x_vec = np.stack(x_list, axis=0).astype(np.float32)  # (N, 295)
    # 为保持和原 CNN 输入相同的 3D 格式 (N, seq_len, x_dim)，seq_len=1
    x_3d = x_vec[:, None, :]  # (N, 1, 295)
    y_arr = np.stack(y_list, axis=0).astype(np.float32)
    d_arr = np.asarray(d_list, dtype=np.int64)
    sub_arr = np.asarray(sub_list, dtype=np.int64)
    ts_arr = np.asarray(ts_list, dtype=np.float32)

    np.save(os.path.join(args.output_dir, "X.npy"), x_3d)
    np.save(os.path.join(args.output_dir, "X_vec.npy"), x_vec)
    np.save(os.path.join(args.output_dir, "y.npy"), y_arr)
    np.save(os.path.join(args.output_dir, "d.npy"), d_arr)
    np.save(os.path.join(args.output_dir, "sub.npy"), sub_arr)
    np.save(os.path.join(args.output_dir, "ts.npy"), ts_arr)

    meta = {
        "feature_type": "DE_flat",
        "bands": ["delta(1-4)", "theta(4-8)", "alpha(8-14)", "beta(14-30)", "gamma(30-47)"],
        "n_bands": DE_BANDS,
        "n_channels": args.n_channels,
        "de_dim": de_dim_expected,
        "seq_len": 1,
        "x_dim": de_dim_expected,
        "n_labels": args.n_labels,
        "label_names": LABEL_NAMES,
        "normalization": "day-wise z-score applied in test_load.py",
        "domain_offset_map": {str(k): v for k, v in domain_offset_map.items()},
        "domains_per_subject_map": {str(k): v for k, v in domains_per_subject_map.items()},
        "num_samples": int(x_vec.shape[0]),
        "num_subjects": int(len(np.unique(sub_arr))),
        "num_domains": int(len(np.unique(d_arr))),
        "skipped_samples": skipped,
        "x_shape_3d": list(x_3d.shape),
        "x_shape_vec": list(x_vec.shape),
        "y_shape": list(y_arr.shape),
        "d_range": [int(d_arr.min()), int(d_arr.max())],
        "ts_range": [float(ts_arr.min()), float(ts_arr.max())],
        "source_h5_files": [os.path.basename(p) for p in h5_files],
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("DE 特征抽取完成")
    print(f"  有效样本: {x_vec.shape[0]}，跳过: {skipped}")
    print(f"  X_3d shape: {x_3d.shape}  (N, seq_len=1, de_dim={de_dim_expected})")
    print(f"  X_vec shape: {x_vec.shape}  (N, de_dim)")
    print(f"  y shape:     {y_arr.shape}  (N, {args.n_labels})")
    print(f"  d shape:     {d_arr.shape}  域范围 [{d_arr.min()}, {d_arr.max()}]")
    print(f"  sub shape:   {sub_arr.shape}  被试范围 [{sub_arr.min()}, {sub_arr.max()}]")
    print(f"  ts shape:    {ts_arr.shape}  时间戳 [{ts_arr.min():.1f}, {ts_arr.max():.1f}]s")
    print(f"  输出目录:    {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
