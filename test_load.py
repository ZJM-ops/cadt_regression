import numpy as np
import os
import pandas as pd
from datetime import datetime
import mne
import h5py
import glob
import logging

DEVICE = "local"  # local / cuda

#eeg切片+de特征提取+平滑+存储写入h5
'''
生成的.h5文件结构示例：
sub_XX_de.h5
├─ trial0/            # 第0份情绪问卷（一份trial=一次情绪填写）
│  ├─ sample0/        # 该问卷下第0个平滑窗口样本
│  │  ├─ Dataset "de"
│  │  ├─ Dataset "raw_eeg_10s_window"
│  │  └─ 4个属性：label / sub_id / day_id/start_sec(该样本的起始时间戳，单位秒，相对于session开始)
│  ├─ sample1/
│  └─ ...
├─ trial1/
└─ ...

每个trial下的sample数量 = (该trial的10s EEG样本数 - 平滑窗口长度) / 平滑步长 + 1
每个trail 120s,10s切分,5s滑动,一共23段,分段时间节点（起始秒）: 0，5，10，15，20，25，30，35，40，45，50，55，60，65，70，75，80，85，90，95，100，105，110

'''

# ========== BIDS数据集根路径 ==========
if DEVICE == "local":
    DATA_ROOT = r"/vePFS-0x0d/DailyEEG"
    OUTPUT_DIR = r"/vePFS-0x0d/DailyEEG/de_output_addtime_new"
else:
    DATA_ROOT = "/vePFS-0x0d/DailyEEG"
    OUTPUT_DIR = "/vePFS-0x0d/DailyEEG/de_output_addtime_new"

# 输出特征文件夹不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- 日志配置新增 --------------------------
LOG_PATH = os.path.join(OUTPUT_DIR, "sample_count_log.txt")
# 配置logging，同时输出控制台+日志文件，追加写入，utf8中文不乱码
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EEG_Sample_Count")


def _format_index_preview(indices: np.ndarray, max_items: int = 8) -> str:
    if indices.size == 0:
        return "[]"
    preview = indices[:max_items].tolist()
    suffix = "" if indices.shape[0] <= max_items else f" ... total={indices.shape[0]}"
    return f"{preview}{suffix}"


def _raise_on_nonfinite_or_nonpositive_var(var: np.ndarray, *, rank: int, band: list[int]) -> None:
    nonfinite_idx = np.argwhere(~np.isfinite(var))
    if nonfinite_idx.size > 0:
        raise ValueError(
            "Non-finite variance detected before DE log transform: "
            f"rank={rank}, band={band}, positions={_format_index_preview(nonfinite_idx)}"
        )

    nonpositive_idx = np.argwhere(var <= 0.0)
    if nonpositive_idx.size > 0:
        preview = nonpositive_idx[:8].tolist()
        vals = var[nonpositive_idx[:8, 0], nonpositive_idx[:8, 1]].tolist()
        raise ValueError(
            "Non-positive variance detected before DE log transform: "
            f"rank={rank}, band={band}, positions={preview}, values={vals}, total={len(nonpositive_idx)}"
        )


def _raise_on_invalid_day_std(day_std: np.ndarray, *, day: int) -> None:
    invalid_idx = np.argwhere((~np.isfinite(day_std)) | (day_std <= 0.0))
    if invalid_idx.size > 0:
        preview = invalid_idx[:8].tolist()
        vals = day_std[invalid_idx[:8, 0], invalid_idx[:8, 1]].tolist()
        raise ValueError(
            "Invalid std detected before day-wise z-score normalization: "
            f"day={day}, positions={preview}, values={vals}, total={len(invalid_idx)}"
        )


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


def remove_uniform_block(data: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    chan_range = np.max(data, axis=1) - np.min(data, axis=1)
    flat_mask = chan_range < atol
    return data[~flat_mask, :]


# ===================== 加载函数（兼容分段fif、分段tsv，新增返回全部10s原始EEG） =====================
def load_preprocessed_data(subject_id, fs, timeLen=10, timeStep=5, t=120): # timeLen=10：基础片段10秒；timeStep=5：滑动步长5秒，重叠5s
    sub_str = f"sub-{subject_id:02d}"
    # 修正路径：eeg-prep 带横杠
    eeg_deriv_root = os.path.join(DATA_ROOT, "derivatives", "eeg-prep", sub_str)
    beh_root = os.path.join(DATA_ROOT, sub_str)

    ses_dirs = sorted(glob.glob(os.path.join(eeg_deriv_root, "ses-*")))
    if len(ses_dirs) == 0:
        print(f"Warning: No session fif folder found under {eeg_deriv_root}")
        return np.array([]), np.array([]), [], np.array([]), np.array([])

    data_global = []
    labels_global = []
    sample_num_each_day = []
    n_samples_per_rank = []
    timestamps_global = []  # 新增：每个样本的起始时间戳（相对于session开始的秒数）
    day_ids_global = []     # 新增：每个trial对应的day_id（基于实际session编号）
    points_len = int(timeLen * fs)
    points_step = int(timeStep * fs)

    # 11维标签固定顺序
    label_cols = [
        "inspired", "alert", "determined", "attentive", "active",
        "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue"
    ]

    for ses_dir in ses_dirs:
        ses_id = os.path.basename(ses_dir)
        # 从session目录名提取day_id（0-based）
        # ses-01 → day_id=0, ses-02 → day_id=1, ..., ses-10 → day_id=9
        day_id = int(ses_id.replace("ses-", "")) - 1
        print(f"\nProcessing {sub_str} {ses_id} (day_id={day_id}) ...")

        # Step1 读取并拼接所有分段预处理fif
        # 使用 *_prep_*eeg*.fif 同时匹配正常和 abnormal 文件
        # 正常: *prep_eeg.fif, 分段: *prep_eeg-01.fif, abnormal: *prep_abnormal_eeg.fif
        fif_paths = sorted(glob.glob(os.path.join(ses_dir, "*_prep_*eeg*.fif")))
        if len(fif_paths) == 0:
            print(f"No fif file in {ses_dir}, skip")
            continue
        raw_list = []
        for fp in fif_paths:
            r = mne.io.read_raw_fif(fp, preload=True, verbose=False)
            raw_list.append(r)
        raw = mne.concatenate_raws(raw_list)
        raw_data = raw.get_data()  # (n_chan, n_time)
        sfreq = raw.info["sfreq"]
        if not np.isclose(sfreq, fs):
            print(f"Skip session: sampling rate {sfreq} != target fs {fs}")
            continue
        total_pts = raw_data.shape[1]

        # Step2 读取分段tsv标签，自动合并 -01/-02/-03
        beh_pattern = os.path.join(beh_root, ses_id, "beh", f"{sub_str}_{ses_id}_task-dailylife_emotion_beh*.tsv")
        beh_files = sorted(glob.glob(beh_pattern))
        if len(beh_files) == 0:
            print(f"No emotion tsv found under {os.path.join(beh_root, ses_id, 'beh')}, skip session")
            continue
        df_list = []
        for bf in beh_files:
            df_part = pd.read_csv(bf, sep="\t")
            df_list.append(df_part)
        df = pd.concat(df_list, ignore_index=True)
        if len(df) == 0:
            print(f"Empty emotion tsv table, skip session")
            continue

        scores_all = df[label_cols].astype(np.float32).values
        onset_arr = df["onset"].values

        day_data = []
        day_labels = []
        day_timestamps = []  # 新增：当前session的时间戳（相对于session开始的秒数）
        valid_trial_count = 0  # 当前session有效问卷数量
        # 跳过原因计数（诊断用）
        skip_onset_early = 0
        skip_end_exceed = 0
        skip_signal_short = 0
        skip_after_clean_short = 0
        skip_no_slide_window = 0
        cleaned_flat_count = 0

        # Step3 逐问卷截取前120s信号
        for trial_idx in range(len(df)):
            onset_sec = onset_arr[trial_idx]
            scores = scores_all[trial_idx]
            end_pt = int(onset_sec * fs)
            start_pt = int((onset_sec - t) * fs)

            if start_pt < 0:
                logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Trial{trial_idx}, onset_early, onset={onset_sec:.1f}s")
                skip_onset_early += 1
                continue
            if end_pt > total_pts:
                logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Trial{trial_idx}, end_exceed, onset={onset_sec:.1f}s, end_pt={end_pt}, total_pts={total_pts}")
                skip_end_exceed += 1
                continue

            t_data = raw_data[:, start_pt:end_pt]
            if t_data.shape[1] < int(t * fs):
                logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Trial{trial_idx}, signal_short, t_data_len={t_data.shape[1]}, need={int(t*fs)}")
                skip_signal_short += 1
                continue

            # 清洗连续平线噪声
            flat_block_len = 5*fs
            if has_uniform_block(t_data, block_len=flat_block_len):
                cleaned_flat_count += 1
                t_data = remove_uniform_block(t_data)
            if t_data.shape[1] < points_len:
                logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Trial{trial_idx}, after_clean_short, t_data_len={t_data.shape[1]}, need={points_len}")
                skip_after_clean_short += 1
                continue

            # 滑动切分10s样本
            trial_n_samples = int((t_data.shape[1] - points_len) / points_step) + 1
            if trial_n_samples <= 0:
                skip_no_slide_window += 1
                continue

            day_labels.append(scores)
            n_samples_per_rank.append(trial_n_samples)
            day_ids_global.append(day_id)  # 记录该trial的day_id
            valid_trial_count += 1
            
            # 该trial的120秒窗口起始时间 = onset - 120秒
            trial_window_start = onset_sec - t
            
            for j in range(trial_n_samples):
                s = j * points_step
                e = s + points_len
                day_data.append(t_data[:, s:e])
                
                # 记录每个样本的起始时间（秒），相对于session开始
                # 计算方式：onset - 120 + j * timeStep
                # 例如：onset=1713 → sample0=1593, sample1=1598, sample2=1603, ...
                sample_start_sec = trial_window_start + j * timeStep
                day_timestamps.append(sample_start_sec)

        # 汇总当前session
        total_trials = len(df)
        total_skipped = skip_onset_early + skip_end_exceed + skip_signal_short + skip_after_clean_short + skip_no_slide_window
        if len(day_data) > 0:
            data_global.extend(day_data)
            labels_global.extend(day_labels)
            timestamps_global.extend(day_timestamps)  # 新增：汇总时间戳
            sample_num_each_day.append(len(day_data))
            print(f"{ses_id}: Collect {len(day_data)} valid 10s EEG samples")
            # -------------------------- 写入日志 --------------------------
            logger.info(
                f"Subject={subject_id:02d}, Session={ses_id}, ValidTrials={valid_trial_count}, Total10sSample={len(day_data)}"
            )
        else:
            print(f"{ses_id}: No valid samples")
            logger.info(
                f"Subject={subject_id:02d}, Session={ses_id}, ValidTrials={valid_trial_count}, Total10sSample=0"
            )
        # 诊断：输出跳过原因汇总（ValidTrials=0 或 跳过数>0 时一定输出）
        if valid_trial_count == 0 or total_skipped > 0:
            logger.info(
                f"SkipDetail: Subject={subject_id:02d}, Session={ses_id}, TotalTrials={total_trials}, "
                f"Valid={valid_trial_count}, Skipped={total_skipped}, "
                f"OnsetEarly={skip_onset_early}, EndExceed={skip_end_exceed}, "
                f"SignalShort={skip_signal_short}, AfterCleanShort={skip_after_clean_short}, "
                f"NoSlideWindow={skip_no_slide_window}, CleanedFlat={cleaned_flat_count}"
            )

    data_arr = np.array(data_global) if len(data_global) > 0 else np.empty((0, 0, 0))
    label_arr = np.array(labels_global) if len(labels_global) > 0 else np.empty((0, 11))
    npr_arr = np.array(n_samples_per_rank)
    raw_10s_all = np.array(data_global)
    timestamps_arr = np.array(timestamps_global, dtype=np.float32) if len(timestamps_global) > 0 else np.empty((0,))
    day_ids_arr = np.array(day_ids_global, dtype=np.int32) if len(day_ids_global) > 0 else np.empty((0,))

    # 多返回一项：全部10s原始EEG波形和时间戳、每个trial的day_id
    return data_arr, label_arr, sample_num_each_day, npr_arr, raw_10s_all, timestamps_arr, day_ids_arr


# ===================== DE提取、归一化、平滑、存储（新增原始EEG写入h5） =====================
def ext_de(sub_idx):
    data, onesub_label, n_samples_per_day, n_samples_per_rank, raw_10s_eeg_all, timestamps_all, day_ids_all = load_preprocessed_data(sub_idx, fs=200)

    if data.shape[0] == 0:
        print(f"sub-{sub_idx:02d}: No valid EEG data")
        logger.info(f"Subject={sub_idx:02d}, All sessions no valid data")
        return

    n_samples, n_chans, sfreqs = data.shape

    freqs = [[1,4],[4,8],[8,14],[14,30],[30,47]]
    de_data = np.zeros((n_samples, n_chans, len(freqs)))

    # ====== 1. 逐sample计算DE（不再按trial拼接）======
    for i in range(n_samples):
        sample = data[i]  # (chan, time)

        for b_idx, band in enumerate(freqs):
            filtered = mne.filter.filter_data(
                sample, 200,
                l_freq=band[0],
                h_freq=band[1],
                verbose=False
            )

            var = np.var(filtered, axis=1, keepdims=True)
            if np.any(var <= 0):
                raise ValueError(f"Zero variance at sample {i}")

            de = 0.5 * np.log(2 * np.pi * np.e * var)
            de_data[i, :, b_idx] = de.squeeze()

    # reshape
    fea = de_data.reshape(n_samples, -1)

    # ====== 2. day-wise z-score ======
    n_samples_per_day_cum = np.concatenate(([0], np.cumsum(n_samples_per_day)))

    for day in range(len(n_samples_per_day)):
        s = n_samples_per_day_cum[day]
        e = n_samples_per_day_cum[day+1]

        day_slice = fea[s:e]
        mean = np.mean(day_slice, axis=0, keepdims=True)
        std = np.std(day_slice, axis=0, keepdims=True)

        if np.any(std == 0):
            raise ValueError("std=0 in normalization")

        fea[s:e] = (day_slice - mean) / std

    de_data = fea.reshape(n_samples, n_chans, len(freqs))

    # ====== 3. 写H5（按trial结构）======
    out_path = os.path.join(OUTPUT_DIR, f"sub_{sub_idx:02d}_de.h5")

    with h5py.File(out_path, "w") as f:
        sample_ptr = 0
        trial_start = 0

        for trial_idx, trial_sample_num in enumerate(n_samples_per_rank):
            trial_grp = f.create_group(f"trial{trial_idx}")

            label = onesub_label[trial_idx]
            day_id = day_ids_all[trial_idx]  # 使用实际session编号对应的day_id

            for sample_idx in range(trial_sample_num):
                sample_grp = trial_grp.create_group(f"sample{sample_idx}")

                # DE
                sample_grp.create_dataset(
                    "de",
                    data=de_data[sample_ptr],
                    dtype=np.float32
                )

                # 原始EEG (10s)
                sample_grp.create_dataset(
                    "raw_eeg_10s_window",
                    data=raw_10s_eeg_all[sample_ptr],
                    dtype=np.float32
                )

                # 属性
                sample_grp.attrs["label"] = label
                sample_grp.attrs["sub_id"] = sub_idx
                sample_grp.attrs["day_id"] = day_id
                sample_grp.attrs["start_sec"] = float(timestamps_all[sample_ptr])  # 时间戳（相对于session开始的秒数）

                sample_ptr += 1

            trial_start += trial_sample_num

    print(f"Saved: {out_path}")
    logger.info(f"Subject={sub_idx:02d} processing finished, total 10s samples: {n_samples}")


if __name__ == '__main__':
    #sub_list = [1]
    sub_list = list(range(1, 16))
    for sub_idx in sub_list:
        try:
            print(f"\n========== Start processing sub-{sub_idx:02d} ==========")
            logger.info(f"===== Start process Subject {sub_idx:02d} =====")
            ext_de(sub_idx)
            logger.info(f"===== Finish process Subject {sub_idx:02d} ====\n")
        except Exception as err:
            print(f"sub-{sub_idx:02d} processing failed, error: {err}")
            logger.error(f"Subject={sub_idx:02d} failed, error info: {str(err)}")
            continue