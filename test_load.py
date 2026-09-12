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
    OUTPUT_DIR = r"/vePFS-0x0d/home/zjm/de_output_addtime_new/"
else:
    DATA_ROOT = "/vePFS-0x0d/DailyEEG"
    OUTPUT_DIR = "/vePFS-0x0d/home/zjm/de_output_addtime_new/"

# 输出特征文件夹不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- 日志配置新增 --------------------------
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_count_log.txt")
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


def _extract_segment_number(fname):
    basename = os.path.basename(fname)
    basename_no_ext = basename.split('.')[0]
    parts = basename_no_ext.split('-')
    for i in range(len(parts)-1, 0, -1):
        part = parts[i]
        prev_part = parts[i-1].lower()
        if part.isdigit() and ('eeg' in prev_part or 'beh' in prev_part):
            return int(part)
    return 0


FS = 200
TIME_LEN = 10
TIME_STEP = 5
TRIAL_DURATION = 120


def _process_segmented_session(subject_id, ses_id, day_id, fif_paths, beh_files, label_cols, fs, t, points_len, points_step):
    print(f"  Detected segmented data, processing with time offset...")
    
    fif_segments = {}
    all_meas_dates = []
    
    for fp in fif_paths:
        seg_num = _extract_segment_number(fp)
        if seg_num == 0:
            continue
        r = mne.io.read_raw_fif(fp, preload=True, verbose=False)
        sfreq = r.info["sfreq"]
        if not np.isclose(sfreq, fs):
            print(f"  Skip segment {seg_num}: sampling rate mismatch")
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
        print(f"  No valid segments found")
        return None, None, None, None, 0
    
    all_meas_dates.sort(key=lambda x: x[1])
    earliest_date = all_meas_dates[0][1]
    
    for seg_num, info in fif_segments.items():
        offset_sec = (info['meas_date'] - earliest_date).total_seconds()
        info['offset_sec'] = offset_sec
        info['end_sec'] = offset_sec + info['duration']
    
    all_trials = []
    for bf in beh_files:
        seg_num = _extract_segment_number(bf)
        if seg_num == 0 or seg_num not in fif_segments:
            continue
        df_part = pd.read_csv(bf, sep="\t")
        if len(df_part) == 0:
            continue
        
        offset_sec = fif_segments[seg_num]['offset_sec']
        for _, row in df_part.iterrows():
            all_trials.append({
                'session_onset': row['onset'] + offset_sec,
                'scores': row[label_cols].astype(np.float32).values
            })
    
    if len(all_trials) == 0:
        print(f"  No valid trials found")
        return None, None, None, None, 0
    
    all_trials.sort(key=lambda x: x['session_onset'])
    
    max_end_sec = max(info['end_sec'] for info in fif_segments.values())
    total_samples = int(max_end_sec * fs) + 1
    n_chans = list(fif_segments.values())[0]['raw_data'].shape[0]
    
    session_eeg = np.full((n_chans, total_samples), np.nan)
    
    for seg_num, info in fif_segments.items():
        offset_samples = int(info['offset_sec'] * fs)
        raw_data = info['raw_data']
        n_samples = raw_data.shape[1]
        end_sample = min(offset_samples + n_samples, total_samples)
        session_eeg[:, offset_samples:end_sample] = raw_data[:, :end_sample - offset_samples]
    
    return _generate_samples_from_eeg(session_eeg, total_samples, all_trials, day_id, subject_id, ses_id, fs, t, points_len, points_step)


def _generate_samples_from_eeg(eeg_data, total_samples, all_trials, day_id, subject_id, ses_id, fs, t, points_len, points_step):
    session_data = []
    session_labels = []
    session_n_samples = []
    session_timestamps = []
    valid_trial_count = 0
    
    for trial_idx, trial in enumerate(all_trials):
        onset_sec = trial['session_onset']
        scores = trial['scores']
        
        end_pt = int(onset_sec * fs)
        start_pt = int((onset_sec - t) * fs)
        
        if start_pt < 0:
            print(f"  Skip trial {trial_idx}: onset too early")
            continue
        if end_pt > total_samples:
            print(f"  Skip trial {trial_idx}: onset exceeds session")
            continue
        
        t_data = eeg_data[:, start_pt:end_pt]
        
        if np.any(np.isnan(t_data)):
            print(f"  Skip trial {trial_idx}: contains NaN (gap in data)")
            continue
        
        if t_data.shape[1] < int(t * fs):
            print(f"  Skip trial {trial_idx}: signal too short")
            continue
        
        flat_block_len = 5 * fs
        if has_uniform_block(t_data, block_len=flat_block_len):
            t_data = remove_uniform_block(t_data)
        if t_data.shape[1] < points_len:
            print(f"  Skip trial {trial_idx}: too short after flat block removal")
            continue
        
        trial_n_samples = int((t_data.shape[1] - points_len) / points_step) + 1
        if trial_n_samples <= 0:
            print(f"  Skip trial {trial_idx}: no valid slide windows")
            continue
        
        session_labels.append(scores)
        session_n_samples.append(trial_n_samples)
        valid_trial_count += 1
        
        trial_window_start = onset_sec - t
        
        for j in range(trial_n_samples):
            s = j * points_step
            e = s + points_len
            session_data.append(t_data[:, s:e])
            sample_start_sec = trial_window_start + j * (points_step / fs)
            session_timestamps.append(sample_start_sec)
    
    if len(session_data) == 0:
        print(f"  No valid samples")
        return None, None, None, None, 0
    
    print(f"  Collect {len(session_data)} valid 10s EEG samples")
    
    return session_data, session_labels, session_timestamps, session_n_samples, valid_trial_count


def _process_segment(raw_data, df, label_cols, fs, t, points_len, points_step, 
                    day_id, subject_id, ses_id, segment_num, logger):
    scores_all = df[label_cols].astype(np.float32).values
    onset_arr = df["onset"].values
    total_pts = raw_data.shape[1]

    seg_data = []
    seg_labels = []
    seg_timestamps = []
    seg_n_samples = []
    valid_trial_count = 0

    skip_onset_early = 0
    skip_end_exceed = 0
    skip_signal_short = 0
    skip_after_clean_short = 0
    skip_no_slide_window = 0
    cleaned_flat_count = 0

    for trial_idx in range(len(df)):
        onset_sec = onset_arr[trial_idx]
        scores = scores_all[trial_idx]
        end_pt = int(onset_sec * fs)
        start_pt = int((onset_sec - t) * fs)

        if start_pt < 0:
            logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Segment{segment_num}, Trial{trial_idx}, onset_early, onset={onset_sec:.1f}s")
            skip_onset_early += 1
            continue
        if end_pt > total_pts:
            logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Segment{segment_num}, Trial{trial_idx}, end_exceed, onset={onset_sec:.1f}s, end_pt={end_pt}, total_pts={total_pts}")
            skip_end_exceed += 1
            continue

        t_data = raw_data[:, start_pt:end_pt]
        if t_data.shape[1] < int(t * fs):
            logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Segment{segment_num}, Trial{trial_idx}, signal_short, t_data_len={t_data.shape[1]}, need={int(t*fs)}")
            skip_signal_short += 1
            continue

        flat_block_len = 5 * fs
        if has_uniform_block(t_data, block_len=flat_block_len):
            cleaned_flat_count += 1
            t_data = remove_uniform_block(t_data)
        if t_data.shape[1] < points_len:
            logger.info(f"Skip reason: Subject={subject_id:02d}, Session={ses_id}, Segment{segment_num}, Trial{trial_idx}, after_clean_short, t_data_len={t_data.shape[1]}, need={points_len}")
            skip_after_clean_short += 1
            continue

        trial_n_samples = int((t_data.shape[1] - points_len) / points_step) + 1
        if trial_n_samples <= 0:
            skip_no_slide_window += 1
            continue

        seg_labels.append(scores)
        seg_n_samples.append(trial_n_samples)
        valid_trial_count += 1

        trial_window_start = onset_sec - t

        for j in range(trial_n_samples):
            s = j * points_step
            e = s + points_len
            seg_data.append(t_data[:, s:e])
            sample_start_sec = trial_window_start + j * (points_step / fs)
            seg_timestamps.append(sample_start_sec)

    total_trials = len(df)
    total_skipped = skip_onset_early + skip_end_exceed + skip_signal_short + skip_after_clean_short + skip_no_slide_window

    if valid_trial_count == 0 or total_skipped > 0:
        logger.info(
            f"SkipDetail: Subject={subject_id:02d}, Session={ses_id}, Segment{segment_num}, TotalTrials={total_trials}, "
            f"Valid={valid_trial_count}, Skipped={total_skipped}, "
            f"OnsetEarly={skip_onset_early}, EndExceed={skip_end_exceed}, "
            f"SignalShort={skip_signal_short}, AfterCleanShort={skip_after_clean_short}, "
            f"NoSlideWindow={skip_no_slide_window}, CleanedFlat={cleaned_flat_count}"
        )

    return seg_data, seg_labels, seg_timestamps, seg_n_samples, valid_trial_count


def load_preprocessed_data(subject_id, fs, timeLen=10, timeStep=5, t=120):
    sub_str = f"sub-{subject_id:02d}"
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
    timestamps_global = []
    day_ids_global = []
    points_len = int(timeLen * fs)
    points_step = int(timeStep * fs)

    label_cols = [
        "inspired", "alert", "determined", "attentive", "active",
        "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue"
    ]

    for ses_dir in ses_dirs:
        ses_id = os.path.basename(ses_dir)
        day_id = int(ses_id.replace("ses-", "")) - 1
        print(f"\nProcessing {sub_str} {ses_id} (day_id={day_id}) ...")

        fif_paths = sorted(glob.glob(os.path.join(ses_dir, "*_prep_*eeg*.fif")))
        if len(fif_paths) == 0:
            print(f"No fif file in {ses_dir}, skip")
            continue

        beh_pattern = os.path.join(beh_root, ses_id, "beh", f"{sub_str}_{ses_id}_task-dailylife_emotion_beh*.tsv")
        beh_files = sorted(glob.glob(beh_pattern))
        if len(beh_files) == 0:
            print(f"No emotion tsv found under {os.path.join(beh_root, ses_id, 'beh')}, skip session")
            continue

        day_data = []
        day_labels = []
        day_timestamps = []
        day_n_samples = []
        valid_trial_count = 0

        has_segments = any(_extract_segment_number(fp) > 0 for fp in fif_paths)

        if has_segments:
            seg_data, seg_labels, seg_timestamps, seg_n_samples, seg_valid = _process_segmented_session(
                subject_id, ses_id, day_id, fif_paths, beh_files, label_cols, fs, t, points_len, points_step
            )

            if seg_data is not None:
                day_data.extend(seg_data)
                day_labels.extend(seg_labels)
                day_timestamps.extend(seg_timestamps)
                day_n_samples.extend(seg_n_samples)
                valid_trial_count += seg_valid

        else:
            raw_list = []
            for fp in fif_paths:
                r = mne.io.read_raw_fif(fp, preload=True, verbose=False)
                raw_list.append(r)
            raw = mne.concatenate_raws(raw_list)
            raw_data = raw.get_data()
            sfreq = raw.info["sfreq"]
            if not np.isclose(sfreq, fs):
                print(f"Skip session: sampling rate {sfreq} != target fs {fs}")
                continue

            df_list = []
            for bf in beh_files:
                df_part = pd.read_csv(bf, sep="\t")
                df_list.append(df_part)
            df = pd.concat(df_list, ignore_index=True)
            if len(df) == 0:
                print(f"Empty emotion tsv table, skip session")
                continue

            seg_data, seg_labels, seg_timestamps, seg_n_samples, seg_valid = _process_segment(
                raw_data, df, label_cols, fs, t, points_len, points_step,
                day_id, subject_id, ses_id, 0, logger
            )

            day_data.extend(seg_data)
            day_labels.extend(seg_labels)
            day_timestamps.extend(seg_timestamps)
            day_n_samples.extend(seg_n_samples)
            valid_trial_count += seg_valid

        n_samples_per_rank.extend(day_n_samples)
        day_ids_global.extend([day_id] * len(day_n_samples))

        if len(day_data) > 0:
            data_global.extend(day_data)
            labels_global.extend(day_labels)
            timestamps_global.extend(day_timestamps)
            sample_num_each_day.append(len(day_data))
            print(f"{ses_id}: Collect {len(day_data)} valid 10s EEG samples")
            logger.info(
                f"Subject={subject_id:02d}, Session={ses_id}, ValidTrials={valid_trial_count}, Total10sSample={len(day_data)}"
            )
        else:
            print(f"{ses_id}: No valid samples")
            logger.info(
                f"Subject={subject_id:02d}, Session={ses_id}, ValidTrials={valid_trial_count}, Total10sSample=0"
            )

    data_arr = np.array(data_global) if len(data_global) > 0 else np.empty((0, 0, 0))
    label_arr = np.array(labels_global) if len(labels_global) > 0 else np.empty((0, 11))
    npr_arr = np.array(n_samples_per_rank)
    raw_10s_all = np.array(data_global)
    timestamps_arr = np.array(timestamps_global, dtype=np.float32) if len(timestamps_global) > 0 else np.empty((0,))
    day_ids_arr = np.array(day_ids_global, dtype=np.int32) if len(day_ids_global) > 0 else np.empty((0,))

    return data_arr, label_arr, sample_num_each_day, npr_arr, raw_10s_all, timestamps_arr, day_ids_arr


def ext_de(sub_idx):
    data, onesub_label, n_samples_per_day, n_samples_per_rank, raw_10s_eeg_all, timestamps_all, day_ids_all = load_preprocessed_data(sub_idx, fs=200)

    if data.shape[0] == 0:
        print(f"sub-{sub_idx:02d}: No valid EEG data")
        logger.info(f"Subject={sub_idx:02d}, All sessions no valid data")
        return

    n_samples, n_chans, sfreqs = data.shape

    freqs = [[1,4],[4,8],[8,14],[14,30],[30,47]]
    de_data = np.zeros((n_samples, n_chans, len(freqs)))

    for i in range(n_samples):
        sample = data[i]

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

    fea = de_data.reshape(n_samples, -1)

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

    out_path = os.path.join(OUTPUT_DIR, f"sub_{sub_idx:02d}_de.h5")

    with h5py.File(out_path, "w") as f:
        sample_ptr = 0
        trial_start = 0

        for trial_idx, trial_sample_num in enumerate(n_samples_per_rank):
            trial_grp = f.create_group(f"trial{trial_idx}")

            label = onesub_label[trial_idx]
            day_id = day_ids_all[trial_idx]

            for sample_idx in range(trial_sample_num):
                sample_grp = trial_grp.create_group(f"sample{sample_idx}")

                sample_grp.create_dataset(
                    "de",
                    data=de_data[sample_ptr],
                    dtype=np.float32
                )

                sample_grp.create_dataset(
                    "raw_eeg_10s_window",
                    data=raw_10s_eeg_all[sample_ptr],
                    dtype=np.float32
                )

                sample_grp.attrs["label"] = label
                sample_grp.attrs["sub_id"] = sub_idx
                sample_grp.attrs["day_id"] = day_id
                sample_grp.attrs["start_sec"] = float(timestamps_all[sample_ptr])

                sample_ptr += 1

            trial_start += trial_sample_num

    print(f"Saved: {out_path}")
    logger.info(f"Subject={sub_idx:02d} processing finished, total 10s samples: {n_samples}")


if __name__ == '__main__':
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