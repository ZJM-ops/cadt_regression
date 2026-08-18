# dataset.py
import numpy as np
from torch.utils.data import Dataset, DataLoader


def norm(z):
    z = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-6)
    return z


def norm_with_stats(z, mean, std):
    """Normalize with pre-computed statistics."""
    return (z - mean) / (std + 1e-6)



class MultiSourceBatchSampler:
    """
    多源域Batch采样器：从每个源域取相同数量的样本组成一个batch。
    """
    def __init__(self, domain_labels, source_domains, batch_size):
        self.source_domains = source_domains
        self.n_domains = len(source_domains)
        self.total_batch_size = batch_size
        self.per_domain_batch = batch_size // self.n_domains
        if self.per_domain_batch == 0:
            raise ValueError(f'batch_size={batch_size} 小于源域数量={self.n_domains}')
        self.actual_batch_size = self.per_domain_batch * self.n_domains
        self.domain_indices = {}
        for domain in source_domains:
            indices = np.where(domain_labels == domain)[0].tolist()
            if len(indices) == 0:
                raise ValueError(f'源域 {domain} 没有样本')
            self.domain_indices[domain] = indices
            print(f'  源域 {domain}: {len(indices)} 个样本')
        print(f'MultiSourceBatchSampler: {self.n_domains} domains, batch={self.actual_batch_size}')
        max_samples = max(len(indices) for indices in self.domain_indices.values())
        self.n_batches = max_samples // self.per_domain_batch
        if self.n_batches == 0:
            raise ValueError(f'源域样本不足 {self.per_domain_batch}')

    def __iter__(self):
        import itertools
        domain_iterators = {}
        for domain, indices in self.domain_indices.items():
            shuffled = indices.copy()
            np.random.shuffle(shuffled)
            domain_iterators[domain] = itertools.cycle(shuffled)
        for _ in range(self.n_batches):
            batch_indices = []
            for domain in self.source_domains:
                for _ in range(self.per_domain_batch):
                    batch_indices.append(next(domain_iterators[domain]))
            yield batch_indices

    def __len__(self):
        return self.n_batches


def load_emotion_split(split_dir, split_mode, split_type):
    """
    Load emotion project's split JSON file which contains sample indices.

    Args:
        split_dir: Path to splits directory
        split_mode: One of "cross_subject", "cross_day", "within_subject_day"
        split_type: One of "pretrain", "finetune", "val", "test"

    Returns:
        List of sample indices (integers)
    """
    import json
    import os
    json_path = os.path.join(split_dir, split_mode, f"{split_type}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Split file not found: {json_path}")

    with open(json_path, 'r') as f:
        indices = json.load(f)

    return indices


class DailyEEG_SplitData(Dataset):
    """
    DailyEEG 数据集，使用 emotion 项目生成的样本索引划分。

    根据不同的 split_mode，域的定义会不同：
    - cross_subject: 域 = subject（每个被试作为一个域）
    - cross_day: 域 = (subject, day)（每个被试-天对作为一个域）
    - within_subject_day: 域 = (subject, day)（每个被试-天对作为一个域）

    用法示例:
        train_indices = load_emotion_split(".../splits_new", "cross_subject", "pretrain")
        train_indices += load_emotion_split(".../splits_new", "cross_subject", "finetune")
        data_train = DailyEEG_SplitData(train_indices, split_mode="cross_subject", data_dir=".../processed_cadt_addtime_new")
    """
    def __init__(self, sample_indices, split_mode="cross_subject",
                 data_dir="/root/zjm/cadt_test_code/processed_cadt",
                 norm_mode="sample"):
        """
        Args:
            sample_indices: 样本索引列表
            split_mode: cross_subject, cross_day, within_subject_day
            data_dir: NPY 数据目录（支持波形 X.npy 或 DE 特征 X_vec.npy）
            norm_mode: 标准化模式
                - "sample": 逐样本独立标准化（默认）
                - "day": 按天标准化（每个被试的每一天使用该天所有样本统计量）
                - "subject": 按被试标准化（每个被试使用该被试所有样本统计量）
        """
        import os
        # 优先加载 X_vec.npy（DE 特征，2D: N x D），否则加载 X.npy（波形，3D: N x seq_len x ch）
        x_vec_path = os.path.join(data_dir, "X_vec.npy")
        x_path = os.path.join(data_dir, "X.npy")
        if os.path.exists(x_vec_path):
            all_x = np.load(x_vec_path).astype(np.float32)
        elif os.path.exists(x_path):
            all_x = np.load(x_path).astype(np.float32)
        else:
            raise FileNotFoundError(f"Neither X_vec.npy nor X.npy found in {data_dir}")

        y_path = os.path.join(data_dir, "y.npy")
        d_path = os.path.join(data_dir, "d.npy")

        all_y = np.load(y_path).astype(np.float32)
        all_d = np.load(d_path).astype(np.int64)

        sub_path = os.path.join(data_dir, "sub.npy")
        if os.path.exists(sub_path):
            self.all_sub = np.load(sub_path).astype(np.int64)
        else:
            self.all_sub = None

        meta_path = os.path.join(data_dir, "meta.json")
        self.domain_offset_map = {}
        self.domains_per_subject_map = {}
        if os.path.exists(meta_path):
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.domain_offset_map = {int(k): v for k, v in meta.get('domain_offset_map', {}).items()}
            self.domains_per_subject_map = {int(k): v for k, v in meta.get('domains_per_subject_map', {}).items()}

        if all_y.ndim == 1:
            all_y = all_y.reshape(-1, 1)

        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        max_index = len(all_x) - 1
        invalid_mask = (sample_indices < 0) | (sample_indices > max_index)
        if np.any(invalid_mask):
            invalid_indices = sample_indices[invalid_mask]
            raise ValueError(
                f"Invalid sample indices: {invalid_indices[:10]} (max valid index={max_index})"
            )

        self.x = all_x[sample_indices]
        self.y = all_y[sample_indices]
        self.sub = self.all_sub[sample_indices] if self.all_sub is not None else None
        self.norm_mode = norm_mode

        # 根据 split_mode 确定域的定义
        if split_mode == "cross_subject":
            # 跨被试模式：域 = subject，每个被试作为一个独立的域
            if self.sub is None:
                raise ValueError("cross_subject mode requires sub.npy file")
            self.d = self.sub.copy()
            self.day = None
        else:
            # cross_day 和 within_subject_day：域 = (subject, day)，使用原始的 d.npy
            self.d = all_d[sample_indices]
            if self.sub is not None:
                self.day = np.zeros_like(self.d)
                for sub_id, offset in self.domain_offset_map.items():
                    sub_mask = self.sub == sub_id
                    self.day[sub_mask] = self.d[sub_mask] - offset
            else:
                self.day = None

        # 预计算标准化统计量（仅当 x 是 3D 波形时按 day/subject 预计算；2D DE 特征只支持 sample 或全局）
        if all_x.ndim == 3 and norm_mode == "day":
            print(f"  预计算按天标准化统计量...")
            self._precompute_day_norm(all_x, all_d)
        elif all_x.ndim == 3 and norm_mode == "subject":
            print(f"  预计算按被试标准化统计量...")
            self._precompute_subject_norm(all_x, all_d)

        norm_desc = {"sample": "逐样本", "day": "按天", "subject": "按被试"}
        print(f"DailyEEG_SplitData ({split_mode}): {len(sample_indices)} samples, "
              f"x={self.x.shape}, y={self.y.shape}, "
              f"domain range [{self.d.min()}, {self.d.max()}], "
              f"norm={norm_desc.get(norm_mode, norm_mode)}")

    def _precompute_day_norm(self, all_x, all_d):
        """预计算每个 (被试, 天) 组合的均值和标准差（仅 3D 波形）"""
        unique_domains = np.unique(all_d)
        self.day_mean = {}
        self.day_std = {}

        for domain_id in unique_domains:
            mask = all_d == domain_id
            day_data = all_x[mask]
            mean = day_data.mean(axis=(0, 1))
            std = day_data.std(axis=(0, 1))
            std = np.maximum(std, 1e-6)
            self.day_mean[domain_id] = mean
            self.day_std[domain_id] = std

        self._norm_mean = np.zeros_like(self.x)
        self._norm_std = np.zeros_like(self.x)
        for i in range(len(self.x)):
            domain_id = self.d[i]
            self._norm_mean[i] = self.day_mean.get(domain_id, 0)
            self._norm_std[i] = self.day_std.get(domain_id, 1)

    def _precompute_subject_norm(self, all_x, all_d):
        """预计算每个被试的均值和标准差（仅 3D 波形）"""
        if self.all_sub is None:
            raise ValueError("subject norm mode requires sub.npy file")

        unique_subjects = np.unique(self.all_sub)
        self.sub_mean = {}
        self.sub_std = {}

        for sub_id in unique_subjects:
            mask = self.all_sub == sub_id
            sub_data = all_x[mask]
            mean = sub_data.mean(axis=(0, 1))
            std = sub_data.std(axis=(0, 1))
            std = np.maximum(std, 1e-6)
            self.sub_mean[sub_id] = mean
            self.sub_std[sub_id] = std

        self._norm_mean = np.zeros_like(self.x)
        self._norm_std = np.zeros_like(self.x)
        for i in range(len(self.x)):
            sub_id = self.sub[i]
            self._norm_mean[i] = self.sub_mean.get(sub_id, 0)
            self._norm_std[i] = self.sub_std.get(sub_id, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.norm_mode == "sample":
            data = norm(self.x[index])
        elif hasattr(self, '_norm_mean'):
            data = norm_with_stats(self.x[index], self._norm_mean[index], self._norm_std[index])
        else:
            # 2D DE 特征的 day/subject 模式目前退化为逐样本
            data = norm(self.x[index])
        label = self.y[index].astype(np.float32)
        domain = self.d[index]
        return data, label, domain
