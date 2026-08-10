#dataset.py
import tarfile
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def one_to_n(n):
    """ Return [1, 2, 3, ..., n] """
    return list(range(1, n+1))


def zero_to_n(n):
    return list(range(0, n+1))


def norm(z):
    # print(np.mean(z, axis=0).shape)
    z = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-6)
    return z


def norm_with_stats(z, mean, std):
    """Normalize with pre-computed statistics."""
    return (z - mean) / (std + 1e-6)


class Dataset1:
    """
    Base class for datasets

    class Something(Dataset):
        num_classes = 2
        class_labels = ["class1", "class2"]
        window_size = 250
        window_overlap = False

        def __init__(self, *args, **kwargs):
            super().__init__(Something.num_classes, Something.class_labels,
                Something.window_size, Something.window_overlap,
                *args, **kwargs)

        def process(self, data, labels):
            ...
            return super().process(data, labels)

        def load(self):
            ...
            return train_data, train_labels, test_data, test_labels

    Also, add to the datasets={"something": Something, ...} dictionary below.
    """
    already_normalized = False

    def __init__(self, num_classes, class_labels, window_size, window_overlap,
            feature_names=None, test_percent=0.2):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).

        For example,
            Dataset(num_classes=2, class_labels=["class1", "class2"])

        This calls load() to get the data, process() to normalize, convert to
        float, etc.

        At the end, look at dataset.{train,test}_{data,labels}
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.feature_names = feature_names
        self.test_percent = test_percent

        # Load the dataset
        self.x, self.y = self.load()
        print(len(self.x[0][0]))
        print(len(self.y))
        # train_data, train_labels, test_data, test_labels = self.load()
        #
        # if train_data is not None and train_labels is not None:
        #     self.train_data, self.train_labels = \
        #         self.process(train_data, train_labels)
        # else:
        #     self.train_data = None
        #     self.train_labels = None
        #
        # if test_data is not None and test_labels is not None:
        #     self.test_data, self.test_labels = \
        #         self.process(test_data, test_labels)
        # else:
        #     self.test_data = None
        #     self.test_labels = None

    def load(self):
        raise NotImplementedError("must implement load() for Dataset class")

    def download_dataset(self, files_to_download, url):
        """
        Download url/file for file in files_to_download
        Returns: the downloaded filenames for each of the files given
        """
        downloaded_files = []

        for f in files_to_download:
            downloaded_files.append(tf.keras.utils.get_file(
                fname=f, origin=url+"/"+f))

        return downloaded_files

    def process(self, data, labels):
        """ Perform conversions, etc. If you override,
        you should `return super().process(data, labels)` to make sure these
        options are handled. """
        return data, labels

    def train_test_split(self, x, y, random_state=42):
        """
        Split x and y data into train/test sets

        Warning: train_test_split() is from sklearn but self.train_test_split()
        is this function, which is what you should use.
        """
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=self.test_percent,
            stratify=y, random_state=random_state)
        return x_train, y_train, x_test, y_test

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def create_windows_x(self, x, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process x (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        x = np.expand_dims(x, axis=1)

        # No work required if the window size is 1, only part required is
        # the above expand dims
        if window_size == 1:
            return x

        windows_x = []
        i = 0

        while i < len(x)-window_size:
            window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
            windows_x.append(window_x)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        return np.vstack(windows_x)

    def create_windows_y(self, y, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process y (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        # No work required if the window size is 1
        if window_size == 1:
            return y

        windows_y = []
        i = 0

        while i < len(y)-window_size:
            window_y = y[i+window_size-1]
            windows_y.append(window_y)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        return np.hstack(windows_y)

    def create_windows(self, x, y, window_size, overlap):
        """ Split time-series data into windows """
        x = self.create_windows_x(x, window_size, overlap)
        y = self.create_windows_y(y, window_size, overlap)
        return x, y

    def pad_to(self, data, desired_length):
        """
        Pad the number of time steps to the desired length

        Accepts data in one of two formats:
            - shape: (time_steps, features) -> (desired_length, features)
            - shape: (batch_size, time_steps, features) ->
                (batch_size, desired_length, features)
        """
        if len(data.shape) == 2:
            current_length = data.shape[0]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        elif len(data.shape) == 3:
            current_length = data.shape[1]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, 0), (0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        else:
            raise NotImplementedError("pad_to requires 2 or 3-dim data")

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


class UciHarBase(Dataset1):
    """
    Loads human activity recognition data files in datasets/UCI HAR Dataset.zip

    Download from:
    https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    """
    feature_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    num_classes = 6
    class_labels = [
        "walking", "walking_upstairs", "walking_downstairs",
        "sitting", "standing", "laying",
    ]
    users = one_to_n(30)  # 30 people
    already_normalized = True

    def __init__(self, users, *args, **kwargs):
        self.users = users

        super().__init__(UciHarBase.num_classes, UciHarBase.class_labels,
            None, None, UciHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["UCI%20HAR%20Dataset.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240")
        return dataset_fp

    def get_feature(self, content):
        """
        Read the space-separated, example on each line file

        Returns 2D array with dimensions: [num_examples, num_time_steps]
        """
        lines = content.decode("utf-8").strip().split("\n")
        features = []

        for line in lines:
            features.append([float(v) for v in line.strip().split()])

        return features

    def get_data(self, archive, name):
        """ To shorten duplicate code for name=train or name=test cases """
        def get_data_single(f):
            return self.get_feature(self.get_file_in_archive(archive,
                "UCI HAR Dataset/"+f))

        data = [
            get_data_single(name+"/Inertial Signals/body_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_z_"+name+".txt"),
        ]

        labels = get_data_single(name+"/y_"+name+".txt")

        subjects = get_data_single(name+"/subject_"+name+".txt")

        data = np.array(data, dtype=np.float32)
        labels = np.squeeze(np.array(labels, dtype=np.float32))
        # Squeeze so we can easily do selection on this later on
        subjects = np.squeeze(np.array(subjects, dtype=np.float32))

        # Transpose from [features, examples, time_steps] to
        # [examples, time_steps (128), features (9)]
        data = np.transpose(data, axes=[1, 2, 0])

        return data, labels, subjects

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            train_data, train_labels, train_subjects = self.get_data(archive, "train")
            test_data, test_labels, test_subjects = self.get_data(archive, "test")

        all_data = np.vstack([train_data, test_data]).astype(np.float32)
        all_labels = np.hstack([train_labels, test_labels]).astype(np.float32)
        all_subjects = np.hstack([train_subjects, test_subjects]).astype(np.float32)

        # All data if no selection
        if self.users is None:
            return all_data, all_labels

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            selection = all_subjects == user
            data.append(all_data[selection])
            current_labels = all_labels[selection]
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        # train_data, train_labels, test_data, test_labels = \
        #     self.train_test_split(x, y)
        # return train_data, train_labels, test_data, test_labels
        return x, y

    def process(self, data, labels):
        # Index one
        labels = labels - 1
        return super().process(data, labels)


class WisdmBase(Dataset1):
    """
    Base class for the WISDM datasets
    http://www.cis.fordham.edu/wisdm/dataset.php
    """
    feature_names = [
        "acc_x", "acc_y", "acc_z",
    ]
    window_size = 128  # similar to HAR
    window_overlap = False

    def __init__(self, users, num_classes, class_labels, *args, **kwargs):
        self.users = users
        super().__init__(num_classes, class_labels,
            WisdmBase.window_size, WisdmBase.window_overlap,
            WisdmBase.feature_names, *args, **kwargs)
        # Override and set these
        #self.filename_prefix = ""
        #self.download_filename = ""

    def download(self):
        (dataset_fp,) = self.download_dataset([self.download_filename],
            "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/")
        return dataset_fp

    def read_data(self, lines, user_list):
        """ Read the raw data CSV file """
        data_x = []
        data_label = []
        data_subject = []

        for line in lines:
            parts = line.strip().replace(";", "").split(",")

            # For some reason there's blank rows in the data, e.g.
            # a bunch of lines like "577,,;"
            # Though, allow 7 since sometimes there's an extra comma at the end:
            # "21,Jogging,117687701514000,3.17,9,1.23,;"
            if len(parts) != 6 and len(parts) != 7:
                continue

            # Skip if x, y, or z is blank
            if parts[3] == "" or parts[4] == "" or parts[5] == "":
                continue

            user = int(parts[0])

            # Skip users that may not have enough data
            if user in user_list:
                user = user_list.index(user)  # non-consecutive to consecutive

                # Skip users we don't care about
                if user in self.users:
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    label = self.class_labels.index(parts[1])  # name --> number

                    data_x.append((x, y, z))
                    data_label.append(label)
                    data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def read_user_list(self, lines, min_test_samples=30):
        """ Read first column of the CSV file to get a unique list of uid's
        Also, skip users with too few samples """
        user_sample_count = {}

        for line in lines:
            parts = line.strip().split(",")

            # There's some lines without the right number of parts, e.g. blank
            if len(parts) != 6 and len(parts) != 7:
                continue

            # Skip if x, y, or z is blank
            if parts[3] == "" or parts[4] == "" or parts[5] == "":
                continue

            uid = int(parts[0])

            # There are duplicates in the file for some reason (so, either the
            # same person or it's not truly unique)
            if uid not in user_sample_count:
                user_sample_count[uid] = 0
            else:
                user_sample_count[uid] += 1

        # Remove users with too few samples
        user_list = []

        # How many samples we need: to stratify the sklearn function says
        # The test_size = A should be greater or equal to the number of classes = B
        # x/128*.2 > 6 classes
        # x > 6*128/.2
        # Though, actually, just set the minimum test samples. It's probably not
        # enough to have only 7...
        test_percentage = 0.20  # default
        #min_samples = int(len(self.class_labels)*self.window_size/test_percentage)
        min_samples = int(min_test_samples*self.window_size/test_percentage)

        for user, count in user_sample_count.items():
            if count > min_samples:
                user_list.append(user)

        # Data isn't sorted by user in the file
        user_list.sort()

        return user_list

    def get_lines(self, archive, name):
        """ Open and load file in tar file, get lines from file """
        f = archive.extractfile(self.filename_prefix+name)

        if f is None:
            return None

        return f.read().decode("utf-8").strip().split("\n")

    def load_file(self, filename):
        """ Load desired participants' data """
        # Get data
        with tarfile.open(filename, "r") as archive:
            raw_data = self.get_lines(archive, "raw.txt")

        # Some of the data doesn't have a uid in the demographics file? So,
        # instead just get the user list from the raw data. Also, one person
        # have very little data, so skip them (e.g. one person only has 25
        # samples, which is only 0.5 seconds of data -- not useful).
        user_list = self.read_user_list(raw_data)

        #print("Number of users:", len(user_list))

        # For now just use phone data since the positions may differ too much
        all_data, all_labels, all_subjects = self.read_data(raw_data, user_list)

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        # train_data, train_labels, test_data, test_labels = \
        #     self.train_test_split(x, y)
        print(x.shape)
        return x, y


class WisdmArBase(WisdmBase):
    """
    Loads WISDM Activity prediction/recognition dataset
    http://www.cis.fordham.edu/wisdm/dataset.php
    """
    num_classes = 6
    class_labels = [
        "Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs",
    ]
    users = zero_to_n(32)  # 33 people

    def __init__(self, users, *args, **kwargs):
        self.filename_prefix = "WISDM_ar_v1.1/WISDM_ar_v1.1_"
        self.download_filename = "WISDM_ar_latest.tar.gz"
        super().__init__(users,
            WisdmArBase.num_classes, WisdmArBase.class_labels, *args, **kwargs)


class U_data(Dataset):
    def __init__(self, users):
        data = UciHarBase(users)
        self.x = data.x
        self.y = data.y
        for index in range(len(self.y)):
            self.y[index] -= 1
        # self.x = norm(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = self.x[index]
        data = norm(data)
        label = self.y[index]
        return data, label


class Wisdm_ar_data(Dataset):
    def __init__(self, users):
        data = WisdmArBase(users)
        self.x = data.x
        self.y = data.y
        self.x = norm(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return data, label


class DailyEEG_data(Dataset):
    """
    DailyEEG 情绪回归数据集，从 align_cadt_input.py 生成的 NPY 文件加载。
    每个样本:
      - x: (seq_len, n_channels) 原始 EEG 片段
      - y: (n_labels,) 11 维情绪回归标签
      - d: 域 ID
      - sub_id: 被试编号
      - day_id: 天/session 编号
    """
    def __init__(self, domain_ids, data_dir="/root/zjm/cadt_test_code/processed_cadt", id_map=None,
                 norm_mode="sample"):
        """
        Args:
            domain_ids: 域 ID 列表
            data_dir: NPY 数据目录
            id_map: 域 ID 映射
            norm_mode: 标准化模式
                - "sample": 逐样本独立标准化（默认）
                - "day": 按天标准化
                - "subject": 按被试标准化
        """
        import os
        x_path = os.path.join(data_dir, "X.npy")
        y_path = os.path.join(data_dir, "y.npy")
        d_path = os.path.join(data_dir, "d.npy")
        all_x = np.load(x_path).astype(np.float32)
        all_y = np.load(y_path).astype(np.float32)
        all_d = np.load(d_path).astype(np.int64)
        
        # 加载被试信息
        sub_path = os.path.join(data_dir, "sub.npy")
        if os.path.exists(sub_path):
            self.all_sub = np.load(sub_path).astype(np.int64)
        else:
            self.all_sub = None
        
        # 加载 meta.json 获取域偏移映射
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
        domain_set = set(domain_ids)
        mask = np.isin(all_d, list(domain_set)) #筛选需要的domain_id
        self.x = all_x[mask]
        self.y = all_y[mask]
        self.d = all_d[mask]
        self.norm_mode = norm_mode
        
        if self.all_sub is not None:
            self.sub = self.all_sub[mask]
            # 计算 day_id
            self.day = np.zeros_like(self.d)
            for sub_id, offset in self.domain_offset_map.items():
                sub_mask = self.sub == sub_id
                self.day[sub_mask] = self.d[sub_mask] - offset
        else:
            self.sub = None
            self.day = None
            
        if id_map is not None:
            self.d = np.array([id_map[int(d)] for d in self.d], dtype=np.int64)
        
        # 预计算标准化统计量
        if norm_mode == "day":
            print(f"  预计算按天标准化统计量...")
            self._precompute_day_norm(all_x, all_d, mask)
        elif norm_mode == "subject":
            print(f"  预计算按被试标准化统计量...")
            self._precompute_subject_norm(all_x, all_d, mask)
        
        if len(self.x) == 0:
            raise ValueError(
                f"域 {domain_ids} 在 {data_dir} 中没有匹配样本，"
                f"可用域范围 [{all_d.min()}, {all_d.max()}]"
            )
        norm_desc = {"sample": "逐样本", "day": "按天", "subject": "按被试"}
        print(f"DailyEEG_data: domains={domain_ids}, samples={len(self.x)}, "
              f"x={self.x.shape}, y={self.y.shape}, norm={norm_desc.get(norm_mode, norm_mode)}")
    
    def _precompute_day_norm(self, all_x, all_d, mask):
        """预计算每个 (被试, 天) 组合的均值和标准差"""
        unique_domains = np.unique(all_d)
        self.day_mean = {}
        self.day_std = {}
        
        for domain_id in unique_domains:
            domain_mask = all_d == domain_id
            day_data = all_x[domain_mask]
            mean = day_data.mean(axis=(0, 1))
            std = day_data.std(axis=(0, 1))
            std = np.maximum(std, 1e-6)
            self.day_mean[domain_id] = mean
            self.day_std[domain_id] = std
        
        # 为选中的样本分配统计量
        selected_indices = np.where(mask)[0]
        self._norm_mean = np.zeros_like(self.x)
        self._norm_std = np.zeros_like(self.x)
        for i, idx in enumerate(selected_indices):
            domain_id = all_d[idx]
            self._norm_mean[i] = self.day_mean.get(domain_id, 0)
            self._norm_std[i] = self.day_std.get(domain_id, 1)
    
    def _precompute_subject_norm(self, all_x, all_d, mask):
        """预计算每个被试的均值和标准差"""
        if self.all_sub is None:
            raise ValueError("subject norm mode requires sub.npy file")
        
        unique_subjects = np.unique(self.all_sub)
        self.sub_mean = {}
        self.sub_std = {}
        
        for sub_id in unique_subjects:
            sub_mask = self.all_sub == sub_id
            sub_data = all_x[sub_mask]
            mean = sub_data.mean(axis=(0, 1))
            std = sub_data.std(axis=(0, 1))
            std = np.maximum(std, 1e-6)
            self.sub_mean[sub_id] = mean
            self.sub_std[sub_id] = std
        
        # 为选中的样本分配统计量
        selected_indices = np.where(mask)[0]
        self._norm_mean = np.zeros_like(self.x)
        self._norm_std = np.zeros_like(self.x)
        for i, idx in enumerate(selected_indices):
            sub_id = self.all_sub[idx]
            self._norm_mean[i] = self.sub_mean.get(sub_id, 0)
            self._norm_std[i] = self.sub_std.get(sub_id, 1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.norm_mode == "sample":
            data = norm(self.x[index])
        else:
            data = norm_with_stats(self.x[index], self._norm_mean[index], self._norm_std[index])
        label = self.y[index].astype(np.float32)
        domain = self.d[index]
        return data, label, domain


class MultiSourceBatchSampler:
    """
    多源域Batch采样器：从每个源域取相同数量的样本组成一个batch。
    
    用法:
        dataset = DailyEEG_data([6, 12, 18])
        sampler = MultiSourceBatchSampler(dataset.d, [6, 12, 18], batch_size=32)
        loader = DataLoader(dataset, batch_sampler=sampler)
    
    参数:
        domain_labels: 每个样本的域标签数组，shape=(n_samples,)
        source_domains: 源域ID列表，如 [6, 12, 18]
        batch_size: 总的batch大小
    
    特性:
        - 每个源域贡献 batch_size // n_domains 个样本
        - 如果batch_size不能被源域数量整除，实际batch大小为 (batch_size // n_domains) * n_domains
        - 数据量少的源域自动循环采样（使用itertools.cycle）
        - 支持任意数量的源域
    """
    def __init__(self, domain_labels, source_domains, batch_size):
        self.source_domains = source_domains
        self.n_domains = len(source_domains)
        self.total_batch_size = batch_size
        self.per_domain_batch = batch_size // self.n_domains
        
        if self.per_domain_batch == 0:
            raise ValueError(
                f"batch_size={batch_size} 小于源域数量={self.n_domains}，"
                f"请增大batch_size或减少源域数量"
            )
        
        # 实际batch大小
        self.actual_batch_size = self.per_domain_batch * self.n_domains
        
        # 为每个源域收集样本索引
        self.domain_indices = {}
        for domain in source_domains:
            indices = np.where(domain_labels == domain)[0].tolist()
            if len(indices) == 0:
                raise ValueError(f"源域 {domain} 没有样本")
            self.domain_indices[domain] = indices
            print(f"  源域 {domain}: {len(indices)} 个样本")
        
        print(f"MultiSourceBatchSampler: {self.n_domains} 个源域, "
              f"总batch_size={batch_size}, 实际batch_size={self.actual_batch_size}, "
              f"每域样本数={self.per_domain_batch}")
        
        # 计算一个epoch的batch数量（基于最大的源域）
        max_samples = max(len(indices) for indices in self.domain_indices.values())
        self.n_batches = max_samples // self.per_domain_batch
        
        if self.n_batches == 0:
            raise ValueError(
                f"某个源域样本数不足 {self.per_domain_batch}，无法组成一个batch"
            )
    
    def __iter__(self):
        import itertools
        
        # 为每个源域创建随机打乱的迭代器（循环使用）
        domain_iterators = {}
        for domain, indices in self.domain_indices.items():
            # 每次迭代前打乱
            shuffled_indices = indices.copy()
            np.random.shuffle(shuffled_indices)
            # 创建循环迭代器
            domain_iterators[domain] = itertools.cycle(shuffled_indices)
        
        # 生成batch
        for _ in range(self.n_batches):
            batch_indices = []
            for domain in self.source_domains:
                # 从每个源域取per_domain_batch个样本
                for _ in range(self.per_domain_batch):
                    batch_indices.append(next(domain_iterators[domain]))
            yield batch_indices
    
    def __len__(self):
        return self.n_batches


def get_available_domains(data_dir="/root/zjm/cadt_test_code/processed_cadt"):
    """
    Reads d.npy directly instead of assuming a contiguous 0..N range --
    domain 124 is confirmed missing, and there may be other gaps.
    """
    import os
    d_path = os.path.join(data_dir, "d.npy")
    all_d = np.load(d_path)
    return sorted(np.unique(all_d).tolist())


def load_emotion_split(split_dir, split_mode, split_type):
    """
    Load emotion project's split JSON file which contains sample indices.
    
    Args:
        split_dir: Path to splits directory (e.g., "/vePFS-0x0d/DailyEEG/splits_new")
        split_mode: One of "cross_subject", "cross_day", "within_subject_day"
        split_type: One of "pretrain", "finetune", "val", "test"
    
    Returns:
        List of sample indices (integers)
    """
    import json
    import os
    if split_type in ["pretrain", "finetune"]:
        json_path = os.path.join(split_dir, split_mode, f"{split_type}.json")
    else:
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
        # 使用 cross_subject 模式
        train_indices = load_emotion_split("vePFS-0x0d/DailyEEG/splits_new", "cross_subject", "pretrain")
        train_indices += load_emotion_split("vePFS-0x0d/DailyEEG/splits_new", "cross_subject", "finetune")
        data_train = DailyEEG_SplitData(train_indices, split_mode="cross_subject", data_dir="vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
    """
    def __init__(self, sample_indices, split_mode="cross_subject", data_dir="/root/zjm/cadt_test_code/processed_cadt",
                 norm_mode="sample"):
        """
        Args:
            sample_indices: 样本索引列表
            split_mode: cross_subject, cross_day, within_subject_day
            data_dir: NPY 数据目录
            norm_mode: 标准化模式
                - "sample": 逐样本独立标准化（默认）
                - "day": 按天标准化（每个被试的每一天使用该天所有样本统计量）
                - "subject": 按被试标准化（每个被试使用该被试所有样本统计量）
        """
        import os
        x_path = os.path.join(data_dir, "X.npy")
        y_path = os.path.join(data_dir, "y.npy")
        d_path = os.path.join(data_dir, "d.npy")
        
        all_x = np.load(x_path).astype(np.float32)
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
        
        # 预计算标准化统计量
        if norm_mode == "day":
            # 按天标准化：使用完整数据中每个 (被试, 天) 的统计量
            print(f"  预计算按天标准化统计量...")
            self._precompute_day_norm(all_x, all_d)
        elif norm_mode == "subject":
            # 按被试标准化：使用完整数据中每个被试的统计量
            print(f"  预计算按被试标准化统计量...")
            self._precompute_subject_norm(all_x, all_d)
        
        norm_desc = {"sample": "逐样本", "day": "按天", "subject": "按被试"}
        print(f"DailyEEG_SplitData ({split_mode}): {len(sample_indices)} samples, "
              f"x={self.x.shape}, y={self.y.shape}, "
              f"domain range [{self.d.min()}, {self.d.max()}], "
              f"norm={norm_desc.get(norm_mode, norm_mode)}")
    
    def _precompute_day_norm(self, all_x, all_d):
        """预计算每个 (被试, 天) 组合的均值和标准差"""
        # 计算每个域 ID 的均值和标准差
        unique_domains = np.unique(all_d)
        self.day_mean = {}
        self.day_std = {}
        
        for domain_id in unique_domains:
            mask = all_d == domain_id
            day_data = all_x[mask]  # (n_samples, 2000, 59)
            # 计算该天所有样本合并后的均值和标准差（沿时间轴和样本轴）
            mean = day_data.mean(axis=(0, 1))  # (59,)
            std = day_data.std(axis=(0, 1))    # (59,)
            std = np.maximum(std, 1e-6)  # 防止标准差为0
            self.day_mean[domain_id] = mean
            self.day_std[domain_id] = std
        
        # 为每个样本分配对应的统计量
        self._norm_mean = np.zeros_like(self.x)
        self._norm_std = np.zeros_like(self.x)
        for i in range(len(self.x)):
            domain_id = self.d[i]
            self._norm_mean[i] = self.day_mean.get(domain_id, 0)
            self._norm_std[i] = self.day_std.get(domain_id, 1)
    
    def _precompute_subject_norm(self, all_x, all_d):
        """预计算每个被试的均值和标准差"""
        if self.all_sub is None:
            raise ValueError("subject norm mode requires sub.npy file")
        
        unique_subjects = np.unique(self.all_sub)
        self.sub_mean = {}
        self.sub_std = {}
        
        for sub_id in unique_subjects:
            mask = self.all_sub == sub_id
            sub_data = all_x[mask]  # (n_samples, 2000, 59)
            mean = sub_data.mean(axis=(0, 1))  # (59,)
            std = sub_data.std(axis=(0, 1))    # (59,)
            std = np.maximum(std, 1e-6)
            self.sub_mean[sub_id] = mean
            self.sub_std[sub_id] = std
        
        # 为每个样本分配对应的统计量
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
        else:
            # 使用预计算的统计量
            data = norm_with_stats(self.x[index], self._norm_mean[index], self._norm_std[index])
        label = self.y[index].astype(np.float32)
        domain = self.d[index]
        return data, label, domain


