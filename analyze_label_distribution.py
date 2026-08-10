#!/usr/bin/env python3
"""
分析 DailyEEG 数据每个情绪维度的分布情况
查看训练/验证/测试集的标签分布
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/root/zjm/cadt')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset import load_emotion_split

EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]


def analyze_split(y, split_name):
    n_samples, n_dims = y.shape

    sep = '=' * 70
    print()
    print(sep)
    print('【%s】共 %d 个样本, %d 个维度' % (split_name, n_samples, n_dims))
    print(sep)

    header = '%-12s %10s %10s %10s %10s %10s' % ('Label', 'Mean', 'Std', 'Min', 'Max', 'Var')
    print(header)
    print('-' * 62)

    for i in range(n_dims):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else 'Dim_%d' % i
        mean = np.mean(y[:, i])
        std = np.std(y[:, i])
        min_val = np.min(y[:, i])
        max_val = np.max(y[:, i])
        var = np.var(y[:, i])
        print('%-12s %10.4f %10.4f %10.4f %10.4f %10.4f' % (label, mean, std, min_val, max_val, var))

    print('-' * 62)
    print('%-12s %10.4f %10.4f %10.4f %10.4f %10.4f' % ('Overall', np.mean(y), np.std(y), np.min(y), np.max(y), np.var(y)))

    print()
    print('【维度间相关性矩阵 (Pearson)】')
    corr = np.corrcoef(y.T)
    print('%-12s' % '', end='')
    for l in EMOTION_LABELS[:5]:
        print('%12s' % l, end='')
    print()
    for i in range(min(5, n_dims)):
        print('%-12s' % EMOTION_LABELS[i], end='')
        for j in range(min(5, n_dims)):
            print('%12.4f' % corr[i, j], end='')
        print()

    return {
        'mean': np.mean(y, axis=0),
        'std': np.std(y, axis=0),
        'var': np.var(y, axis=0),
        'min': np.min(y, axis=0),
        'max': np.max(y, axis=0)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析 DailyEEG 数据标签分布')
    parser.add_argument('--data_dir',
                        default='/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new',
                        help='数据目录路径')
    parser.add_argument('--split_dir',
                        default='/vePFS-0x0d/DailyEEG/splits_new',
                        help='分割目录路径')
    parser.add_argument('--split_mode',
                        default='cross_day',
                        choices=['cross_subject', 'cross_day', 'within_subject_day'],
                        help='分割模式')
    parser.add_argument('--target_subject',
                        type=int, default=None,
                        help='目标被试 ID (用于 cross_day/within_subject_day 模式)')
    args = parser.parse_args()

    data_dir = args.data_dir
    split_dir = args.split_dir
    split_mode = args.split_mode

    print('数据目录: %s' % data_dir)
    print('分割目录: %s' % split_dir)
    print('分割模式: %s' % split_mode)

    print()
    print('加载 %s/y.npy ...' % data_dir)
    all_y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.float32)
    print('完整数据: %s' % str(all_y.shape))

    print()
    print('加载分割索引...')
    train_idx = load_emotion_split(split_dir, split_mode, 'pretrain')
    train_idx += load_emotion_split(split_dir, split_mode, 'finetune')
    val_idx = load_emotion_split(split_dir, split_mode, 'val')
    test_idx = load_emotion_split(split_dir, split_mode, 'test')

    print('  训练集: %d 样本' % len(train_idx))
    print('  验证集: %d 样本' % len(val_idx))
    print('  测试集: %d 样本' % len(test_idx))

    if args.target_subject is not None:
        print()
        print('过滤目标被试: %d' % args.target_subject)
        sub_path = os.path.join(data_dir, 'sub.npy')
        if os.path.exists(sub_path):
            all_sub = np.load(sub_path).astype(np.int64)
            train_idx = [i for i in train_idx if all_sub[i] == args.target_subject]
            val_idx = [i for i in val_idx if all_sub[i] == args.target_subject]
            test_idx = [i for i in test_idx if all_sub[i] == args.target_subject]
            print('  过滤后训练集: %d 样本' % len(train_idx))
            print('  过滤后验证集: %d 样本' % len(val_idx))
            print('  过滤后测试集: %d 样本' % len(test_idx))
        else:
            print('  警告: 找不到 sub.npy，无法按被试过滤')

    y_train = all_y[train_idx]
    y_val = all_y[val_idx]
    y_test = all_y[test_idx]

    stats_train = analyze_split(y_train, '训练集 (train, %d samples)' % len(train_idx))
    stats_val = analyze_split(y_val, '验证集 (val, %d samples)' % len(val_idx))
    stats_test = analyze_split(y_test, '测试集 (test, %d samples)' % len(test_idx))

    print()
    print('=' * 70)
    print('【数据集对比】')
    print('=' * 70)

    print()
    print('各维度均值对比:')
    print('%-12s %10s %10s %10s %10s' % ('Label', 'Train', 'Val', 'Test', '差异'))
    print('-' * 52)
    for i in range(min(11, all_y.shape[1])):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else 'Dim_%d' % i
        t_mean = stats_train['mean'][i]
        v_mean = stats_val['mean'][i]
        te_mean = stats_test['mean'][i]
        diff = abs(t_mean - te_mean)
        print('%-12s %10.4f %10.4f %10.4f %10.4f' % (label, t_mean, v_mean, te_mean, diff))

    print()
    print('各维度方差对比:')
    print('%-12s %10s %10s %10s' % ('Label', 'Train', 'Val', 'Test'))
    print('-' * 42)
    for i in range(min(11, all_y.shape[1])):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else 'Dim_%d' % i
        t_var = stats_train['var'][i]
        v_var = stats_val['var'][i]
        te_var = stats_test['var'][i]
        print('%-12s %10.4f %10.4f %10.4f' % (label, t_var, v_var, te_var))

    print()
    print('【关键分析】全局均值 vs 各维度均值的偏差')
    global_mean = np.mean(all_y)
    print('  全局均值: %.4f' % global_mean)

    print()
    print('%-12s %10s %10s %10s' % ('Label', 'Dim Mean', '偏差', '贡献'))
    print('-' * 44)
    for i in range(min(11, all_y.shape[1])):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else 'Dim_%d' % i
        dim_mean = stats_train['mean'][i]
        deviation = dim_mean - global_mean
        contribution = abs(deviation) * len(y_train)
        print('%-12s %10.4f %+10.4f %10.2f' % (label, dim_mean, deviation, contribution))

    print()
    print('  贡献 指该维度均值与全局均值的偏差对 SS_tot 的贡献')
    print('  如果某维度偏差大，整体 R2 会被虚高!')


if __name__ == '__main__':
    main()
