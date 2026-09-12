"""
CADT 模型预测结果可视化 - 时序折线图
真实值 vs 预测值 对比图
"""

import numpy as np
import torch
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# 将 cadt 项目根目录添加到 sys.path（model/ dataset 等模块在此目录下）
sys.path.insert(0, '/vePFS-0x0d/home/zjm/cadt_regression')

from model.CADT import CATDModel
from dataset.dataset import DailyEEG_SplitData, load_emotion_split
from torch.utils.data import DataLoader

# 情绪标签名称
EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]

# 颜色方案
COLOR_TRUE = '#2196F3'      # 蓝色 - 真实值
COLOR_PRED = '#FF5722'      # 橙红色 - 预测值
COLOR_ERROR = '#4CAF50'     # 绿色 - 误差填充

def load_model(model_path, args):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, weights_only=True)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # --- 检测 encoder_type ---
    has_mlp = any(k.startswith('share_encoder.trunk') or
                  k.startswith('share_encoder.head_zsc') for k in state_dict)
    has_cnn = any(k.startswith('share_encoder.feature_') or
                  k.startswith('share_encoder.conv') or
                  k.startswith('share_encoder.fc11') for k in state_dict)
    if has_mlp and not has_cnn:
        encoder_type = 'mlp'
    elif has_cnn and not has_mlp:
        encoder_type = 'cnn'
    else:
        encoder_type = getattr(args, 'encoder_type', 'cnn')
    args.encoder_type = encoder_type
    print(f"  encoder_type = {encoder_type}")

    # --- 检测 t_size 和 x_dim / seq_len ---
    t_size_detected = None
    x_dim_detected = None
    seq_len_detected = None

    if encoder_type == 'mlp':
        # MLP: t_size 从 head_zsc 最后一层 Linear 推断
        for key in state_dict:
            if key == 'share_encoder.head_zsc.3.weight':
                t_size_detected = state_dict[key].shape[0]
                break
        # x_dim * seq_len 从 trunk.0.weight 推断
        for key in state_dict:
            if key == 'share_encoder.trunk.0.weight':
                in_dim = state_dict[key].shape[1]
                x_dim_detected = in_dim  # 默认 seq_len=1
                seq_len_detected = 1
                break
    else:  # cnn
        for key in state_dict:
            if key == 'share_encoder.fc11.0.weight':
                t_size_detected = state_dict[key].shape[0]
                break
        x_dim_detected = 59
        seq_len_detected = 2000

    # t_size 处理
    if getattr(args, 't_size', None) is not None:
        t_size = args.t_size
    elif t_size_detected is not None:
        t_size = t_size_detected
    else:
        t_size = 8
    args.t_size = t_size

    # x_dim / seq_len 处理
    if getattr(args, 'x_dim', None) is not None:
        args.x_dim = args.x_dim
    elif x_dim_detected is not None:
        args.x_dim = x_dim_detected
    else:
        args.x_dim = 59

    if getattr(args, 'seq_len', None) is not None:
        args.seq_len = args.seq_len
    elif seq_len_detected is not None:
        args.seq_len = seq_len_detected
    else:
        args.seq_len = 2000

    print(f"  t_size = {args.t_size}, x_dim = {args.x_dim}, seq_len = {args.seq_len}")

    # --- 检测 n_sources ---
    n_sources = 1
    for key in state_dict.keys():
        if 'discriminator.model.4.weight' in key:
            n_sources = state_dict[key].shape[0]
            break
    args.n_sources = n_sources

    # --- 创建模型 ---
    model = CATDModel(args).cuda()

    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    print(f"模型加载成功: {model_path}")
    return model

def get_predictions(model, data_loader):
    """获取真实值和预测值"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xs, ys, ds in data_loader:
            xs = xs.to(torch.float32).cuda()
            ys = ys.to(torch.float32).cuda()
            pred_y = model.predict_target(xs)
            all_preds.append(pred_y.cpu().numpy())
            all_targets.append(ys.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"预测值形状: {all_preds.shape}, 真实值形状: {all_targets.shape}")
    return all_preds, all_targets

def compute_metrics(preds, targets):
    """计算每种情绪的指标"""
    from scipy.stats import pearsonr
    
    n_emotions = preds.shape[1]
    metrics = []
    
    for i in range(n_emotions):
        pred = preds[:, i].flatten()
        target = targets[:, i].flatten()
        
        corr, _ = pearsonr(pred, target)
        pearson = 0.0 if np.isnan(corr) else np.clip(corr, -1.0, 1.0)
        
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        ss_res = np.sum((target - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        mse = np.mean((pred - target) ** 2)
        
        metrics.append({
            'emotion': EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f'Emotion_{i}',
            'pearson': pearson,
            'r2': r2,
            'mse': mse
        })
    
    return metrics

def plot_timeseries_overall(preds, targets, metrics, output_dir):
    """
    图1: 整体时序折线图
    X轴: 样本序号
    Y轴: 情绪值
    真实值线 + 预测值线叠加
    """
    n_emotions = preds.shape[1]
    n_samples = preds.shape[0]
    
    # 计算每个情绪的 R² 排序，用于选择最好/最差情绪
    r2_values = [(i, m['r2']) for i, m in enumerate(metrics)]
    r2_values.sort(key=lambda x: x[1], reverse=True)
    
    best_idx = r2_values[0][0]   # R² 最高的情绪
    worst_idx = r2_values[-1][0]  # R² 最低的情绪
    
    fig = plt.figure(figsize=(20, 8))
    
    # 对所有情绪的真实值和预测值做均值，画整体趋势
    mean_true = targets.mean(axis=1)
    mean_pred = preds.mean(axis=1)
    
    samples = np.arange(n_samples)
    
    plt.plot(samples, mean_true, color=COLOR_TRUE, linewidth=1.5, 
             alpha=0.7, label='True Mean (All Emotions)')
    plt.plot(samples, mean_pred, color=COLOR_PRED, linewidth=1.5, 
             alpha=0.7, label='Predicted Mean (All Emotions)')
    
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Emotion Intensity (Mean across 11 dimensions)', fontsize=13)
    plt.title('Overall Prediction Trend (True vs Predicted)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: overall_timeseries.png")
    
    # 同时画一张"均值 ± 标准差"的阴影图：跨 11 个情绪维度计算 std，用 fill_between 显示波动范围
    std_true = targets.std(axis=1)
    std_pred = preds.std(axis=1)

    fig, ax = plt.subplots(figsize=(20, 8))

    ax.fill_between(samples, mean_true - std_true, mean_true + std_true,
                    color=COLOR_TRUE, alpha=0.15, linewidth=0)
    ax.fill_between(samples, mean_pred - std_pred, mean_pred + std_pred,
                    color=COLOR_PRED, alpha=0.15, linewidth=0)
    ax.plot(samples, mean_true, color=COLOR_TRUE, linewidth=1.5, label='True Mean ± Std')
    ax.plot(samples, mean_pred, color=COLOR_PRED, linewidth=1.5, label='Predicted Mean ± Std')

    ax.set_xlabel('Sample Index', fontsize=14)
    ax.set_ylabel('Emotion Intensity (Mean ± Std across 11 dimensions)', fontsize=13)
    ax.set_title('Overall Prediction Trend with Standard Deviation Band', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_timeseries_std.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: overall_timeseries_std.png")
    
    return best_idx, worst_idx

def plot_timeseries_per_emotion(preds, targets, metrics, output_dir):
    """
    图2: 每种情绪的子图
    4×3 网格（11个情绪 + 1个汇总误差图）
    每个子图: X=样本序号, Y=情绪值, 真实值线 + 预测值线
    """
    n_emotions = preds.shape[1]
    n_samples = preds.shape[0]
    samples = np.arange(n_samples)
    
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    for i in range(n_emotions):
        ax = fig.add_subplot(gs[i])
        pred = preds[:, i]
        target = targets[:, i]
        m = metrics[i]
        
        # 真实值线
        ax.plot(samples, target, color=COLOR_TRUE, linewidth=0.8, 
                alpha=0.8, label='True', rasterized=True)
        # 预测值线
        ax.plot(samples, pred, color=COLOR_PRED, linewidth=0.8, 
                alpha=0.8, label='Predicted', rasterized=True)
        
        ax.set_title(f'{m["emotion"]}\nR²={m["r2"]:.3f} | Pearson={m["pearson"]:.3f} | MSE={m["mse"]:.4f}', 
                     fontsize=10)
        ax.set_xlabel('Sample Index', fontsize=8)
        ax.set_ylabel('Emotion Value', fontsize=8)
        ax.set_xlim(0, n_samples)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')
    
    # 第12格: 汇总误差分布图
    ax_error = fig.add_subplot(gs[11])
    errors = np.abs(preds - targets)
    mean_error_per_sample = errors.mean(axis=1)
    
    ax_error.plot(samples, mean_error_per_sample, color=COLOR_ERROR, linewidth=0.8)
    ax_error.set_title('Mean Absolute Error per Sample', fontsize=10)
    ax_error.set_xlabel('Sample Index', fontsize=8)
    ax_error.set_ylabel('MAE', fontsize=8)
    ax_error.set_xlim(0, n_samples)
    ax_error.grid(True, alpha=0.2)
    
    fig.suptitle('Per-Emotion Prediction: True vs Predicted Time Series', 
                 fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_emotion_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: per_emotion_timeseries.png")

def plot_zoomed_windows(preds, targets, metrics, output_dir, best_idx, worst_idx, window_size=100):
    """
    图3: 局部放大窗口
    选取连续样本窗口，清晰展示预测贴合度
    - R² 最高情绪的窗口
    - R² 最低情绪的窗口
    - 随机选取2个情绪的窗口
    """
    n_samples = preds.shape[0]
    n_emotions = preds.shape[1]
    
    # 选择窗口起始位置
    if n_samples > window_size * 4:
        start_positions = [
            (0, 'Start'),
            (n_samples // 4, 'Quarter'),
            (n_samples // 2, 'Mid'),
            (3 * n_samples // 4, 'Three-Quarter')
        ]
    else:
        start_positions = [(0, 'Full Range')]
        window_size = n_samples
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    # 选择要展示的情绪: 最佳 + 最差 + 随机2个
    r2_order = sorted(range(n_emotions), key=lambda i: metrics[i]['r2'], reverse=True)
    display_emotions = [r2_order[0], r2_order[-1]]
    # 再加两个随机的
    remaining = [i for i in range(n_emotions) if i not in display_emotions]
    np.random.seed(42)
    display_emotions.extend(np.random.choice(remaining, min(2, len(remaining)), replace=False).tolist())
    
    for idx, emo_idx in enumerate(display_emotions):
        ax = axes[idx]
        pred = preds[:, emo_idx]
        target = targets[:, emo_idx]
        m = metrics[emo_idx]
        
        # 选取窗口
        start = start_positions[idx % len(start_positions)][0]
        end = min(start + window_size, n_samples)
        local_samples = np.arange(start, end)
        
        ax.plot(local_samples, target[start:end], color=COLOR_TRUE, 
                linewidth=2, label='True')
        ax.plot(local_samples, pred[start:end], color=COLOR_PRED, 
                linewidth=2, alpha=0.8, label='Predicted')
        
        position_label = start_positions[idx % len(start_positions)][1]
        ax.set_title(f'{m["emotion"]} | Window: {position_label} [{start}-{end}]\n'
                     f'R²={m["r2"]:.3f} | MAE={m["mse"]:.4f}', fontsize=11)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Emotion Value', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Zoomed View: Prediction Accuracy in Detail', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zoomed_windows.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: zoomed_windows.png")

def plot_error_analysis(preds, targets, metrics, output_dir):
    """
    图4: 误差分析图
    - 每种情绪的平均误差柱状图
    - 误差分布图
    """
    n_emotions = preds.shape[1]
    errors = np.abs(preds - targets)
    mean_errors = errors.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # 柱状图: 每种情绪的平均误差
    x = np.arange(n_emotions)
    colors = plt.cm.RdYlGn_r(mean_errors / mean_errors.max())
    
    bars = ax1.bar(x, mean_errors, color=colors, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m['emotion'] for m in metrics], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Absolute Error', fontsize=12)
    ax1.set_title('Per-Emotion Prediction Error', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for bar, err in zip(bars, mean_errors):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{err:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 直方图: 误差分布
    all_errors = errors.flatten()
    ax2.hist(all_errors, bins=50, color=COLOR_ERROR, edgecolor='black', alpha=0.7)
    ax2.axvline(all_errors.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error: {all_errors.mean():.3f}')
    ax2.set_xlabel('Absolute Error |pred - true|', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Prediction Errors', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: error_analysis.png")

def parse_params_file(model_path):
    """从 params.txt 解析训练参数"""
    import re
    params_path = os.path.join(os.path.dirname(model_path), 'params.txt')
    if not os.path.exists(params_path):
        return {}
    
    params = {}
    with open(params_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'(\w[\w_]*)\s*:\s*(.+)', line)
            if match:
                key, value = match.group(1), match.group(2).strip()
                params[key] = value
    return params

def main():
    parser = argparse.ArgumentParser(description='CADT 模型预测结果 - 时序折线图绘制')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录 (默认从 params.txt 读取)')
    parser.add_argument('--split_dir', type=str, default=None, help='划分索引目录 (默认从 params.txt 读取)')
    parser.add_argument('--split_mode', type=str, default=None, help='划分模式 (默认从 params.txt 读取)')
    parser.add_argument('--target_subject', type=int, default=None, help='目标被试编号 (可选，过滤该被试数据)')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--device', type=int, default=0, help='GPU 编号')
    parser.add_argument('--t_size', type=int, default=None,
                        help='embedding size (t_size)，默认从 checkpoint 自动检测；显式指定时优先')
    parser.add_argument('--encoder_type', type=str, default=None, choices=['cnn', 'mlp'],
                        help='编码器类型，默认从 checkpoint 自动检测')
    parser.add_argument('--x_dim', type=int, default=None, help='输入维度，默认从 checkpoint 自动检测')
    parser.add_argument('--seq_len', type=int, default=None, help='序列长度，默认从 checkpoint 自动检测')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录 (默认为模型同目录下的 timeseries_plots)')
    
    args = parser.parse_args()
    
    torch.cuda.set_device(args.device)
    
    # 设置默认值（可能会被 load_model 中的自动检测覆盖）
    args.d_AE = 128
    args.n_targets = 11
    
    # 从 params.txt 自动解析训练参数
    train_params = parse_params_file(args.model_path)
    split_mode = args.split_mode or train_params.get('split_mode', 'cross_subject')
    split_dir = args.split_dir or train_params.get('split_dir', '/vePFS-0x0d/DailyEEG/splits_new')
    data_dir = args.data_dir or train_params.get('data_dir', '/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new')
    norm_mode = train_params.get('norm_mode', 'sample')
    
    # 从 params.txt 读取 encoder_type 和相关参数（如果用户没有在命令行指定）
    if args.encoder_type is None and 'encoder_type' in train_params:
        args.encoder_type = train_params['encoder_type']
    if args.x_dim is None and 'x_dim' in train_params:
        args.x_dim = int(train_params['x_dim'])
    if args.seq_len is None and 'seq_len' in train_params:
        args.seq_len = int(train_params['seq_len'])
    
    print(f"训练参数: split_mode={split_mode}, split_dir={split_dir}")
    print(f"  data_dir={data_dir}, norm_mode={norm_mode}")
    
    # 加载测试集索引（与训练完全一致的方式）
    test_indices = load_emotion_split(split_dir, split_mode, "test")
    print(f"加载测试集索引: {len(test_indices)} 个样本")
    
    # 如果指定了 target_subject，按被试过滤
    if args.target_subject is not None:
        sub_path = os.path.join(data_dir, 'sub.npy')
        if os.path.exists(sub_path):
            all_sub = np.load(sub_path).astype(np.int64)
            test_indices = [i for i in test_indices if all_sub[i] == args.target_subject]
            print(f"  过滤被试 {args.target_subject} 后: {len(test_indices)} 个样本")
    
    args.n_sources = 1
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(args.model_path, args)
    
    # 加载测试集数据（与训练完全一致）
    print("\n加载测试集数据...")
    data_test = DailyEEG_SplitData(test_indices, split_mode=split_mode, 
                                    data_dir=data_dir, norm_mode=norm_mode)
    valid_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, 
                               drop_last=False, num_workers=0)
    print(f"测试集样本数: {len(data_test)}")
    
    # 获取预测结果
    print("\n正在进行预测...")
    preds, targets = get_predictions(model, valid_loader)
    
    # 计算指标
    print("\n正在计算指标...")
    metrics = compute_metrics(preds, targets)
    
    # 创建输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), 'timeseries_plots')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 图1: 整体时序图
    print("\n正在绘制整体时序图...")
    best_idx, worst_idx = plot_timeseries_overall(preds, targets, metrics, args.output_dir)
    
    # 图2: 分情绪子图
    print("正在绘制分情绪子图...")
    plot_timeseries_per_emotion(preds, targets, metrics, args.output_dir)
    
    # 图3: 局部放大图
    print("正在绘制局部放大图...")
    plot_zoomed_windows(preds, targets, metrics, args.output_dir, best_idx, worst_idx)
    
    # 图4: 误差分析
    print("正在绘制误差分析图...")
    plot_error_analysis(preds, targets, metrics, args.output_dir)
    
    # 打印指标汇总
    print("\n" + "="*80)
    print("各情绪预测指标:")
    print("="*80)
    print(f"{'情绪':<12} {'MSE':>10} {'R²':>10} {'Pearson':>10}")
    print("-"*80)
    for m in metrics:
        print(f"{m['emotion']:<12} {m['mse']:>10.4f} {m['r2']:>10.4f} {m['pearson']:>10.4f}")
    
    avg_mse = np.mean([m['mse'] for m in metrics])
    avg_r2 = np.mean([m['r2'] for m in metrics])
    avg_pearson = np.mean([m['pearson'] for m in metrics])
    print("-"*80)
    print(f"{'平均':<12} {avg_mse:>10.4f} {avg_r2:>10.4f} {avg_pearson:>10.4f}")
    
    print("\n所有图表绘制完成!")
    print(f"输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()