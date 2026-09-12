"""
CADT 模型训练结果散点图绘制脚本（仅 Emotion Split 模式）
加载模型权重，在验证集/测试集数据上进行预测，绘制真实值 vs 预测值散点图
支持 CNN 和 MLP 编码器（自动从 checkpoint 检测）
"""

import os
import argparse

_tmp_dir = "/root/zjm/cadt_test_code/tmp"
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir

import numpy as np
import torch
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.CADT import CATDModel
from dataset.dataset import DailyEEG_SplitData, load_emotion_split
from torch.utils.data import DataLoader

EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]


def load_model(model_path, args):
    """加载训练好的模型，并自动检测 n_sources / t_size / encoder_type"""
    checkpoint = torch.load(model_path, weights_only=True)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 检测 n_sources
    n_sources = 1
    for key in state_dict.keys():
        if 'discriminator.model.4.weight' in key:
            n_sources = state_dict[key].shape[0]
            break
    args.n_sources = n_sources
    print(f"  推断 n_sources = {n_sources}")

    # 检测 t_size （share_encoder.fc11.0.weight 或 MLP head_zsc 最后一层）
    t_size_detected = None
    for key in state_dict.keys():
        if key in ('share_encoder.fc11.0.weight', 'share_encoder.head_zsc.3.weight') or \
           key.endswith('.head_zsc.3.weight') or key.endswith('.fc11.0.weight'):
            t_size_detected = state_dict[key].shape[0]
            break
    if t_size_detected is None:
        # fallback: try first weight of zsc head
        for key in state_dict.keys():
            if 'share_encoder' in key and 'zsc' in key and 'weight' in key:
                t_size_detected = state_dict[key].shape[0]
                break
    if getattr(args, 't_size', None) is not None:
        if t_size_detected is not None and args.t_size != t_size_detected:
            print(f"  警告: 命令行 t_size={args.t_size} 与检测值 {t_size_detected} 不一致")
    elif t_size_detected is not None:
        args.t_size = t_size_detected
    else:
        args.t_size = 8
    print(f"  推断 t_size = {args.t_size}")

    # 检测 encoder_type：
    # - CNNEncoder 会有 share_encoder.feature_1x / share_encoder.conv1x / share_encoder.fc11 等层
    # - MLPEncoder 会有 share_encoder.trunk / share_encoder.head_zsc / share_encoder.head_zs
    has_cnn_keys = any(k.startswith('share_encoder.feature_') or
                       k.startswith('share_encoder.conv') or
                       k.startswith('share_encoder.fc11') or
                       k.startswith('share_encoder.fc12')
                       for k in state_dict.keys())
    has_mlp_keys = any(k.startswith('share_encoder.trunk') or
                       k.startswith('share_encoder.head_zsc')
                       for k in state_dict.keys())
    if has_mlp_keys and not has_cnn_keys:
        args.encoder_type = 'mlp'
    elif has_cnn_keys and not has_mlp_keys:
        args.encoder_type = 'cnn'
    else:
        args.encoder_type = getattr(args, 'encoder_type', 'cnn')
    print(f"  推断 encoder_type = {args.encoder_type}")

    # 检测输入维度 x_dim 和 seq_len
    # CNN: share_encoder.conv1 (or fc11.0.weight's fan-in = f11_input_size depends on architecture)
    # MLP: share_encoder.trunk.0.weight shape = [hidden, in_dim], in_dim = x_dim * seq_len
    if args.encoder_type == 'mlp':
        trunk_key = 'share_encoder.trunk.0.weight'
        if trunk_key in state_dict:
            in_dim = state_dict[trunk_key].shape[1]
            # 默认 DE 特征是 (295,) -> x_dim=295, seq_len=1
            args.x_dim = getattr(args, 'x_dim', 295)
            args.seq_len = getattr(args, 'seq_len', 1)
            if args.x_dim * args.seq_len != in_dim:
                # 尝试用 in_dim 反推
                print(f"  警告: x_dim*seq_len != {in_dim}, 重新设置 x_dim={in_dim}, seq_len=1")
                args.x_dim = in_dim
                args.seq_len = 1
    else:  # cnn
        # 默认值 59 通道 2000 长度
        args.x_dim = getattr(args, 'x_dim', 59)
        args.seq_len = getattr(args, 'seq_len', 2000)

    args.d_AE = getattr(args, 'd_AE', 128)
    args.n_targets = getattr(args, 'n_targets', 11)

    print(f"  输入假设: x_dim={args.x_dim}, seq_len={args.seq_len}")

    model = CATDModel(args).cuda()
    model_state_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state_dict}
    missing = [k for k in model_state_dict.keys() if k not in state_dict]
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"模型加载成功: {model_path}")
    print(f"  参数匹配: {len(filtered)}/{len(state_dict)}, 缺失 {len(missing)} 个")
    return model


def get_predictions(model, data_loader):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xs, ys, _ in data_loader:
            xs = xs.to(torch.float32).cuda()
            ys = ys.to(torch.float32).cuda()
            pred_y = model.predict_target(xs)
            all_preds.append(pred_y.cpu().numpy())
            all_targets.append(ys.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    print(f"\n预测结果: pred={all_preds.shape}, true={all_targets.shape}")
    print(f"  标签范围: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
    print(f"  预测范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    return all_preds, all_targets


def compute_correlations(preds, targets):
    from scipy.stats import pearsonr
    n_emotions = preds.shape[1]
    correlations = []
    for i in range(n_emotions):
        pred = preds[:, i].flatten()
        target = targets[:, i].flatten()
        corr, _ = pearsonr(pred, target)
        pearson = 0.0 if np.isnan(corr) else np.clip(corr, -1.0, 1.0)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        ss_res = np.sum((target - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        mse = np.mean((pred - target) ** 2)
        correlations.append({
            'emotion': EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f'Emotion_{i}',
            'pearson': pearson, 'r2': r2, 'mse': mse
        })
    return correlations


def plot_scatter(preds, targets, correlations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n_emotions = preds.shape[1]
    all_pred_flat = preds.flatten()
    all_target_flat = targets.flatten()

    plt.figure(figsize=(10, 10))
    plt.scatter(all_target_flat, all_pred_flat, alpha=0.3, s=20, edgecolors='none', color='#1f77b4')
    min_val = min(all_target_flat.min(), all_pred_flat.min())
    max_val = max(all_target_flat.max(), all_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Value', fontsize=14)
    plt.ylabel('Predicted Value', fontsize=14)
    plt.title(f'True vs Predicted (All Emotions)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_emotions_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.join(output_dir, 'all_emotions_scatter.png')}")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig)
    scatter = None
    for i in range(n_emotions):
        ax = fig.add_subplot(gs[i])
        pred = preds[:, i].flatten()
        target = targets[:, i].flatten()
        corr = correlations[i]
        hist, xedges, yedges = np.histogram2d(target, pred, bins=20)
        z = np.zeros(len(target))
        for j in range(len(target)):
            xi = max(0, min(np.digitize(target[j], xedges) - 1, hist.shape[0]-1))
            yi = max(0, min(np.digitize(pred[j], yedges) - 1, hist.shape[1]-1))
            z[j] = hist[xi, yi]
        z = (z - z.min()) / (z.max() - z.min() + 1e-10)
        scatter = ax.scatter(target, pred, c=z, cmap='viridis', alpha=0.6, s=30, edgecolors='none')
        ax.plot([0.5, 5.5], [0.5, 5.5], 'r--', linewidth=1.5)
        ax.set_title(f'{corr["emotion"]}\nPearson={corr["pearson"]:.2f}, R²={corr["r2"]:.2f}', fontsize=10)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    if scatter is not None:
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Density')
    plt.savefig(os.path.join(output_dir, 'per_emotion_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.join(output_dir, 'per_emotion_scatter.png')}")

    print("\n" + "="*80)
    print(f"{'情绪':<12} {'MSE':>10} {'R²':>10} {'Pearson':>10}")
    print("-" * 80)
    for corr in correlations:
        print(f"{corr['emotion']:<12} {corr['mse']:>10.4f} {corr['r2']:>10.4f} {corr['pearson']:>10.4f}")
    avg_mse = np.mean([c['mse'] for c in correlations])
    avg_r2 = np.mean([c['r2'] for c in correlations])
    avg_pearson = np.mean([c['pearson'] for c in correlations])
    print("\n" + "-"*80)
    print(f"{'平均':<12} {avg_mse:>10.4f} {avg_r2:>10.4f} {avg_pearson:>10.4f}")
    return correlations


def main():
    parser = argparse.ArgumentParser(description='CADT 模型散点图绘制（Emotion Split）')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径 model_best.pth')
    parser.add_argument('--data_dir', type=str, default='/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new',
                        help='NPY 数据目录（波形或 DE 特征）')
    parser.add_argument('--split_dir', type=str, default='/vePFS-0x0d/DailyEEG/splits_new',
                        help='Emotion split 目录')
    parser.add_argument('--split_mode', type=str, default='cross_subject',
                        choices=['cross_subject', 'cross_day', 'within_subject_day'])
    parser.add_argument('--split_type', type=str, default='test', choices=['val', 'test'],
                        help='在验证集还是测试集上画图')
    parser.add_argument('--target_subject', type=int, default=None,
                        help='在 cross_day/within_subject_day 模式下按被试过滤')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--norm_mode', type=str, default='sample',
                        choices=['sample', 'day', 'subject'])
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认与模型同目录下 scatter_plots/')
    parser.add_argument('--x_dim', type=int, default=None, help='可选：覆盖自动检测')
    parser.add_argument('--seq_len', type=int, default=None, help='可选：覆盖自动检测')
    parser.add_argument('--t_size', type=int, default=None, help='可选：覆盖自动检测')
    parser.add_argument('--encoder_type', type=str, default=None, choices=['cnn', 'mlp'],
                        help='可选：覆盖自动检测')
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    print("\n加载模型...")
    model = load_model(args.model_path, args)

    print(f"\n加载 {args.split_type} 集 (mode={args.split_mode})...")
    indices = load_emotion_split(args.split_dir, args.split_mode, args.split_type)
    if args.target_subject is not None:
        sub_path = os.path.join(args.data_dir, "sub.npy")
        if os.path.exists(sub_path):
            all_sub = np.load(sub_path)
            indices = [idx for idx in indices if all_sub[idx] == args.target_subject]
            print(f"  按被试 {args.target_subject} 过滤后: {len(indices)} 个样本")
        else:
            print("  警告: sub.npy 不存在，无法按被试过滤")

    dataset = DailyEEG_SplitData(indices, split_mode=args.split_mode,
                                 data_dir=args.data_dir, norm_mode=args.norm_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=2, pin_memory=True)
    print(f"  样本数: {len(dataset)}")

    print("\n正在预测...")
    preds, targets = get_predictions(model, loader)
    correlations = compute_correlations(preds, targets)

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.model_path),
                                                 f"scatter_plots_{args.split_type}")
    print(f"\n正在绘图 -> {output_dir}")
    plot_scatter(preds, targets, correlations, output_dir=output_dir)
    print("\n散点图绘制完成!")


if __name__ == "__main__":
    main()
