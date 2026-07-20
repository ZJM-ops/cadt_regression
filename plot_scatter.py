"""
真实值 vs 预测值散点图绘制脚本

功能：
1. 加载训练好的模型
2. 在目标域数据上进行预测
3. 绘制每种情绪的真实值 vs 预测值散点图
4. 计算并显示相关系数
"""

import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 情绪标签名称
EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]


def load_model(model_path, args):
    """加载训练好的模型"""
    from zjm.cadt.model.CADT import CATDModel
    
    model = CATDModel(args).cuda()
    checkpoint = torch.load(model_path)
    
    # 处理不同格式的 checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 过滤掉模型中不存在的 key（如 sphere_loss 参数）
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # 加载模型
    model.load_state_dict(filtered_state_dict, strict=False)
    
    model.eval()
    print(f"模型加载成功: {model_path}")
    print(f"  加载参数: {len(filtered_state_dict)}/{len(state_dict)}")
    return model


def get_predictions(model, data_loader):
    """获取真实值和预测值"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xs, ys, ds in data_loader:
            xs = torch.squeeze(xs.to(torch.float32), -1).cuda()
            ys = ys.cuda()
            
            zsc, _ = model.share_encoder(xs)
            pred_y = model.classifier(zsc)
            
            all_preds.append(pred_y.cpu().numpy())
            all_targets.append(ys.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\n预测结果:")
    print(f"  预测值形状: {all_preds.shape}")
    print(f"  真实值形状: {all_targets.shape}")
    print(f"  标签范围: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
    print(f"  预测范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    
    # 检查预测值的小数部分
    pred_fractional = all_preds - np.floor(all_preds)
    unique_fractions = np.unique(pred_fractional.round(4))
    print(f"  预测值小数部分唯一值数量: {len(unique_fractions)}")
    print(f"  前20个小数部分: {unique_fractions[:20]}")
    
    # 检查是否有非整数值
    has_non_integer = np.any(pred_fractional > 0.001)
    print(f"  是否包含非整数值: {has_non_integer}")
    
    # 打印一些样本值
    print(f"\n  部分预测值样本:")
    for i in range(min(5, all_preds.shape[0])):
        print(f"    样本{i}: {all_preds[i, :3].round(4)}")
    print(f"  部分真实值样本:")
    for i in range(min(5, all_targets.shape[0])):
        print(f"    样本{i}: {all_targets[i, :3].round(4)}")
    
    return all_preds, all_targets


def compute_correlations(preds, targets):
    """计算每种情绪的相关系数"""
    n_emotions = preds.shape[1]
    correlations = []
    
    for i in range(n_emotions):
        pred = preds[:, i].flatten()
        target = targets[:, i].flatten()
        
        # Pearson
        pearson = np.corrcoef(pred, target)[0, 1] if len(pred) >= 2 else 0
        pearson = 0.0 if np.isnan(pearson) else np.clip(pearson, -1.0, 1.0)
        
        # Spearman
        if len(pred) >= 2:
            pred_rank = np.argsort(np.argsort(pred))
            target_rank = np.argsort(np.argsort(target))
            spearman = np.corrcoef(pred_rank, target_rank)[0, 1]
            spearman = 0.0 if np.isnan(spearman) else np.clip(spearman, -1.0, 1.0)
        else:
            spearman = 0.0
        
        # R²
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        ss_res = np.sum((target - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # MSE
        mse = np.mean((pred - target) ** 2)
        
        correlations.append({
            'emotion': EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f'Emotion_{i}',
            'pearson': pearson,
            'spearman': spearman,
            'r2': r2,
            'mse': mse
        })
    
    return correlations


def plot_scatter(preds, targets, correlations, output_dir="scatter_plots"):
    """绘制散点图"""
    os.makedirs(output_dir, exist_ok=True)
    n_emotions = preds.shape[1]
    
    # 计算全局统计
    all_pred_flat = preds.flatten()
    all_target_flat = targets.flatten()
    global_pearson = np.corrcoef(all_pred_flat, all_target_flat)[0, 1]
    global_pearson = 0.0 if np.isnan(global_pearson) else np.clip(global_pearson, -1.0, 1.0)
    
    # 1. 所有情绪的汇总散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(all_target_flat, all_pred_flat, alpha=0.3, s=20, edgecolors='none')
    
    # 绘制对角线（理想预测）
    min_val = min(all_target_flat.min(), all_pred_flat.min())
    max_val = max(all_target_flat.max(), all_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Value', fontsize=14)
    plt.ylabel('Predicted Value', fontsize=14)
    plt.title(f'True vs Predicted (All Emotions)\nPearson = {global_pearson:.4f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_emotions_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.join(output_dir, 'all_emotions_scatter.png')}")
    
    # 2. 每种情绪的子图
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig)
    
    for i in range(n_emotions):
        ax = fig.add_subplot(gs[i])
        pred = preds[:, i].flatten()
        target = targets[:, i].flatten()
        corr = correlations[i]
        
        # 计算密度（用于颜色深浅）
        hist, xedges, yedges = np.histogram2d(target, pred, bins=20)
        z = np.zeros(len(target))
        for j in range(len(target)):
            xi = np.digitize(target[j], xedges) - 1
            yi = np.digitize(pred[j], yedges) - 1
            xi = max(0, min(xi, hist.shape[0]-1))
            yi = max(0, min(yi, hist.shape[1]-1))
            z[j] = hist[xi, yi]
        
        # 归一化颜色
        z = (z - z.min()) / (z.max() - z.min() + 1e-10)
        
        # 绘制散点图
        scatter = ax.scatter(target, pred, c=z, cmap='viridis', alpha=0.6, s=30, edgecolors='none')
        
        # 绘制对角线
        ax.plot([1, 5], [1, 5], 'r--', linewidth=1.5)
        
        # 设置标题和标签
        ax.set_title(f'{corr["emotion"]}\nPearson={corr["pearson"]:.2f}, R²={corr["r2"]:.2f}', fontsize=10)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.2)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加颜色条
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Density')
    
    plt.savefig(os.path.join(output_dir, 'per_emotion_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.join(output_dir, 'per_emotion_scatter.png')}")
    
    # 3. 统计信息表格
    print("\n" + "="*80)
    print("每种情绪的预测指标")
    print("="*80)
    print(f"{'情绪':<12} {'MSE':>10} {'R²':>10} {'Pearson':>10} {'Spearman':>10}")
    print("-" * 80)
    
    for corr in correlations:
        print(f"{corr['emotion']:<12} {corr['mse']:>10.4f} {corr['r2']:>10.4f} "
              f"{corr['pearson']:>10.4f} {corr['spearman']:>10.4f}")
    
    # 计算平均指标
    avg_mse = np.mean([c['mse'] for c in correlations])
    avg_r2 = np.mean([c['r2'] for c in correlations])
    avg_pearson = np.mean([c['pearson'] for c in correlations])
    avg_spearman = np.mean([c['spearman'] for c in correlations])
    
    print("\n" + "-"*80)
    print(f"{'平均':<12} {avg_mse:>10.4f} {avg_r2:>10.4f} {avg_pearson:>10.4f} {avg_spearman:>10.4f}")
    
    return correlations


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制真实值 vs 预测值散点图')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, default='/vePFS-0x0e/DailyEEG/processed_cadt', help='数据目录')
    parser.add_argument('--target_domain', type=int, default=28, help='目标域 ID')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    
    args = parser.parse_args()
    
    # 设置模型参数（与训练时一致）
    args.dataset = 'dailyeeg'
    args.task_type = 'regression'
    args.input_dim = 59
    args.timestep = 2000
    args.n_clusters = 6
    args.ball_loss_weight = 10.0
    args.lr = 0.001
    args.weight_decay = 0.0001
    args.tsize = 8
    args.y_dims = 11
    args.x_dim = 59
    args.d_AE = 128
    args.n_class = 11
    
    # 加载模型
    model = load_model(args.model_path, args)
    
    # 加载数据 - 使用正确的 DailyEEG_data
    from torch.utils.data import DataLoader
    from zjm.cadt.dataset.dataset import DailyEEG_data
    
    data_target = DailyEEG_data([args.target_domain], data_dir=args.data_dir)
    valid_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # 获取预测结果
    print("\n正在进行预测...")
    preds, targets = get_predictions(model, valid_loader)
    
    # 计算相关系数
    print("\n正在计算相关系数...")
    correlations = compute_correlations(preds, targets)
    
    # 绘制散点图
    print("\n正在绘制散点图...")
    plot_scatter(preds, targets, correlations)
    
    print("\n" + "="*80)
    print("散点图绘制完成!")
    print("="*80)


if __name__ == "__main__":
    main()