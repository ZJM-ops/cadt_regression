"""
CADT 模型训练结果散点图绘制脚本
加载模型权重，在目标域数据上进行预测，绘制真实值 vs 预测值散点图
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

# 将当前目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.CADT import CATDModel
from dataset.dataset import DailyEEG_data, get_available_domains
from torch.utils.data import DataLoader

# 情绪标签名称
EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]


def load_model(model_path, args):
    """加载训练好的模型"""
    # 先加载 checkpoint 来推断 n_sources
    checkpoint = torch.load(model_path, weights_only=True)
    
    # 处理不同格式的 checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 从 discriminator 的最后一层推断 n_sources
    # discriminator.model.4.weight 的形状是 [n_sources, hidden]
    n_sources = 1
    for key in state_dict.keys():
        if 'discriminator.model.4.weight' in key:
            n_sources = state_dict[key].shape[0]
            break
    
    print(f"从模型推断 n_sources = {n_sources}")
    args.n_sources = n_sources
    
    # 创建正确结构的模型
    model = CATDModel(args).cuda()
    
    # 过滤掉模型中不存在的 key
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
            xs = xs.to(torch.float32).cuda()
            ys = ys.to(torch.float32).cuda()
            
            # 使用 predict_target 方法进行预测（与训练时一致）
            pred_y = model.predict_target(xs)
            
            all_preds.append(pred_y.cpu().numpy())
            all_targets.append(ys.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\n预测结果:")
    print(f"  预测值形状: {all_preds.shape}")
    print(f"  真实值形状: {all_targets.shape}")
    print(f"  标签范围: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
    print(f"  预测范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    
    return all_preds, all_targets


def compute_correlations(preds, targets):
    """计算每种情绪的相关系数"""
    from scipy.stats import pearsonr
    
    n_emotions = preds.shape[1]
    correlations = []
    
    for i in range(n_emotions):
        pred = preds[:, i].flatten()
        target = targets[:, i].flatten()
        
        # Pearson
        corr, _ = pearsonr(pred, target)
        pearson = 0.0 if np.isnan(corr) else np.clip(corr, -1.0, 1.0)
        
        # R²
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        ss_res = np.sum((target - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # MSE
        mse = np.mean((pred - target) ** 2)
        
        correlations.append({
            'emotion': EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f'Emotion_{i}',
            'pearson': pearson,
            'r2': r2,
            'mse': mse
        })
    
    return correlations


def plot_scatter(preds, targets, correlations, output_dir="results4/scatter_plots"):
    """绘制散点图"""
    os.makedirs(output_dir, exist_ok=True)
    n_emotions = preds.shape[1]
    
    # 计算全局统计
    all_pred_flat = preds.flatten()
    all_target_flat = targets.flatten()
    
    # 1. 所有情绪的汇总散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(all_target_flat, all_pred_flat, alpha=0.3, s=20, edgecolors='none', color='#1f77b4')
    
    # 绘制对角线（理想预测）
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
        ax.plot([0.5, 5.5], [0.5, 5.5], 'r--', linewidth=1.5)
        
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
    print(f"{'情绪':<12} {'MSE':>10} {'R²':>10} {'Pearson':>10}")
    print("-" * 80)
    
    for corr in correlations:
        print(f"{corr['emotion']:<12} {corr['mse']:>10.4f} {corr['r2']:>10.4f} {corr['pearson']:>10.4f}")
    
    # 计算平均指标
    avg_mse = np.mean([c['mse'] for c in correlations])
    avg_r2 = np.mean([c['r2'] for c in correlations])
    avg_pearson = np.mean([c['pearson'] for c in correlations])
    
    print("\n" + "-"*80)
    print(f"{'平均':<12} {avg_mse:>10.4f} {avg_r2:>10.4f} {avg_pearson:>10.4f}")
    
    return correlations


def load_domain_metadata(data_dir):
    """Load meta.json to get domain_offset_map and domains_per_subject_map."""
    import json
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return {
            'domain_offset_map': {int(k): v for k, v in meta.get('domain_offset_map', {}).items()},
            'domains_per_subject_map': {int(k): v for k, v in meta.get('domains_per_subject_map', {}).items()}
        }
    return {'domain_offset_map': {}, 'domains_per_subject_map': {}}


def get_domain_ids_by_subject(sub_id, domain_offset_map, domains_per_subject_map):
    """Get all domain IDs belonging to a subject."""
    if sub_id not in domain_offset_map:
        return []
    offset = domain_offset_map[sub_id]
    n_domains = domains_per_subject_map.get(sub_id, 10)
    return list(range(offset, offset + n_domains))


def main():
    parser = argparse.ArgumentParser(description='CADT 模型结果散点图绘制')
    parser.add_argument('--model_path', type=str, default='/root/zjm/cadt/results4/model_best_target0.pth', 
                        help='模型路径')
    parser.add_argument('--data_dir', type=str, default='/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new', 
                        help='数据目录')
    parser.add_argument('--target_subject', type=int, default=None, 
                        help='目标被试编号（用于确定目标域）')
    parser.add_argument('--target_domain_ids', type=str, default=None,
                        help='直接指定目标域ID (例如 "1,2,3")')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--device', type=int, default=0, help='GPU 编号')
    parser.add_argument('--output_dir', type=str, default='results4/scatter_plots', 
                        help='散点图输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    torch.cuda.set_device(args.device)
    
    # 设置模型参数（与训练时一致）
    args.x_dim = 59
    args.seq_len = 2000
    args.d_AE = 128
    args.n_targets = 11
    
    # 确定目标域 IDs
    target_domain_ids = []
    
    if args.target_domain_ids:
        # 直接从参数解析 ID 列表
        target_domain_ids = [int(x.strip()) for x in args.target_domain_ids.split(',')]
        print(f"直接指定目标域 IDs: {target_domain_ids}")
    elif args.target_subject is not None:
        # 通过 target_subject 查找
        print("加载数据元信息...")
        available_domains = get_available_domains(args.data_dir)
        meta = load_domain_metadata(args.data_dir)
        target_domain_ids = get_domain_ids_by_subject(args.target_subject, 
                                                       meta['domain_offset_map'], 
                                                       meta['domains_per_subject_map'])
        
        # 过滤有效域
        target_domain_ids = [d for d in target_domain_ids if d in available_domains]
        print(f"目标被试 {args.target_subject} 的有效域 ID: {target_domain_ids}")
    else:
        print("错误: 必须指定 --target_subject 或 --target_domain_ids")
        sys.exit(1)
    
    # 设置 n_sources（模型初始化需要，但预测时不使用）
    args.n_sources = 1  # 预测时不需要实际的源域数量
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(args.model_path, args)
    
    # 加载目标域数据
    print("\n加载目标域数据...")
    data_target = DailyEEG_data(target_domain_ids, data_dir=args.data_dir)
    valid_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"  目标域样本数: {len(data_target)}")
    
    # 获取预测结果
    print("\n正在进行预测...")
    preds, targets = get_predictions(model, valid_loader)
    
    # 计算相关系数
    print("\n正在计算相关系数...")
    correlations = compute_correlations(preds, targets)
    
    # 绘制散点图
    print("\n正在绘制散点图...")
    plot_scatter(preds, targets, correlations, output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print("散点图绘制完成!")
    print("="*80)


if __name__ == "__main__":
    main()