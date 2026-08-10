"""
脚本功能：为已有的实验结果添加每个情绪维度的 R2 指标和平均 R2

使用方式：
    python compute_per_dim_r2.py --result_dir /root/zjm/cadt/results4/20260727_013029_emotion_cross_day
    
参数说明：
    --result_dir: 实验结果目录，包含 model_best.pth, params.txt, results.txt
"""

import argparse
import sys
import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 添加 cadt 项目目录到 Python 路径
sys.path.insert(0, '/root/zjm/cadt')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from dataset.dataset import DailyEEG_SplitData, load_emotion_split
from model.CADT import CATDModel


EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]


def load_params(params_path):
    """从 params.txt 中解析训练参数"""
    params = {}
    current_section = None
    
    with open(params_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('==='):
                continue
            if line.startswith('---'):
                current_section = line.strip('- ')
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 尝试转换类型
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
    
    return params


def evaluate_per_dim(model, loader, device):
    """评估并返回每个维度的指标"""
    model.eval()
    all_preds = []
    all_ys = []
    
    with torch.no_grad():
        for xt, yt, _ in loader:
            xt = xt.to(torch.float32).cuda(device)
            yt = yt.to(torch.float32).cuda(device)
            pred = model.predict_target(xt)
            all_preds.append(pred.detach().cpu().numpy())
            all_ys.append(yt.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_ys = np.concatenate(all_ys, axis=0)
    
    # 计算每个维度的指标
    per_dim_metrics = []
    n_dim = all_ys.shape[1]
    
    # 打印诊断信息
    print(f"\n  [诊断] 数据集大小: {len(all_ys)} 样本, {n_dim} 维度")
    print(f"  [诊断] 全局均值: {np.mean(all_ys):.6f}")
    print(f"  [诊断] 全局 SS_tot: {np.sum((all_ys - np.mean(all_ys)) ** 2):.6f}")
    print(f"  [诊断] 全局 SS_res: {np.sum((all_ys - all_preds) ** 2):.6f}")
    print()
    
    for dim in range(n_dim):
        y_true = all_ys[:, dim]
        y_pred = all_preds[:, dim]
        
        # R2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # 诊断信息
        label = EMOTION_LABELS[dim] if dim < len(EMOTION_LABELS) else f'Dim_{dim}'
        print(f"  [诊断] {label}: 均值={np.mean(y_true):.4f}, 标准差={np.std(y_true):.4f}, "
              f"SS_tot={ss_tot:.4f}, SS_res={ss_res:.4f}, R²={r2:.4f}")
        
        # Pearson
        p_corr, _ = pearsonr(y_true, y_pred)
        pearson = p_corr if not np.isnan(p_corr) else 0.0
        
        # Spearman
        s_corr, _ = spearmanr(y_true, y_pred)
        spearman = s_corr if not np.isnan(s_corr) else 0.0
        
        # MSE, MAE
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        per_dim_metrics.append({
            'dim': dim,
            'r2': r2,
            'pearson': pearson,
            'spearman': spearman,
            'mse': mse,
            'mae': mae,
            'ss_tot': ss_tot,
            'ss_res': ss_res,
            'mean_true': np.mean(y_true),
            'std_true': np.std(y_true)
        })
    
    # 计算整体指标（展平方式，保持与原代码一致）
    ss_res_all = np.sum((all_ys - all_preds) ** 2)
    ss_tot_all = np.sum((all_ys - np.mean(all_ys)) ** 2)
    r2_overall = 1 - (ss_res_all / ss_tot_all) if ss_tot_all != 0 else 0.0
    
    avg_r2 = np.mean([m['r2'] for m in per_dim_metrics])
    avg_pearson = np.mean([m['pearson'] for m in per_dim_metrics])
    avg_spearman = np.mean([m['spearman'] for m in per_dim_metrics])
    
    return {
        'r2': r2_overall,
        'avg_r2': avg_r2,
        'avg_pearson': avg_pearson,
        'avg_spearman': avg_spearman,
        'per_dim_metrics': per_dim_metrics
    }


def update_results_txt(results_path, val_metrics, test_metrics):
    """更新 results.txt 文件，添加每个维度的 R2"""
    # 读取原文件内容
    with open(results_path, 'r') as f:
        lines = f.readlines()
    
    # 找到需要修改的位置
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 修改验证集结果部分
        if line.strip() == '--- Validation Set Results (Best Model) ---':
            new_lines.append(line)
            i += 1
            # 跳过原有的验证集指标行，直到遇到空行或下一个 section
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('---'):
                # 如果是 Val R2 行，添加平均 R2（如果尚未存在）
                if lines[i].startswith('Val R2:'):
                    new_lines.append(lines[i])
                    # 检查下一行是否已经是 Val Avg R2
                    if i + 1 < len(lines) and 'Val Avg R2' not in lines[i + 1]:
                        new_lines.append(f"Val Avg R2 (per-dim): {val_metrics['avg_r2']:.6f}\n")
                elif 'Val Avg R2' not in lines[i]:  # 跳过已有的 Avg R2 行
                    new_lines.append(lines[i])
                i += 1
            continue
        
        # 修改测试集结果部分
        if line.strip() == '--- Test Set Results (Best Model) ---':
            new_lines.append(line)
            i += 1
            # 跳过原有的测试集指标行
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('---'):
                if lines[i].startswith('Test R2:'):
                    new_lines.append(lines[i])
                    # 检查下一行是否已经是 Test Avg R2
                    if i + 1 < len(lines) and 'Test Avg R2' not in lines[i + 1]:
                        new_lines.append(f"Test Avg R2 (per-dim): {test_metrics['avg_r2']:.6f}\n")
                elif 'Test Avg R2' not in lines[i]:  # 跳过已有的 Avg R2 行
                    new_lines.append(lines[i])
                i += 1
            continue
        
        # 修改 Per-Emotion Metrics 部分
        if line.strip() == '--- Per-Emotion Metrics ---':
            new_lines.append(line)
            i += 1
            
            # 添加新的表头
            new_lines.append(f"{'Emotion':<12} {'MSE':>10} {'MAE':>10} {'R2':>10} {'Pearson':>10} {'Spearman':>10}\n")
            new_lines.append("-" * 70 + "\n")
            
            # 添加每个维度的指标（使用 test_metrics 的 per_dim_metrics）
            for metric in test_metrics['per_dim_metrics']:
                label = EMOTION_LABELS[metric['dim']] if metric['dim'] < len(EMOTION_LABELS) else f'Emotion_{metric["dim"]}'
                new_lines.append(f"{label:<12} {metric['mse']:>10.4f} {metric['mae']:>10.4f} "
                                f"{metric['r2']:>10.4f} {metric['pearson']:>10.4f} {metric['spearman']:>10.4f}\n")
            
            # 添加平均值行
            avg_mse = np.mean([m['mse'] for m in test_metrics['per_dim_metrics']])
            avg_mae = np.mean([m['mae'] for m in test_metrics['per_dim_metrics']])
            avg_r2_all = np.mean([m['r2'] for m in test_metrics['per_dim_metrics']])
            avg_p = np.mean([m['pearson'] for m in test_metrics['per_dim_metrics']])
            avg_s = np.mean([m['spearman'] for m in test_metrics['per_dim_metrics']])
            
            new_lines.append("-" * 70 + "\n")
            new_lines.append(f"{'Average':<12} {avg_mse:>10.4f} {avg_mae:>10.4f} "
                            f"{avg_r2_all:>10.4f} {avg_p:>10.4f} {avg_s:>10.4f}\n")
            
            # 跳过原有的 Per-Emotion Metrics 部分
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('---'):
                i += 1
            while i < len(lines) and lines[i].strip() != 'Model Saved:':
                i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    # 写回文件
    with open(results_path, 'w') as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser("为已有实验结果添加每个维度的 R2")
    parser.add_argument("--result_dir", required=True, help="实验结果目录")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备号")
    args = parser.parse_args()
    
    result_dir = args.result_dir
    device = args.device
    
    # 检查目录结构
    params_path = os.path.join(result_dir, "params.txt")
    model_path = os.path.join(result_dir, "model_best.pth")
    results_path = os.path.join(result_dir, "results.txt")
    
    if not os.path.exists(params_path):
        print(f"错误：找不到 params.txt 在 {params_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"错误：找不到 model_best.pth 在 {model_path}")
        sys.exit(1)
    if not os.path.exists(results_path):
        print(f"错误：找不到 results.txt 在 {results_path}")
        sys.exit(1)
    
    print(f"正在处理结果目录: {result_dir}")
    
    # 加载参数
    params = load_params(params_path)
    print(f"加载的参数: {params}")
    
    # 创建 args 对象
    class Args:
        pass
    
    args_obj = Args()
    args_obj.x_dim = params.get('x_dim', 59)
    args_obj.seq_len = params.get('seq_len', 2000)
    args_obj.d_AE = params.get('d_AE', 128)
    args_obj.n_targets = params.get('n_targets', 11)
    args_obj.contrast_tau = params.get('contrast_tau', 1.0)
    
    # 先加载 checkpoint，从 state_dict 中推断 n_sources
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    # n_sources = discriminator.model.4.weight 的输出维度
    n_sources = checkpoint['discriminator.model.4.weight'].shape[0]
    args_obj.n_sources = n_sources
    print(f"从 checkpoint 推断 n_sources = {n_sources}")
    
    # 构建模型并加载权重
    model = CATDModel(args_obj)
    model.load_state_dict(checkpoint)
    model = model.cuda(device)
    model.eval()
    print("模型加载成功")
    
    # 加载数据
    data_dir = params.get('data_dir', '/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new')
    split_dir = params.get('split_dir', '/vePFS-0x0d/DailyEEG/splits_new')
    split_mode = params.get('split_mode', 'cross_subject')
    
    print(f"数据目录: {data_dir}")
    print(f"分割模式: {split_mode}")
    
    # 使用 emotion split 加载数据
    train_idx = load_emotion_split(split_dir, split_mode, "pretrain")
    train_idx += load_emotion_split(split_dir, split_mode, "finetune")
    val_idx = load_emotion_split(split_dir, split_mode, "val")
    test_idx = load_emotion_split(split_dir, split_mode, "test")
    
    # 创建数据集和 DataLoader
    val_dataset = DailyEEG_SplitData(
        data_dir=data_dir,
        sample_indices=val_idx,
        split_mode=split_mode
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    test_dataset = DailyEEG_SplitData(
        data_dir=data_dir,
        sample_indices=test_idx,
        split_mode=split_mode
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 评估验证集
    print("\n正在评估验证集...")
    val_metrics = evaluate_per_dim(model, val_loader, device)
    
    # 评估测试集
    print("\n正在评估测试集...")
    test_metrics = evaluate_per_dim(model, test_loader, device)
    
    # 打印结果
    print("\n=== 验证集指标 ===")
    print(f"整体 R2: {val_metrics['r2']:.6f}")
    print(f"平均 R2 (按维度): {val_metrics['avg_r2']:.6f}")
    
    print("\n=== 测试集指标 ===")
    print(f"整体 R2: {test_metrics['r2']:.6f}")
    print(f"平均 R2 (按维度): {test_metrics['avg_r2']:.6f}")
    
    print("\n=== 每个情绪维度的 R2 ===")
    for metric in test_metrics['per_dim_metrics']:
        label = EMOTION_LABELS[metric['dim']] if metric['dim'] < len(EMOTION_LABELS) else f'Emotion_{metric["dim"]}'
        print(f"{label}: R2={metric['r2']:.4f}, Pearson={metric['pearson']:.4f}, Spearman={metric['spearman']:.4f}")
    
    # 更新 results.txt
    print("\n正在更新 results.txt...")
    update_results_txt(results_path, val_metrics, test_metrics)
    print("更新完成!")


if __name__ == "__main__":
    main()