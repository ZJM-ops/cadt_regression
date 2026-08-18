import argparse
import sys
import os

_tmp_dir = "/root/zjm/cadt_test_code/tmp"
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir

import numpy as np
from scipy.stats import pearsonr, spearmanr

# 将当前目录添加到 Python 路径，确保能导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from dataset.dataset import DailyEEG_SplitData, MultiSourceBatchSampler, load_emotion_split
from model.CADT import CATDModel

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--x_dim", type=int, default=59, help="EEG channel count (CNN) or DE feature dim (MLP)")
parser.add_argument("--seq_len", type=int, default=2000, help="10s @ 200Hz for waveform CNN; 1 for DE-MLP")
parser.add_argument("--d_AE", type=int, default=128, help="AE hidden dimension")
parser.add_argument("--n_targets", type=int, default=11, help="regression output dim (11 emotion labels)")
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256,
                     help="must satisfy batch_size // n_sources >= 1 for MultiSourceBatchSampler")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers count")
parser.add_argument("--sphere_loss_weight", type=float, default=10.0)
parser.add_argument("--discriminator_loss_weight", type=float, default=1.0)
parser.add_argument("--domain_cls_weight", type=float, default=1.0)
parser.add_argument("--contrast_tau", type=float, default=1.0)
parser.add_argument("--result_base", type=str, default=None,
                    help="Override result directory name (e.g., results_mlp_search). If None, auto-determined.")
parser.add_argument("--t_size", type=int, default=128,
                     help="embedding size (t_size) in encoder; results saved under result_base/tsize{xx}/")
parser.add_argument("--data_dir", type=str, default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
parser.add_argument("--split_dir", type=str, default="/vePFS-0x0d/DailyEEG/splits_new",
                     help="Path to emotion project's splits directory")
parser.add_argument("--split_mode", type=str, default="cross_subject",
                     choices=["cross_subject", "cross_day", "within_subject_day"],
                     help="Emotion split mode: cross_subject, cross_day, or within_subject_day")
parser.add_argument("--target_subject", type=int, default=None,
                     help="Used in cross_day/within_subject_day mode to filter data by subject. "
                          "Must NOT be set in cross_subject mode (split handles it).")
parser.add_argument("--norm_mode", type=str, default="sample",
                     choices=["sample", "day", "subject"],
                     help="Normalization mode: sample (per-sample), day (per-day), subject (per-subject). "
                          "For DE features, prefer 'sample' (already day-normed in extraction).")
parser.add_argument("--ablation", type=str, default="none",
                     choices=["none", "variant"],
                     help="Ablation mode: 'none' for full model; 'variant' keeps only regression + "
                          "contrastive loss, removing all domain-related losses.")
parser.add_argument("--encoder_type", type=str, default="cnn",
                     choices=["cnn", "mlp"],
                     help="Encoder type: 'cnn' uses CNNEncoder (raw EEG waveform), "
                          "'mlp' uses MLPEncoder (flattened DE vectors). "
                          "With 'mlp' set --x_dim 295 --seq_len 1 and --data_dir pointing to DE npy dir.")
args = parser.parse_args()


def load_domain_metadata(data_dir):
    """Load meta.json for downstream scripts. Training pipeline uses emotion split only."""
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


if __name__ == '__main__':
    print(f"\n=== GPU 状态检测 ===")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前设备索引: {args.device}")
        print(f"设备名称: {torch.cuda.get_device_name(args.device)}")
        print(f"设备内存: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: CUDA 不可用！将使用 CPU 运行，速度会很慢！")

    torch.cuda.set_device(args.device)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 结果文件夹路径
    if args.result_base is not None:
        result_base = args.result_base
    else:
        if args.norm_mode == "day":
            result_base = "result_day_td"
        elif args.norm_mode == "subject":
            result_base = "result_sub_std"
        else:
            result_base = "results4"
        if args.ablation != "none":
            result_base = result_base + "_ablation"
        result_base = os.path.join(result_base, f"tsize{args.t_size}")
        if args.encoder_type == "mlp":
            #result_base = result_base + "_enc_mlp"results_
            result_base = "results_mlp_ablation" if args.ablation != "none" else "results_mlp"
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_base,
                              f"{timestamp}_emotion_{args.split_mode}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果将保存到: {result_dir}")
    print(f"标准化模式: {args.norm_mode}")
    print(f"Encoder: {args.encoder_type}")
    print(f"输入形状假设: (N, seq_len={args.seq_len}, x_dim={args.x_dim})")
    if args.ablation != "none":
        print(f"消融模式: {args.ablation}")

    # ========== 加载 Emotion Split 划分 ==========
    print(f"\n=== 使用 Emotion 项目划分 ===")
    print(f"  Split mode: {args.split_mode}")
    print(f"  Split dir:  {args.split_dir}")

    train_indices = load_emotion_split(args.split_dir, args.split_mode, "pretrain")
    train_indices += load_emotion_split(args.split_dir, args.split_mode, "finetune")
    val_indices = load_emotion_split(args.split_dir, args.split_mode, "val")
    test_indices = load_emotion_split(args.split_dir, args.split_mode, "test")

    if args.target_subject is not None and args.split_mode != "cross_subject":
        print(f"\n  === 按被试 {args.target_subject} 过滤数据 ===")
        all_sub = np.load(os.path.join(args.data_dir, "sub.npy"))
        train_indices = [idx for idx, mask in zip(train_indices, all_sub[train_indices] == args.target_subject) if mask]
        val_indices = [idx for idx, mask in zip(val_indices, all_sub[val_indices] == args.target_subject) if mask]
        test_indices = [idx for idx, mask in zip(test_indices, all_sub[test_indices] == args.target_subject) if mask]
        print(f"  过滤后 - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    print(f"  Train samples: {len(train_indices)} (pretrain + finetune)")
    print(f"  Val samples:   {len(val_indices)}")
    print(f"  Test samples:  {len(test_indices)}")

    data_train = DailyEEG_SplitData(train_indices, split_mode=args.split_mode, data_dir=args.data_dir,
                                     norm_mode=args.norm_mode)
    args.n_sources = len(np.unique(data_train.d))
    unique_domains = sorted(np.unique(data_train.d).tolist())
    id_map = {raw: i for i, raw in enumerate(unique_domains)}
    train_domain_counts = {d: int((data_train.d == d).sum()) for d in unique_domains}
    print(f"  Training domains: {args.n_sources} unique domains")
    print(f"  训练集域 (原始ID): {unique_domains}")
    print(f"  训练集各域样本数: {train_domain_counts}")

    data_train.d = np.array([id_map[int(d)] for d in data_train.d], dtype=np.int64)
    sampler = MultiSourceBatchSampler(data_train.d, list(id_map.values()), batch_size=args.batch_size)
    train_loader = DataLoader(data_train, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    data_val = DailyEEG_SplitData(val_indices, split_mode=args.split_mode, data_dir=args.data_dir,
                                   norm_mode=args.norm_mode)
    valid_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    target_domain_ids = sorted(np.unique(data_val.d).tolist())
    val_domain_counts = {d: int((data_val.d == d).sum()) for d in target_domain_ids}
    print(f"  验证集域 (原始ID): {target_domain_ids}")
    print(f"  验证集各域样本数: {val_domain_counts}")

    data_test = DailyEEG_SplitData(test_indices, split_mode=args.split_mode, data_dir=args.data_dir,
                                    norm_mode=args.norm_mode)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    test_domain_ids = sorted(np.unique(data_test.d).tolist())
    test_domain_counts = {d: int((data_test.d == d).sum()) for d in test_domain_ids}
    print(f"  测试集域 (原始ID): {test_domain_ids}")
    print(f"  测试集各域样本数: {test_domain_counts}")

    if args.split_mode == "cross_subject":
        print(f"\n  === 被试信息 (cross_subject) ===")
        print(f"  训练被试: {unique_domains} (共 {len(unique_domains)} 个)")
        print(f"  验证被试: {target_domain_ids} (共 {len(target_domain_ids)} 个)")
        print(f"  测试被试: {test_domain_ids} (共 {len(test_domain_ids)} 个)")

    # ========== 模型构建 + 形状 sanity check ==========
    model = CATDModel(args).cuda()

    x_dummy = torch.randn(4, args.seq_len, args.x_dim).cuda()
    zsc_dummy, zs_dummy = model.share_encoder(x_dummy)
    print(f"Sanity check -- zsc: {tuple(zsc_dummy.shape)}, zs: {tuple(zs_dummy.shape)} (expect (4, {args.t_size}) each)")

    # ========== 训练循环 ==========
    best_mae, best_mse, best_epoch = float('inf'), float('inf'), 0
    best_r2, best_pearson, best_spearman = -float('inf'), -float('inf'), -float('inf')
    best_rmse, best_per_dim_metrics = None, None

    for epoch in range(args.n_epoch):
        epoch_start = time.time()
        model.train()
        for xs, ys, dom_ids in train_loader:
            xs = xs.to(torch.float32).cuda()
            ys = ys.to(torch.float32).cuda()
            dom_ids = dom_ids.to(torch.long).cuda()
            model.train_step(
                xs, ys, dom_ids,
                sphere_w=args.sphere_loss_weight,
                dis_w=args.discriminator_loss_weight,
                dom_w=args.domain_cls_weight,
                with_argumentation=True,
                ablation=args.ablation,
            )

        epoch_time = time.time() - epoch_start
        model.eval()
        sq_err_sum, abs_err_sum, n = 0.0, 0.0, 0
        all_preds, all_ys = [], []
        with torch.no_grad():
            for xt, yt, _ in valid_loader:
                xt = xt.to(torch.float32).cuda()
                yt = yt.to(torch.float32).cuda()
                pred = model.predict_target(xt)
                sq_err_sum += torch.sum((pred - yt) ** 2).item()
                abs_err_sum += torch.sum(torch.abs(pred - yt)).item()
                n += yt.numel()
                all_preds.append(pred.detach().cpu().numpy())
                all_ys.append(yt.detach().cpu().numpy())

        mse, mae = sq_err_sum / n, abs_err_sum / n
        rmse = np.sqrt(mse)
        all_preds = np.concatenate(all_preds, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)

        ss_res = np.sum((all_ys - all_preds) ** 2)
        ss_tot = np.sum((all_ys - np.mean(all_ys)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        pearson_corr = 0.0
        spearman_corr = 0.0
        n_dim = all_ys.shape[1]
        per_dim_metrics = []
        for dim in range(n_dim):
            p_corr, _ = pearsonr(all_ys[:, dim], all_preds[:, dim])
            s_corr, _ = spearmanr(all_ys[:, dim], all_preds[:, dim])
            pearson_corr += p_corr if not np.isnan(p_corr) else 0.0
            spearman_corr += s_corr if not np.isnan(s_corr) else 0.0
            ss_res_dim = np.sum((all_ys[:, dim] - all_preds[:, dim]) ** 2)
            ss_tot_dim = np.sum((all_ys[:, dim] - np.mean(all_ys[:, dim])) ** 2)
            r2_dim = 1 - (ss_res_dim / ss_tot_dim) if ss_tot_dim != 0 else 0.0
            per_dim_metrics.append({
                'dim': dim,
                'pearson': p_corr if not np.isnan(p_corr) else 0.0,
                'spearman': s_corr if not np.isnan(s_corr) else 0.0,
                'r2': r2_dim,
                'mse': np.mean((all_ys[:, dim] - all_preds[:, dim]) ** 2),
                'mae': np.mean(np.abs(all_ys[:, dim] - all_preds[:, dim])),
            })
        pearson_corr /= n_dim
        spearman_corr /= n_dim

        if mae < best_mae:
            best_mae, best_mse, best_epoch = mae, mse, epoch
            best_r2, best_pearson, best_spearman = r2, pearson_corr, spearman_corr
            best_rmse = rmse
            best_per_dim_metrics = per_dim_metrics
            save_path = os.path.join(result_dir, "model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Best model saved to {save_path}")

        print('Target %s, Epoch %d [%.2fs], MSE: %.6f, MAE: %.6f, RMSE: %.6f, R2: %.6f, Pearson: %.6f, Spearman: %.6f' %
              (str(target_domain_ids), epoch, epoch_time, mse, mae, rmse, r2, pearson_corr, spearman_corr))

    print('Target %s, Best Epoch %d: MSE=%.6f, MAE=%.6f, RMSE=%.6f, R2=%.6f, Pearson=%.6f, Spearman=%.6f' %
          (str(target_domain_ids), best_epoch, best_mse, best_mae, best_rmse,
           best_r2, best_pearson, best_spearman))

    # ========== 最终评估 (val + test) ==========
    model.load_state_dict(torch.load(os.path.join(result_dir, "model_best.pth")))
    model.eval()

    def evaluate_dataset(loader, name):
        sq_err_sum, abs_err_sum, n = 0.0, 0.0, 0
        all_preds, all_ys = [], []
        with torch.no_grad():
            for xt, yt, _ in loader:
                xt = xt.to(torch.float32).cuda()
                yt = yt.to(torch.float32).cuda()
                pred = model.predict_target(xt)
                sq_err_sum += torch.sum((pred - yt) ** 2).item()
                abs_err_sum += torch.sum(torch.abs(pred - yt)).item()
                n += yt.numel()
                all_preds.append(pred.detach().cpu().numpy())
                all_ys.append(yt.detach().cpu().numpy())
        mse = sq_err_sum / n
        mae = abs_err_sum / n
        rmse = np.sqrt(mse)
        all_preds = np.concatenate(all_preds, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)
        ss_res = np.sum((all_ys - all_preds) ** 2)
        ss_tot = np.sum((all_ys - np.mean(all_ys)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        pearson_corr = 0.0
        spearman_corr = 0.0
        n_dim = all_ys.shape[1]
        per_dim_metrics = []
        r2_scores = []
        for dim in range(n_dim):
            p_corr, _ = pearsonr(all_ys[:, dim], all_preds[:, dim])
            s_corr, _ = spearmanr(all_ys[:, dim], all_preds[:, dim])
            pearson_corr += p_corr if not np.isnan(p_corr) else 0.0
            spearman_corr += s_corr if not np.isnan(s_corr) else 0.0
            ss_res_dim = np.sum((all_ys[:, dim] - all_preds[:, dim]) ** 2)
            ss_tot_dim = np.sum((all_ys[:, dim] - np.mean(all_ys[:, dim])) ** 2)
            r2_dim = 1 - (ss_res_dim / ss_tot_dim) if ss_tot_dim != 0 else 0.0
            r2_scores.append(r2_dim)
            per_dim_metrics.append({
                'dim': dim,
                'pearson': p_corr if not np.isnan(p_corr) else 0.0,
                'spearman': s_corr if not np.isnan(s_corr) else 0.0,
                'r2': r2_dim,
                'mse': np.mean((all_ys[:, dim] - all_preds[:, dim]) ** 2),
                'mae': np.mean(np.abs(all_ys[:, dim] - all_preds[:, dim])),
            })
        pearson_corr /= n_dim
        spearman_corr /= n_dim
        avg_r2 = np.mean(r2_scores)
        print(f"{name} - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
              f"R2: {r2:.6f}, Avg R2: {avg_r2:.6f}, Pearson: {pearson_corr:.6f}, Spearman: {spearman_corr:.6f}")
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'pearson': pearson_corr, 'spearman': spearman_corr,
            'avg_r2': avg_r2, 'per_dim_metrics': per_dim_metrics
        }

    print(f"\n=== 在验证集上评估最佳模型 ===")
    val_results = evaluate_dataset(valid_loader, "验证集")
    print(f"\n=== 在测试集上评估最佳模型 ===")
    test_results = evaluate_dataset(test_loader, "测试集")

    EMOTION_LABELS = [
        'inspired', 'alert', 'determined', 'attentive', 'active',
        'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
    ]

    # 保存 results.txt
    with open(os.path.join(result_dir, "results.txt"), 'w') as f:
        f.write(f"=== CADT Training Results ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Split Mode: {args.split_mode}\n")
        f.write(f"Split Dir: {args.split_dir}\n")
        if args.target_subject is not None:
            f.write(f"Target Subject: {args.target_subject}\n")
        else:
            f.write(f"Target Subject: all subjects\n")
        f.write(f"Target Domain IDs: {target_domain_ids}\n")
        f.write(f"Number of Source Domains: {args.n_sources}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Ablation: {args.ablation}\n")
        f.write(f"Encoder: {args.encoder_type}\n")
        f.write(f"t_size: {args.t_size}\n")

        f.write(f"\n--- Domain Information ---\n")
        f.write(f"Training domains (raw IDs): {unique_domains}\n")
        f.write(f"Training domain counts: {train_domain_counts}\n")
        f.write(f"Validation domains (raw IDs): {target_domain_ids}\n")
        f.write(f"Validation domain counts: {val_domain_counts}\n")
        f.write(f"Test domains (raw IDs): {test_domain_ids}\n")
        f.write(f"Test domain counts: {test_domain_counts}\n")
        if args.split_mode == "cross_subject":
            f.write(f"\n--- Subject Information (cross_subject) ---\n")
            f.write(f"Training subjects: {unique_domains} (total {len(unique_domains)})\n")
            f.write(f"Validation subjects: {target_domain_ids} (total {len(target_domain_ids)})\n")
            f.write(f"Test subjects: {test_domain_ids} (total {len(test_domain_ids)})\n")

        f.write(f"\n--- Validation Set Results (Best Model) ---\n")
        f.write(f"Val MSE: {val_results['mse']:.6f}\n")
        f.write(f"Val MAE: {val_results['mae']:.6f}\n")
        f.write(f"Val RMSE: {val_results['rmse']:.6f}\n")
        f.write(f"Val R2: {val_results['r2']:.6f}\n")
        f.write(f"Val Avg R2 (per-dim): {val_results['avg_r2']:.6f}\n")
        f.write(f"Val Pearson: {val_results['pearson']:.6f}\n")
        f.write(f"Val Spearman: {val_results['spearman']:.6f}\n")

        f.write(f"\n--- Test Set Results (Best Model) ---\n")
        f.write(f"Test MSE: {test_results['mse']:.6f}\n")
        f.write(f"Test MAE: {test_results['mae']:.6f}\n")
        f.write(f"Test RMSE: {test_results['rmse']:.6f}\n")
        f.write(f"Test R2: {test_results['r2']:.6f}\n")
        f.write(f"Test Avg R2 (per-dim): {test_results['avg_r2']:.6f}\n")
        f.write(f"Test Pearson: {test_results['pearson']:.6f}\n")
        f.write(f"Test Spearman: {test_results['spearman']:.6f}\n")

        f.write(f"\n--- Per-Emotion Metrics (Test Set) ---\n")
        f.write(f"{'Emotion':<12} {'MSE':>10} {'MAE':>10} {'R2':>10} {'Pearson':>10} {'Spearman':>10}\n")
        f.write("-" * 70 + "\n")
        for metric in test_results['per_dim_metrics']:
            label = EMOTION_LABELS[metric['dim']] if metric['dim'] < len(EMOTION_LABELS) else f'Emotion_{metric["dim"]}'
            f.write(f"{label:<12} {metric['mse']:>10.4f} {metric['mae']:>10.4f} "
                    f"{metric['r2']:>10.4f} {metric['pearson']:>10.4f} {metric['spearman']:>10.4f}\n")
        avg_r2_all = np.mean([m['r2'] for m in test_results['per_dim_metrics']])
        f.write("-" * 70 + "\n")
        f.write(f"{'Average':<12} {np.mean([m['mse'] for m in test_results['per_dim_metrics']]):>10.4f} "
                f"{np.mean([m['mae'] for m in test_results['per_dim_metrics']]):>10.4f} "
                f"{avg_r2_all:>10.4f} "
                f"{np.mean([m['pearson'] for m in test_results['per_dim_metrics']]):>10.4f} "
                f"{np.mean([m['spearman'] for m in test_results['per_dim_metrics']]):>10.4f}\n")
        f.write(f"\n--- Per-Emotion Metrics (Validation Set) ---\n")
        f.write(f"{'Emotion':<12} {'MSE':>10} {'MAE':>10} {'R2':>10} {'Pearson':>10} {'Spearman':>10}\n")
        f.write("-" * 70 + "\n")
        for metric in val_results['per_dim_metrics']:
            label = EMOTION_LABELS[metric['dim']] if metric['dim'] < len(EMOTION_LABELS) else f'Emotion_{metric["dim"]}'
            f.write(f"{label:<12} {metric['mse']:>10.4f} {metric['mae']:>10.4f} "
                    f"{metric['r2']:>10.4f} {metric['pearson']:>10.4f} {metric['spearman']:>10.4f}\n")
        avg_r2_all_val = np.mean([m['r2'] for m in val_results['per_dim_metrics']])
        f.write("-" * 70 + "\n")
        f.write(f"{'Average':<12} {np.mean([m['mse'] for m in val_results['per_dim_metrics']]):>10.4f} "
                f"{np.mean([m['mae'] for m in val_results['per_dim_metrics']]):>10.4f} "
                f"{avg_r2_all_val:>10.4f} "
                f"{np.mean([m['pearson'] for m in val_results['per_dim_metrics']]):>10.4f} "
                f"{np.mean([m['spearman'] for m in val_results['per_dim_metrics']]):>10.4f}\n")
        f.write(f"\nModel Saved: {save_path}\n")

    # 保存 params.txt
    with open(os.path.join(result_dir, "params.txt"), 'w') as f:
        f.write(f"=== Training Parameters ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n--- Data Parameters ---\n")
        f.write(f"data_dir: {args.data_dir}\n")
        f.write(f"x_dim: {args.x_dim}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"n_targets: {args.n_targets}\n")
        f.write(f"norm_mode: {args.norm_mode}\n")
        f.write(f"\n--- Split Parameters ---\n")
        f.write(f"split_mode: {args.split_mode}\n")
        f.write(f"split_dir: {args.split_dir}\n")
        f.write(f"\n--- Training Parameters ---\n")
        f.write(f"n_epoch: {args.n_epoch}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"\n--- Loss Weights ---\n")
        f.write(f"sphere_loss_weight: {args.sphere_loss_weight}\n")
        f.write(f"discriminator_loss_weight: {args.discriminator_loss_weight}\n")
        f.write(f"domain_cls_weight: {args.domain_cls_weight}\n")
        f.write(f"contrast_tau: {args.contrast_tau}\n")
        f.write(f"d_AE: {args.d_AE}\n")
        f.write(f"\n--- Ablation ---\n")
        f.write(f"ablation: {args.ablation}\n")
        if args.ablation == "variant":
            f.write(f"  (regression + contrastive only; confusion_loss, domain_class_loss, disc_loss removed)\n")
        f.write(f"\n--- Encoder ---\n")
        f.write(f"encoder_type: {args.encoder_type}\n")
        f.write(f"t_size: {args.t_size}\n")
        if args.encoder_type == "mlp":
            f.write(f"  (MLP encoder, input is flattened DE vectors. Verify data_dir points to DE npy dir.)\n")

    print(f"\n训练结果已保存到: {result_dir}")
