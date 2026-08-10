import argparse
import sys
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 将当前目录添加到 Python 路径，确保能导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from dataset.dataset import DailyEEG_data, DailyEEG_SplitData, MultiSourceBatchSampler, get_available_domains, load_emotion_split
from model.CADT import CATDModel

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--x_dim", type=int, default=59, help="EEG channel count")
parser.add_argument("--seq_len", type=int, default=2000, help="10s @ 200Hz")
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
parser.add_argument("--t_size", type=int, default=128,
                     help="embedding size (t_size) in CNNEncoder, results saved under result_base/tsize{xx}/")
parser.add_argument("--data_dir", type=str, default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
parser.add_argument("--target_domain", type=int, default=0,
                     help="must be a domain id actually present in d.npy -- see get_available_domains()")
parser.add_argument("--target_subject", type=int, default=None,
                     help="target subject id for cross-subject adaptation (overrides --target_domain). "
                          "Also used in emotion split's cross_day/within_subject_day mode to filter data by subject")
parser.add_argument("--target_day", type=int, default=None,
                     help="target day/session id for cross-day adaptation (overrides --target_domain)")
parser.add_argument("--adapt_mode", type=str, default="cross_both",
                     choices=["cross_subject", "cross_day", "cross_both"],
                     help="adaptation mode: cross_subject (hold out one subject), "
                          "cross_day (hold out one day across all subjects), "
                          "cross_both (hold out one subject-day pair, i.e., one domain)")
parser.add_argument("--source_domains", type=str, default=None,
                     help="optional list of source domain ids (supports range syntax like '0-50,55,60-100'); if not provided, all available domains except target are used")
parser.add_argument("--source_subjects", type=str, default=None,
                     help="optional list of source subject ids (comma-separated like '0,1,2,3'); expands to all domains of those subjects")
parser.add_argument("--use_emotion_split", action="store_true", default=False,
                     help="Use emotion project's sample-index-based splits instead of domain-based splits")
parser.add_argument("--split_dir", type=str, default="/vePFS-0x0d/DailyEEG/splits_new",
                     help="Path to emotion project's splits directory")
parser.add_argument("--split_mode", type=str, default="cross_subject",
                     choices=["cross_subject", "cross_day", "within_subject_day"],
                     help="Emotion split mode: cross_subject, cross_day, or within_subject_day")
parser.add_argument("--norm_mode", type=str, default="sample",
                     choices=["sample", "day", "subject"],
                     help="Normalization mode: sample (per-sample), day (per-day), subject (per-subject)")
parser.add_argument("--ablation", type=str, default="none",
                     choices=["none", "variant"],
                     help="Ablation mode: 'none' for full model; 'variant' keeps only regression + "
                          "contrastive loss, removing all domain-related losses (confusion/disc/domain_class). "
                          "Encoder cross-branch connection is left intact.")
args = parser.parse_args()


def parse_range_string(range_str):
    """
    解析范围字符串，支持格式如 "0-50,55,60-100"
    返回整数列表
    """
    result = []
    if not range_str:
        return result
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result

def load_domain_metadata(data_dir):
    """Load meta.json to get domain_offset_map and domains_per_subject_map."""
    import json
    import os
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


def get_domain_ids_by_day(day_id, domain_offset_map, domains_per_subject_map):
    """Get all domain IDs with the same day_id across all subjects."""
    domain_ids = []
    for sub_id, offset in domain_offset_map.items():
        domain_ids.append(offset + day_id)
    return domain_ids


if __name__ == '__main__':
    # === GPU 可用性检测 ===
    print(f"\n=== GPU 状态检测 ===")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前设备索引: {args.device}")
        print(f"设备名称: {torch.cuda.get_device_name(args.device)}")
        print(f"设备内存: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: CUDA 不可用！将使用 CPU 运行，速度会很慢！")
    
    torch.cuda.set_device(args.device)
    
    # 创建时间戳命名的结果文件夹
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if args.use_emotion_split:
        if args.target_subject is not None and args.split_mode != "cross_subject":
            target_info = f"emotion_{args.split_mode}_subject{args.target_subject}"
        else:
            target_info = f"emotion_{args.split_mode}"
    else:
        # 根据 adapt_mode 构建目标信息
        if args.adapt_mode == 'cross_subject':
            target_info = f"subject{args.target_subject}"
        elif args.adapt_mode == 'cross_day':
            target_info = f"day{args.target_day}"
        else:
            target_info = f"domain{args.target_domain}"
    
    # 结果文件夹路径（根据标准化模式选择输出目录）
    if args.norm_mode == "day":
        result_base = "result_day_td"
    elif args.norm_mode == "subject":
        result_base = "result_sub_std"
    else:
        result_base = "results4"

    # 消融实验结果存到单独的目录，与完整模型区分开
    if args.ablation != "none":
        result_base = result_base + "_ablation"

    # 按 t_size 分子目录存放，例如 result_sub_std/tsize8/
    result_base = os.path.join(result_base, f"tsize{args.t_size}")

    if args.use_emotion_split:
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_base,
                                  f"{timestamp}_emotion_{args.split_mode}")
    else:
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_base,
                                  f"{timestamp}_{args.adapt_mode}_{target_info}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果将保存到: {result_dir}")
    print(f"标准化模式: {args.norm_mode}")
    if args.ablation != "none":
        print(f"消融模式: {args.ablation}")

    if args.use_emotion_split:
        # 使用 emotion 项目的样本索引划分
        print(f"\n=== 使用 Emotion 项目划分 ===")
        print(f"  Split mode: {args.split_mode}")
        print(f"  Split dir: {args.split_dir}")
        
        train_indices = load_emotion_split(args.split_dir, args.split_mode, "pretrain")
        train_indices += load_emotion_split(args.split_dir, args.split_mode, "finetune")
        val_indices = load_emotion_split(args.split_dir, args.split_mode, "val")
        test_indices = load_emotion_split(args.split_dir, args.split_mode, "test")
        
        # 如果指定了 target_subject，过滤出该被试的数据（仅适用于 cross_day 和 within_subject_day）
        if args.target_subject is not None and args.split_mode != "cross_subject":
            print(f"  \n  === 按被试 {args.target_subject} 过滤数据 ===")
            
            # 加载 sub.npy 来获取样本对应的被试信息
            sub_path = os.path.join(args.data_dir, "sub.npy")
            all_sub = np.load(sub_path)
            
            # 过滤训练集
            train_mask = all_sub[train_indices] == args.target_subject
            train_indices = [idx for idx, mask in zip(train_indices, train_mask) if mask]
            
            # 过滤验证集
            val_mask = all_sub[val_indices] == args.target_subject
            val_indices = [idx for idx, mask in zip(val_indices, val_mask) if mask]
            
            # 过滤测试集
            test_mask = all_sub[test_indices] == args.target_subject
            test_indices = [idx for idx, mask in zip(test_indices, test_mask) if mask]
            
            print(f"  过滤后 - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        print(f"  Train samples: {len(train_indices)} (pretrain + finetune)")
        print(f"  Val samples: {len(val_indices)}")
        print(f"  Test samples: {len(test_indices)}")
        
        data_train = DailyEEG_SplitData(train_indices, split_mode=args.split_mode, data_dir=args.data_dir,
                                         norm_mode=args.norm_mode)
        args.n_sources = len(np.unique(data_train.d))
        
        unique_domains = sorted(np.unique(data_train.d).tolist())
        id_map = {raw: i for i, raw in enumerate(unique_domains)}
        print(f"  Training domains: {args.n_sources} unique domains")
        
        # 输出详细的域信息
        print(f"  \n  === 域信息详情 ===")
        print(f"  训练集域 (原始ID): {unique_domains}")
        
        # 统计每个域的样本数
        train_domain_counts = {d: int((data_train.d == d).sum()) for d in unique_domains}
        print(f"  训练集各域样本数: {train_domain_counts}")
        
        data_train.d = np.array([id_map[int(d)] for d in data_train.d], dtype=np.int64)
        sampler = MultiSourceBatchSampler(data_train.d, list(id_map.values()), batch_size=args.batch_size)
        train_loader = DataLoader(data_train, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        
        data_target = DailyEEG_SplitData(val_indices, split_mode=args.split_mode, data_dir=args.data_dir,
                                          norm_mode=args.norm_mode)
        valid_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        
        target_domain_ids = sorted(np.unique(data_target.d).tolist())
        print(f"  验证集域 (原始ID): {target_domain_ids}")
        val_domain_counts = {d: int((data_target.d == d).sum()) for d in target_domain_ids}
        print(f"  验证集各域样本数: {val_domain_counts}")
        
        # 创建测试集（emotion split 专门有 test 划分）
        data_test = DailyEEG_SplitData(test_indices, split_mode=args.split_mode, data_dir=args.data_dir,
                                        norm_mode=args.norm_mode)
        test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        
        test_domain_ids = sorted(np.unique(data_test.d).tolist())
        print(f"  测试集域 (原始ID): {test_domain_ids}")
        test_domain_counts = {d: int((data_test.d == d).sum()) for d in test_domain_ids}
        print(f"  测试集各域样本数: {test_domain_counts}")
        
        # 在 cross_subject 模式下，额外显示被试信息
        if args.split_mode == "cross_subject":
            print(f"  \n  === 被试信息 (cross_subject 模式) ===")
            print(f"  训练被试: {unique_domains} (共 {len(unique_domains)} 个)")
            print(f"  验证被试: {target_domain_ids} (共 {len(target_domain_ids)} 个)")
            print(f"  测试被试: {test_domain_ids} (共 {len(test_domain_ids)} 个)")
        if args.split_mode == "cross_subject":
            print(f"  \n  === 被试信息 (cross_subject 模式) ===")
            print(f"  训练被试: {unique_domains} (共 {len(unique_domains)} 个)")
            print(f"  验证被试: {target_domain_ids} (共 {len(target_domain_ids)} 个)")
            print(f"  测试被试: {test_domain_ids} (共 {len(test_domain_ids)} 个)")
        
    else:
        # 使用传统的域划分方式
        available_domains = get_available_domains(args.data_dir)
        meta = load_domain_metadata(args.data_dir)
        domain_offset_map = meta['domain_offset_map']
        domains_per_subject_map = meta['domains_per_subject_map']

        # 根据 adapt_mode 确定目标域 ID 列表
        target_domain_ids = []
        
        if args.adapt_mode == 'cross_subject':
            # 跨被试：目标域是某个被试的所有 day
            if args.target_subject is None:
                raise ValueError("cross_subject mode requires --target_subject argument")
            target_domain_ids = get_domain_ids_by_subject(args.target_subject, domain_offset_map, domains_per_subject_map)
            if not target_domain_ids:
                raise ValueError(f"subject {args.target_subject} has no domains in data")
            print(f"Cross-subject adaptation: target_subject={args.target_subject}, "
                  f"target_domains={target_domain_ids}")
        
        elif args.adapt_mode == 'cross_day':
            # 跨天：目标域是所有被试的同一天
            if args.target_day is None:
                raise ValueError("cross_day mode requires --target_day argument")
            target_domain_ids = get_domain_ids_by_day(args.target_day, domain_offset_map, domains_per_subject_map)
            if not target_domain_ids:
                raise ValueError(f"day {args.target_day} has no domains in data")
            print(f"Cross-day adaptation: target_day={args.target_day}, "
                  f"target_domains={target_domain_ids}")
        
        else:
            # cross_both：目标域是单个域（被试-天对）
            target_domain_ids = [args.target_domain]
            if args.target_domain not in available_domains:
                raise ValueError(f"target_domain={args.target_domain} not present in data "
                                  f"(available: {len(available_domains)} domains, "
                                  f"range [{available_domains[0]}, {available_domains[-1]}], "
                                  f"note some ids like 124 may be missing)")
            print(f"Cross-subject-cross-day adaptation: target_domain={args.target_domain}")

        # 过滤掉不存在的目标域（某些域可能因数据缺失而不存在，如域124）
        original_target_ids = target_domain_ids.copy()
        target_domain_ids = [d for d in target_domain_ids if d in available_domains]
        
        # 检查是否有域被过滤
        missing_domains = set(original_target_ids) - set(target_domain_ids)
        if missing_domains:
            print(f"WARNING: {len(missing_domains)} target domain(s) not found in data and will be skipped: {sorted(missing_domains)}")
        
        # 确保过滤后还有目标域
        if not target_domain_ids:
            raise ValueError(f"No target domains remaining after filtering. Check your target_subject/target_day settings.")

        # 源域推导逻辑：手动指定 or 自动推导
        if args.source_subjects is not None:
            # 使用指定的源被试，展开为域 ID
            subject_ids = parse_range_string(args.source_subjects)
            original_source_ids = []
            for sub_id in subject_ids:
                domains = get_domain_ids_by_subject(sub_id, domain_offset_map, domains_per_subject_map)
                original_source_ids.extend(domains)
            
            # 过滤掉不存在的源域
            source_ids = [d for d in original_source_ids if d in available_domains]
            
            # 检查是否有域被过滤
            missing_source_domains = set(original_source_ids) - set(source_ids)
            if missing_source_domains:
                print(f"WARNING: {len(missing_source_domains)} source domain(s) not found in data and will be skipped: {sorted(missing_source_domains)}")
            
            # 检查过滤后还有源域
            if not source_ids:
                raise ValueError("No source domains remaining after filtering. Check your source_subjects settings.")
            
            # 验证目标域不在源域列表中
            for target_d in target_domain_ids:
                if target_d in source_ids:
                    raise ValueError(f"target_domain={target_d} cannot be in source_domains")
            
            print(f"Using source subjects {subject_ids} -> {len(source_ids)} domains")
        
        elif args.source_domains is not None:
            # 使用手动指定的源域列表（支持范围语法）
            original_source_ids = parse_range_string(args.source_domains)
            # 过滤掉不存在的源域
            source_ids = [d for d in original_source_ids if d in available_domains]
            # 检查是否有域被过滤
            missing_source_domains = set(original_source_ids) - set(source_ids)
            if missing_source_domains:
                print(f"WARNING: {len(missing_source_domains)} source domain(s) not found in data and will be skipped: {sorted(missing_source_domains)}")
            # 检查过滤后还有源域
            if not source_ids:
                raise ValueError("No source domains remaining after filtering. Check your source_domains settings.")
            # 验证目标域不在源域列表中
            for target_d in target_domain_ids:
                if target_d in source_ids:
                    raise ValueError(f"target_domain={target_d} cannot be in source_domains")
            print(f"Using specified source domains: {len(source_ids)} domains (from range: {args.source_domains})")
        
        else:
            # 自动推导：所有可用域除了目标域
            source_ids = [d for d in available_domains if d not in target_domain_ids]
            print(f"Auto-derived source domains: {len(source_ids)} domains")

        id_map = {raw: i for i, raw in enumerate(source_ids)}
        args.n_sources = len(source_ids)

        data_train = DailyEEG_data(source_ids, data_dir=args.data_dir, id_map=id_map,
                                    norm_mode=args.norm_mode)
        sampler = MultiSourceBatchSampler(data_train.d, list(id_map.values()), batch_size=args.batch_size)
        train_loader = DataLoader(data_train, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

        data_target = DailyEEG_data(target_domain_ids, data_dir=args.data_dir,
                                     norm_mode=args.norm_mode)   # no id_map -- not used for domain-id losses
        valid_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    model = CATDModel(args).cuda()

    # --- shape sanity check before a full training run ---
    x_dummy = torch.randn(4, args.seq_len, args.x_dim).cuda()
    zsc_dummy, zs_dummy = model.share_encoder(x_dummy)
    print(f"Sanity check -- zsc: {zsc_dummy.shape}, zs: {zs_dummy.shape} (expect (4, 8) each)")

    best_mae, best_mse, best_epoch = float('inf'), float('inf'), 0
    best_r2, best_pearson, best_spearman = -float('inf'), -float('inf'), -float('inf')
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
        all_preds = []
        all_ys = []
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
        
        # 计算 R²、Pearson 和 Spearman 相关系数
        all_preds = np.concatenate(all_preds, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)
        
        # R²: 1 - SS_res / SS_tot
        ss_res = np.sum((all_ys - all_preds) ** 2)
        ss_tot = np.sum((all_ys - np.mean(all_ys)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Pearson 和 Spearman 相关系数（按维度计算，取平均）
        pearson_corr = 0.0
        spearman_corr = 0.0
        n_dim = all_ys.shape[1]
        per_dim_metrics = []
        for dim in range(n_dim):
            p_corr, _ = pearsonr(all_ys[:, dim], all_preds[:, dim])
            s_corr, _ = spearmanr(all_ys[:, dim], all_preds[:, dim])
            pearson_corr += p_corr if not np.isnan(p_corr) else 0.0
            spearman_corr += s_corr if not np.isnan(s_corr) else 0.0
            # 计算每个维度的 R2
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
            # 保存最佳模型
            model_name = f"model_best.pth"
            save_path = os.path.join(result_dir, model_name)
            torch.save(model.state_dict(), save_path)
            print(f"  Best model saved to {save_path}")
        
        print('Target %s, Epoch %d [%.2fs], MSE: %.6f, MAE: %.6f, RMSE: %.6f, R2: %.6f, Pearson: %.6f, Spearman: %.6f' %
              (str(target_domain_ids), epoch, epoch_time, mse, mae, rmse, r2, pearson_corr, spearman_corr))

    print('Target %s, Best Epoch %d: MSE=%.6f, MAE=%.6f, RMSE=%.6f, R2=%.6f, Pearson=%.6f, Spearman=%.6f' % 
          (str(target_domain_ids), best_epoch, best_mse, best_mae, best_rmse, 
           best_r2, best_pearson, best_spearman))

    # 加载最佳模型并进行最终评估
    model.load_state_dict(torch.load(os.path.join(result_dir, "model_best.pth")))
    model.eval()
    
    def evaluate_dataset(loader, name):
        """评估数据集并返回指标"""
        sq_err_sum, abs_err_sum, n = 0.0, 0.0, 0
        all_preds = []
        all_ys = []
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
            # 计算每个维度的 R2
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
        
        # 计算平均 R2（按维度平均）
        avg_r2 = np.mean(r2_scores)
        
        print(f"{name}结果 - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"{name}结果 - R2: {r2:.6f}, Avg R2 (per-dim): {avg_r2:.6f}, Pearson: {pearson_corr:.6f}, Spearman: {spearman_corr:.6f}")
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'pearson': pearson_corr, 'spearman': spearman_corr,
            'avg_r2': avg_r2, 'per_dim_metrics': per_dim_metrics
        }
    
    # 在验证集上评估最佳模型
    print(f"\n=== 在验证集上评估最佳模型 ===")
    val_results = evaluate_dataset(valid_loader, "验证集")
    
    # 在测试集上评估最佳模型
    print(f"\n=== 在测试集上评估最佳模型 ===")
    if args.use_emotion_split:
        test_results = evaluate_dataset(test_loader, "测试集")
    else:
        # 传统模式下，验证集和测试集使用相同的数据
        print("警告：传统模式下没有单独的测试集划分，测试集结果与验证集相同")
        test_results = val_results.copy()

    # 情绪标签名称
    EMOTION_LABELS = [
        'inspired', 'alert', 'determined', 'attentive', 'active',
        'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
    ]

    # 保存训练结果到文件
    result_path = os.path.join(result_dir, "results.txt")
    
    with open(result_path, 'w') as f:
        f.write(f"=== CADT Training Results ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        if args.use_emotion_split:
            f.write(f"Split Mode: {args.split_mode}\n")
            f.write(f"Split Dir: {args.split_dir}\n")
            if args.target_subject is not None:
                f.write(f"Target Subject: {args.target_subject}\n")
            else:
                f.write(f"Target Subject: all subjects\n")
        else:
            f.write(f"Adapt Mode: {args.adapt_mode}\n")
            if args.adapt_mode == 'cross_subject':
                f.write(f"Target Subject: {args.target_subject}\n")
            elif args.adapt_mode == 'cross_day':
                f.write(f"Target Day: {args.target_day}\n")
            else:
                f.write(f"Target Domain: {args.target_domain}\n")
        f.write(f"Target Domain IDs: {target_domain_ids}\n")
        f.write(f"Number of Source Domains: {args.n_sources}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Ablation: {args.ablation}\n")
        
        # emotion split 模式下保存详细的域信息
        if args.use_emotion_split:
            f.write(f"\n--- Domain Information ---")
            f.write(f"\nTraining domains (raw IDs): {unique_domains}\n")
            f.write(f"Training domain counts: {train_domain_counts}\n")
            f.write(f"Validation domains (raw IDs): {target_domain_ids}\n")
            f.write(f"Validation domain counts: {val_domain_counts}\n")
            f.write(f"Test domains (raw IDs): {test_domain_ids}\n")
            f.write(f"Test domain counts: {test_domain_counts}\n")
            if args.split_mode == "cross_subject":
                f.write(f"\n--- Subject Information (cross_subject mode) ---")
                f.write(f"\nTraining subjects: {unique_domains} (total {len(unique_domains)})\n")
                f.write(f"Validation subjects: {target_domain_ids} (total {len(target_domain_ids)})\n")
                f.write(f"Test subjects: {test_domain_ids} (total {len(test_domain_ids)})\n")
        
        # 验证集结果（最优模型）
        f.write(f"\n--- Validation Set Results (Best Model) ---\n")
        f.write(f"Val MSE: {val_results['mse']:.6f}\n")
        f.write(f"Val MAE: {val_results['mae']:.6f}\n")
        f.write(f"Val RMSE: {val_results['rmse']:.6f}\n")
        f.write(f"Val R2: {val_results['r2']:.6f}\n")
        f.write(f"Val Avg R2 (per-dim): {val_results['avg_r2']:.6f}\n")
        f.write(f"Val Pearson: {val_results['pearson']:.6f}\n")
        f.write(f"Val Spearman: {val_results['spearman']:.6f}\n")
        
        # 测试集结果
        f.write(f"\n--- Test Set Results (Best Model) ---\n")
        if not args.use_emotion_split:
            f.write(f"Note: Traditional mode uses validation set as test set\n")
        f.write(f"Test MSE: {test_results['mse']:.6f}\n")
        f.write(f"Test MAE: {test_results['mae']:.6f}\n")
        f.write(f"Test RMSE: {test_results['rmse']:.6f}\n")
        f.write(f"Test R2: {test_results['r2']:.6f}\n")
        f.write(f"Test Avg R2 (per-dim): {test_results['avg_r2']:.6f}\n")
        f.write(f"Test Pearson: {test_results['pearson']:.6f}\n")
        f.write(f"Test Spearman: {test_results['spearman']:.6f}\n")
        f.write(f"\n--- Per-Emotion Metrics ---")
        f.write(f"{'Emotion':<12} {'MSE':>10} {'MAE':>10} {'R2':>10} {'Pearson':>10} {'Spearman':>10}\n")
        f.write("-" * 70 + "\n")
        for metric in best_per_dim_metrics:
            label = EMOTION_LABELS[metric['dim']] if metric['dim'] < len(EMOTION_LABELS) else f'Emotion_{metric["dim"]}'
            f.write(f"{label:<12} {metric['mse']:>10.4f} {metric['mae']:>10.4f} "
                    f"{metric['r2']:>10.4f} {metric['pearson']:>10.4f} {metric['spearman']:>10.4f}\n")
        # 计算并写入平均 R2
        avg_r2_all = np.mean([m['r2'] for m in best_per_dim_metrics])
        f.write("-" * 70 + "\n")
        f.write(f"{'Average':<12} {np.mean([m['mse'] for m in best_per_dim_metrics]):>10.4f} "
                f"{np.mean([m['mae'] for m in best_per_dim_metrics]):>10.4f} "
                f"{avg_r2_all:>10.4f} "
                f"{np.mean([m['pearson'] for m in best_per_dim_metrics]):>10.4f} "
                f"{np.mean([m['spearman'] for m in best_per_dim_metrics]):>10.4f}\n")
        f.write(f"\nModel Saved: {save_path}\n")
    
    # 保存参数设置到文件
    params_path = os.path.join(result_dir, "params.txt")
    with open(params_path, 'w') as f:
        f.write(f"=== Training Parameters ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n--- Data Parameters ---\n")
        f.write(f"data_dir: {args.data_dir}\n")
        f.write(f"x_dim: {args.x_dim}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"n_targets: {args.n_targets}\n")
        f.write(f"norm_mode: {args.norm_mode}\n")
        f.write(f"\n--- Adaptation Parameters ---\n")
        if args.use_emotion_split:
            f.write(f"split_mode: {args.split_mode}\n")
            f.write(f"split_dir: {args.split_dir}\n")
        else:
            f.write(f"adapt_mode: {args.adapt_mode}\n")
            if args.adapt_mode == 'cross_subject':
                f.write(f"target_subject: {args.target_subject}\n")
            elif args.adapt_mode == 'cross_day':
                f.write(f"target_day: {args.target_day}\n")
            else:
                f.write(f"target_domain: {args.target_domain}\n")
            if args.source_subjects:
                f.write(f"source_subjects: {args.source_subjects}\n")
            if args.source_domains:
                f.write(f"source_domains: {args.source_domains}\n")
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
    
    print(f"\n训练结果已保存到: {result_dir}")