#!/usr/bin/env python3
"""
Fix existing results.txt files:
  - The original Per-Emotion Metrics section only had validation-set metrics
    (from best_per_dim_metrics during training).
  - This script reloads model_best.pth, re-evaluates val+test sets,
    and rewrites results.txt with two separate per-emotion tables:
    Test Set first, then Validation Set.

Usage:
  python fix_results.py [--root-dir DIR] [--device 0]

  If --root-dir is omitted, defaults to processing the three known dirs:
    /root/zjm/cadt/result_day_td
    /root/zjm/cadt/result_day_td_ablation
    /root/zjm/cadt/results_mlp
"""

import argparse
import os
import re
import sys
import copy

# --- TMPDIR fix (must happen BEFORE scipy import) ---
_tmp_dir = "/root/zjm/cadt_test_code/tmp"
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.CADT import CATDModel
from dataset.dataset import DailyEEG_SplitData, load_emotion_split
from torch.utils.data import DataLoader

EMOTION_LABELS = [
    'inspired', 'alert', 'determined', 'attentive', 'active',
    'hostile', 'nervous', 'upset', 'afraid', 'ashamed', 'fatigue'
]

DEFAULT_ROOTS = [
    "/root/zjm/cadt/result_day_td",
    "/root/zjm/cadt/result_day_td_ablation",
    #"/root/zjm/cadt/results_mlp",
]


def parse_params_txt(params_path):
    """Parse params.txt into a dict."""
    cfg = {}
    with open(params_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('==='):
                continue
            if ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                if val == '':
                    continue
                # Try to convert to int / float
                try:
                    if '.' in val or 'e' in val.lower():
                        cfg[key] = float(val)
                    else:
                        cfg[key] = int(val)
                except ValueError:
                    cfg[key] = val
    return cfg


def parse_results_txt(results_path):
    """Parse results.txt for quick reference values. Returns dict."""
    info = {}
    with open(results_path, 'r') as f:
        text = f.read()
    # Target Subject
    m = re.search(r'Target Subject:\s*(.+)', text)
    if m:
        ts = m.group(1).strip()
        if ts.lower().startswith('all'):
            info['target_subject'] = None
        else:
            try:
                info['target_subject'] = int(ts)
            except ValueError:
                info['target_subject'] = None
    else:
        info['target_subject'] = None
    return info


def load_model_auto(model_path, cfg):
    """Load model_best.pth, auto-detecting architecture details from state_dict.
    cfg is parsed params.txt dict, used as hints / fallbacks.
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # --- encoder_type (detect FIRST, needed for t_size detection) ---
    has_cnn_keys = any(k.startswith('share_encoder.feature_') or
                       k.startswith('share_encoder.conv') or
                       k.startswith('share_encoder.fc11') or
                       k.startswith('share_encoder.fc12')
                       for k in state_dict.keys())
    has_mlp_keys = any(k.startswith('share_encoder.trunk') or
                       k.startswith('share_encoder.head_zsc')
                       for k in state_dict.keys())
    if has_mlp_keys and not has_cnn_keys:
        encoder_type = 'mlp'
    elif has_cnn_keys and not has_mlp_keys:
        encoder_type = 'cnn'
    else:
        encoder_type = str(cfg.get('encoder_type', 'cnn'))

    # --- n_sources ---
    n_sources = 1
    for k, v in state_dict.items():
        if 'discriminator.model.4.weight' in k:
            n_sources = int(v.shape[0])
            break

    # --- t_size ---
    t_size = None
    if encoder_type == 'mlp':
        # MLP head_zsc = Sequential([_mlp_block, Linear(256, t_size)])
        #   last linear is at index 1: head_zsc.1.weight  shape=(t_size, 256)
        key = 'share_encoder.head_zsc.1.weight'
        if key in state_dict:
            t_size = int(state_dict[key].shape[0])
    else:  # cnn
        # CNN: fc11.0.weight OR head_zsc.3.weight (depending on variant)
        for k in state_dict.keys():
            if k in ('share_encoder.fc11.0.weight', 'share_encoder.head_zsc.3.weight') or \
               k.endswith('.head_zsc.3.weight') or k.endswith('.fc11.0.weight'):
                t_size = int(state_dict[k].shape[0])
                break
    if t_size is None:
        # Fallback: look at the last weight of regressor.linear → (11, t_size)
        rkey = 'regressor.linear.weight'
        if rkey in state_dict:
            t_size = int(state_dict[rkey].shape[1])
    if t_size is None:
        t_size = int(cfg.get('t_size', 128))

    # --- x_dim, seq_len ---
    if encoder_type == 'mlp':
        trunk_key = 'share_encoder.trunk.0.weight'
        if trunk_key in state_dict:
            in_dim = int(state_dict[trunk_key].shape[1])
            x_dim = int(cfg.get('x_dim', 295))
            seq_len = int(cfg.get('seq_len', 1))
            if x_dim * seq_len != in_dim:
                x_dim = in_dim
                seq_len = 1
        else:
            x_dim = int(cfg.get('x_dim', 295))
            seq_len = int(cfg.get('seq_len', 1))
    else:
        x_dim = int(cfg.get('x_dim', 59))
        seq_len = int(cfg.get('seq_len', 2000))

    d_AE = int(cfg.get('d_AE', 128))
    n_targets = int(cfg.get('n_targets', 11))
    device = int(cfg.get('device', 0))

    # Build an args namespace
    class Args:
        pass
    args = Args()
    args.x_dim = x_dim
    args.seq_len = seq_len
    args.t_size = t_size
    args.d_AE = d_AE
    args.n_targets = n_targets
    args.n_sources = n_sources
    args.encoder_type = encoder_type
    args.device = device

    use_cuda = torch.cuda.is_available() and device is not None
    if use_cuda:
        torch.cuda.set_device(device)
        model = CATDModel(args).cuda()
        args._device_str = f"cuda:{device}"
    else:
        model = CATDModel(args)
        args._device_str = "cpu"
        if not torch.cuda.is_available():
            print("  [load_model] CUDA not available, falling back to CPU")
    model_sd = model.state_dict()
    # Manual filter: only keep keys that exist AND have matching shape
    filtered = {}
    skipped_shape = []
    for k, v in state_dict.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
    missing = [k for k in model_sd.keys() if k not in state_dict]
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"  [load_model] encoder={encoder_type} t_size={t_size} x_dim={x_dim} seq_len={seq_len} "
          f"n_sources={n_sources} | matched {len(filtered)}/{len(state_dict)} params, missing {len(missing)}"
          f" | device={args._device_str}")
    if skipped_shape:
        print(f"  [load_model] skipped {len(skipped_shape)} keys due to shape mismatch")
        for k, src_shape, dst_shape in skipped_shape[:5]:
            print(f"    - {k}: ckpt={src_shape} vs model={dst_shape}")
    return model, args


def build_loaders(args, cfg, info):
    """Build val & test DataLoaders using DailyEEG_SplitData + load_emotion_split."""
    split_mode = str(cfg.get('split_mode', 'cross_subject'))
    split_dir = str(cfg.get('split_dir', '/vePFS-0x0d/DailyEEG/splits_new'))
    data_dir = str(cfg.get('data_dir', '/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new'))
    norm_mode = str(cfg.get('norm_mode', 'sample'))
    batch_size = int(cfg.get('batch_size', 256))
    num_workers = 4
    target_subject = info.get('target_subject', None)

    val_indices = load_emotion_split(split_dir, split_mode, "val")
    test_indices = load_emotion_split(split_dir, split_mode, "test")

    if target_subject is not None and split_mode != "cross_subject":
        sub_path = os.path.join(data_dir, "sub.npy")
        if os.path.exists(sub_path):
            all_sub = np.load(sub_path)
            val_indices = [idx for idx in val_indices if all_sub[idx] == target_subject]
            test_indices = [idx for idx in test_indices if all_sub[idx] == target_subject]
        print(f"  [build_loaders] filtered by target_subject={target_subject}: "
              f"val={len(val_indices)}, test={len(test_indices)}")
    else:
        print(f"  [build_loaders] val={len(val_indices)}, test={len(test_indices)} (split_mode={split_mode})")

    data_val = DailyEEG_SplitData(val_indices, split_mode=split_mode,
                                  data_dir=data_dir, norm_mode=norm_mode)
    data_test = DailyEEG_SplitData(test_indices, split_mode=split_mode,
                                   data_dir=data_dir, norm_mode=norm_mode)

    valid_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=num_workers, pin_memory=True)
    return valid_loader, test_loader


def evaluate_dataset(loader, name):
    """Evaluate best model on a dataset. Returns dict with per_dim_metrics."""
    sq_err_sum, abs_err_sum, n = 0.0, 0.0, 0
    all_preds, all_ys = [], []
    _device = _MODEL_STATE['device']
    _model = _MODEL_STATE['model']
    with torch.no_grad():
        for xt, yt, _ in loader:
            xt = xt.to(torch.float32).to(_device)
            yt = yt.to(torch.float32).to(_device)
            pred = _model.predict_target(xt)
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
            'mse': float(np.mean((all_ys[:, dim] - all_preds[:, dim]) ** 2)),
            'mae': float(np.mean(np.abs(all_ys[:, dim] - all_preds[:, dim]))),
        })
    pearson_corr /= n_dim
    spearman_corr /= n_dim
    avg_r2 = float(np.mean(r2_scores))
    print(f"  [eval {name}] MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, "
          f"R2={r2:.6f}, AvgR2={avg_r2:.6f}, Pearson={pearson_corr:.6f}, Spearman={spearman_corr:.6f}")
    return {
        'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
        'pearson': pearson_corr, 'spearman': spearman_corr,
        'avg_r2': avg_r2, 'per_dim_metrics': per_dim_metrics,
    }


_MODEL_STATE = {'model': None, 'device': 'cpu'}


def predict_target_safe(xt):
    """Predict using model stored in global _MODEL_STATE (kept only for compat)."""
    return _MODEL_STATE['model'].predict_target(xt)


def find_experiment_dirs(root_dir):
    """Return list of experiment dirs (containing model_best.pth + results.txt) under root_dir."""
    exp_dirs = []
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        if os.path.exists(os.path.join(sub, 'model_best.pth')) and \
           os.path.exists(os.path.join(sub, 'results.txt')):
            exp_dirs.append(sub)
    return exp_dirs


def strip_old_per_emotion(text):
    """Remove the old Per-Emotion Metrics section (and any trailing model-saved line
       that follows it) from results.txt text. Returns (header_text, model_saved_line_or_None).
    """
    # Find the "--- Per-Emotion Metrics" marker
    m = re.search(r'^--- Per-Emotion Metrics.*$', text, flags=re.MULTILINE)
    if not m:
        return text, None
    header = text[:m.start()].rstrip() + "\n"
    rest = text[m.end():]
    # Try to extract Model Saved line
    model_saved = None
    m2 = re.search(r'^Model Saved:.*$', rest, flags=re.MULTILINE)
    if m2:
        model_saved = m2.group(0).strip()
    return header, model_saved


def format_per_dim_table(title, per_dim_metrics):
    lines = []
    lines.append(f"\n--- {title} ---")
    lines.append(f"{'Emotion':<12} {'MSE':>10} {'MAE':>10} {'R2':>10} {'Pearson':>10} {'Spearman':>10}")
    lines.append("-" * 70)
    for metric in per_dim_metrics:
        label = EMOTION_LABELS[metric['dim']] if metric['dim'] < len(EMOTION_LABELS) else f'Emotion_{metric["dim"]}'
        lines.append(f"{label:<12} {metric['mse']:>10.4f} {metric['mae']:>10.4f} "
                     f"{metric['r2']:>10.4f} {metric['pearson']:>10.4f} {metric['spearman']:>10.4f}")
    avg_r2 = float(np.mean([m['r2'] for m in per_dim_metrics]))
    lines.append("-" * 70)
    lines.append(f"{'Average':<12} {np.mean([m['mse'] for m in per_dim_metrics]):>10.4f} "
                 f"{np.mean([m['mae'] for m in per_dim_metrics]):>10.4f} "
                 f"{avg_r2:>10.4f} "
                 f"{np.mean([m['pearson'] for m in per_dim_metrics]):>10.4f} "
                 f"{np.mean([m['spearman'] for m in per_dim_metrics]):>10.4f}")
    return "\n".join(lines) + "\n"


def process_one_dir(exp_dir, device_override=None, force=False):
    """Fix a single experiment directory."""
    print(f"\n{'='*80}")
    print(f"Processing: {exp_dir}")
    print(f"{'='*80}")

    results_path = os.path.join(exp_dir, 'results.txt')
    params_path = os.path.join(exp_dir, 'params.txt')
    model_path = os.path.join(exp_dir, 'model_best.pth')

    # Quick skip-check: if already contains "(Test Set)" in Per-Emotion Metrics, assume already fixed
    if not force:
        with open(results_path, 'r') as f:
            preview = f.read()
        if '--- Per-Emotion Metrics (Test Set) ---' in preview:
            print(f"  [skip] already has split per-emotion metrics, pass --force to reprocess")
            return True

    # --- Read original files ---
    with open(results_path, 'r') as f:
        original_text = f.read()
    cfg = parse_params_txt(params_path)
    if device_override is not None:
        cfg['device'] = device_override
    info = parse_results_txt(results_path)

    # --- Load model ---
    model, args = load_model_auto(model_path, cfg)
    _MODEL_STATE['model'] = model
    _MODEL_STATE['device'] = args._device_str

    # --- Build loaders ---
    valid_loader, test_loader = build_loaders(args, cfg, info)

    # --- Evaluate ---
    val_results = evaluate_dataset(valid_loader, "val")
    test_results = evaluate_dataset(test_loader, "test")

    # --- Rebuild results.txt ---
    header, model_saved = strip_old_per_emotion(original_text)

    new_text = header.rstrip() + "\n"
    new_text += format_per_dim_table("Per-Emotion Metrics (Test Set)", test_results['per_dim_metrics'])
    new_text += format_per_dim_table("Per-Emotion Metrics (Validation Set)", val_results['per_dim_metrics'])
    if model_saved:
        new_text += f"\n{model_saved}\n"

    # Backup original
    backup_path = results_path + ".bak"
    with open(backup_path, 'w') as f:
        f.write(original_text)
    print(f"  [backup] original saved to {backup_path}")

    with open(results_path, 'w') as f:
        f.write(new_text)
    print(f"  [write] updated {results_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix per-emotion metrics sections in existing results.txt files")
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Single root dir to process (omitted = process the 3 default dirs)')
    parser.add_argument('--device', type=int, default=None, help='Override GPU device id')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (disable CUDA)')
    parser.add_argument('--dry-run', action='store_true', help='Just list experiments, do not modify')
    parser.add_argument('--force', action='store_true', help='Reprocess even if already fixed')
    parser.add_argument('--only', type=str, default=None,
                        help='Process only this specific experiment directory (full path)')
    args = parser.parse_args()

    if args.cpu:
        # Disable CUDA entirely
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.only:
        all_dirs = [args.only]
    elif args.root_dir:
        all_dirs = find_experiment_dirs(args.root_dir)
    else:
        all_dirs = []
        for root in DEFAULT_ROOTS:
            if os.path.isdir(root):
                all_dirs.extend(find_experiment_dirs(root))

    print(f"Found {len(all_dirs)} experiment directory(s) to process.")
    for d in all_dirs:
        print(f"  - {d}")

    if args.dry_run:
        print("\nDry run - exiting without modifications.")
        return

    ok, fail = 0, 0
    for d in all_dirs:
        try:
            if process_one_dir(d, device_override=args.device, force=args.force):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            import traceback
            print(f"  !! FAILED {d}: {e}")
            traceback.print_exc()
            fail += 1

    print(f"\nDone. Success: {ok}, Failed: {fail}")


if __name__ == '__main__':
    main()
