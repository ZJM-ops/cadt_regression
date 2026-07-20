import argparse
import torch
from torch.utils.data import DataLoader
from zjm.cadt.dataset.dataset import DailyEEG_data, MultiSourceBatchSampler, get_available_domains
from zjm.cadt.model.CADT import CATDModel

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--x_dim", type=int, default=59, help="EEG channel count")
parser.add_argument("--seq_len", type=int, default=2000, help="10s @ 200Hz")
parser.add_argument("--d_AE", type=int, default=128)
parser.add_argument("--n_targets", type=int, default=11, help="regression output dim (11 emotion labels)")
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256,
                     help="must satisfy batch_size // n_sources >= 1 for MultiSourceBatchSampler")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--sphere_loss_weight", type=float, default=10.0)
parser.add_argument("--discriminator_loss_weight", type=float, default=1.0)
parser.add_argument("--domain_cls_weight", type=float, default=1.0)
parser.add_argument("--contrast_tau", type=float, default=1.0)
parser.add_argument("--data_dir", type=str, default="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
parser.add_argument("--target_domain", type=int, default=0,
                     help="must be a domain id actually present in d.npy -- see get_available_domains()")
args = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.set_device(args.device)

    available_domains = get_available_domains(args.data_dir)
    if args.target_domain not in available_domains:
        raise ValueError(f"target_domain={args.target_domain} not present in data "
                          f"(available: {len(available_domains)} domains, "
                          f"range [{available_domains[0]}, {available_domains[-1]}], "
                          f"note some ids like 124 may be missing)")

    source_ids = [d for d in available_domains if d != args.target_domain]
    id_map = {raw: i for i, raw in enumerate(source_ids)}
    args.n_sources = len(source_ids)

    data_train = DailyEEG_data(source_ids, data_dir=args.data_dir, id_map=id_map)
    sampler = MultiSourceBatchSampler(data_train.d, list(id_map.values()), batch_size=args.batch_size)
    train_loader = DataLoader(data_train, batch_sampler=sampler)

    data_target = DailyEEG_data([args.target_domain], data_dir=args.data_dir)   # no id_map -- not used for domain-id losses
    valid_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = CATDModel(args).cuda()

    # --- shape sanity check before a full training run ---
    x_dummy = torch.randn(4, args.seq_len, args.x_dim).cuda()
    zsc_dummy, zs_dummy = model.share_encoder(x_dummy)
    print(f"Sanity check -- zsc: {zsc_dummy.shape}, zs: {zs_dummy.shape} (expect (4, 8) each)")

    best_mae, best_epoch = float('inf'), 0
    for epoch in range(args.n_epoch):
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
            )

        model.eval()
        sq_err_sum, abs_err_sum, n = 0.0, 0.0, 0
        for xt, yt, _ in valid_loader:
            xt = xt.to(torch.float32).cuda()
            yt = yt.to(torch.float32).cuda()
            pred = model.predict_target(xt)
            sq_err_sum += torch.sum((pred - yt) ** 2).item()
            abs_err_sum += torch.sum(torch.abs(pred - yt)).item()
            n += yt.numel()

        mse, mae = sq_err_sum / n, abs_err_sum / n
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
        print('Target %d, Epoch %d, Target MSE: %.6f, Target MAE: %.6f' %
              (args.target_domain, epoch, mse, mae))

    print('Target %d, Best MAE: %.6f at epoch %d' % (args.target_domain, best_mae, best_epoch))