# model/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionAwareContrastLoss(nn.Module):
    """
    Replaces the discrete class-center hypersphere mechanism from the
    original CADT.py (self.a, self.c, init_center_c(), the kl_loss computed
    against py = self.c[y_fake], and the zsd/ztd min-distance-to-center
    features). None of that machinery applies to continuous regression
    targets -- there's no "class" to build a center from.

    Mechanism: pull together zsc features whose regression labels y are
    close, in proportion to how close; pairs with very different labels
    get near-zero attraction (no explicit repulsion term here, kept simple
    to match the original's single kl_loss term -- see note below on
    optionally adding repulsion).
    """
    def __init__(self, tau: float = 1.0, normalize: bool = True):
        super().__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, zsc: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(1)
        N = zsc.size(0)
        if N < 2:
            return zsc.new_zeros(())

        feats = F.normalize(zsc, dim=1) if self.normalize else zsc
        feat_dist = torch.cdist(feats, feats, p=2) ** 2      # (N, N)
        label_dist = torch.cdist(y, y, p=2)                   # (N, N), works for multi-dim y (e.g. your 11 labels)
        weight = torch.exp(-(label_dist ** 2) / (2 * self.tau ** 2))

        mask = 1.0 - torch.eye(N, device=zsc.device)
        return (weight * feat_dist * mask).sum() / mask.sum().clamp(min=1)