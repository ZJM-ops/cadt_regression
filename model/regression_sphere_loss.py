import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionSphereLoss(nn.Module):
    """
    Regression-aware Hypersphere Loss for domain adaptation.
    
    Instead of discrete class assignment (classification), this loss uses soft assignment
    weights based on label distance for continuous labels.
    
    Loss formulation:
    L_sphere = Σ_{j=1}^k [ R_j^2 + σ * E_{x_s} [ w_j(y_s) * max(0, ||H_v(x_s) - c_j||^2 - R_j^2) ] ]
    
    Where:
    - w_j(y) = exp(-||y - c_j^y||^2 / τ) / Σ_{m=1}^k exp(-||y - c_j^y||^2 / τ)  (softmax-normalized RBF)
    - c_j: learnable center of hypersphere j in feature space
    - R_j: learnable radius of hypersphere j (enforced positive via exp)
    - c_j^y: anchor position in label space (fixed or learnable)
    - τ: temperature controlling softness of assignment
    - σ: penalty coefficient for violation
    - k: number of anchor spheres
    """
    
    def __init__(self, k: int, feat_dim: int, label_dim: int, 
                 tau: float = 1.0, sigma: float = 1.0, 
                 min_radius: float = 0.1, radius_reg_weight: float = 1.0):
        """
        Args:
            k: Number of anchor spheres
            feat_dim: Dimension of feature space (H_v output)
            label_dim: Dimension of label space (y)
            tau: Temperature for RBF kernel (higher = softer)
            sigma: Penalty coefficient for the hinge loss
            min_radius: Minimum radius to prevent collapse
            radius_reg_weight: Weight for the radius regularization term
        """
        super(RegressionSphereLoss, self).__init__()
        
        self.k = k
        self.feat_dim = feat_dim
        self.label_dim = label_dim
        self.tau = tau
        self.sigma = sigma
        self.min_radius = min_radius
        self.radius_reg_weight = radius_reg_weight
        
        # Learnable centers in feature space (k x feat_dim)
        self.centers = nn.Parameter(torch.randn(k, feat_dim))
        
        # Learnable radii in log-space (k,) - ensures positivity via exp
        # Initialize with reasonable values
        self.log_radii = nn.Parameter(torch.zeros(k))
        
        # Label space anchors (fixed during training, initialized separately)
        self.register_buffer('label_anchors', torch.randn(k, label_dim))
        
        # Initialize centers using Xavier uniform
        nn.init.xavier_uniform_(self.centers)
        
    def init_label_anchors(self, anchors: torch.Tensor):
        """
        Initialize label space anchors from external data (e.g., KMeans cluster centers).
        
        Args:
            anchors: Tensor of shape (k, label_dim) containing anchor positions in label space
        """
        assert anchors.shape == (self.k, self.label_dim), \
            f"Anchors shape {anchors.shape} doesn't match (k={self.k}, label_dim={self.label_dim})"
        self.label_anchors.copy_(anchors)
        
    def init_feature_centers(self, centers: torch.Tensor):
        """
        Initialize feature space centers from external data.
        
        Args:
            centers: Tensor of shape (k, feat_dim) containing initial centers
        """
        assert centers.shape == (self.k, self.feat_dim), \
            f"Centers shape {centers.shape} doesn't match (k={self.k}, feat_dim={self.feat_dim})"
        self.centers.data.copy_(centers)
        
    def compute_soft_weights(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignment weights w_j(y) using RBF kernel.
        
        Args:
            y: Batch of labels, shape (batch_size, label_dim)
            
        Returns:
            weights: Soft assignment weights, shape (batch_size, k)
        """
        # Compute squared distance from each sample to each label anchor
        # y: (B, D), label_anchors: (k, D) -> dist: (B, k)
        dist = torch.cdist(y, self.label_anchors, p=2) ** 2
        
        # RBF kernel with temperature
        # Use max subtraction for numerical stability (log-sum-exp trick)
        exp_terms = torch.exp(-dist / self.tau)
        
        # Normalize to sum to 1 (softmax)
        weights = exp_terms / (exp_terms.sum(dim=1, keepdim=True) + 1e-10)
        
        return weights
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the regression-aware hypersphere loss.
        
        Args:
            z: Feature embeddings from source domain, shape (batch_size, feat_dim)
            y: Labels for source domain samples, shape (batch_size, label_dim)
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = z.size(0)
        
        # Compute soft assignment weights
        weights = self.compute_soft_weights(y)  # (B, k)
        
        # Radii in positive space (k,)
        radii = torch.exp(self.log_radii)  # (k,)
        
        # Compute distance from each feature to each center
        # z: (B, D), centers: (k, D) -> feat_dist: (B, k)
        feat_dist_sq = torch.cdist(z, self.centers, p=2) ** 2  # (B, k)
        
        # Hinge loss for each sphere: max(0, ||z - c_j||^2 - R_j^2)
        # Shape: (B, k)
        hinge_loss = F.relu(feat_dist_sq - radii.unsqueeze(0) ** 2)
        
        # Weighted hinge loss: w_j(y) * hinge_j
        # Sum over spheres and average over batch
        weighted_hinge = (weights * hinge_loss).sum(dim=1).mean(dim=0)
        
        # Radius regularization: R_j^2 for each sphere
        radius_reg = (radii ** 2).sum()
        
        # Minimum radius penalty: prevent radii from collapsing to zero
        # max(0, min_radius - R_j)^2
        min_radius_penalty = F.relu(self.min_radius - radii).pow(2).sum()
        
        # Total loss
        # L_sphere = Σ_j [ R_j^2 + σ * E[w_j(y) * max(0, ||z - c_j||^2 - R_j^2)] ]
        # Plus minimum radius penalty
        loss = (self.radius_reg_weight * radius_reg + 
                self.sigma * weighted_hinge + 
                min_radius_penalty)
        
        return loss
    
    def extra_repr(self) -> str:
        return (f'k={self.k}, feat_dim={self.feat_dim}, label_dim={self.label_dim}, '
                f'tau={self.tau}, sigma={self.sigma}, min_radius={self.min_radius}')