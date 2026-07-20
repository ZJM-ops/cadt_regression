# CADT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from zjm.cadt.model.networks import CNNEncoder, RegressionHead, MultiDomainDiscriminator, MultiDomainClassifier
from zjm.cadt.model.losses import RegressionAwareContrastLoss


class CATDModel(nn.Module):
    def __init__(self, args):
        super(CATDModel, self).__init__()
        self.args = args
        self.input_size = args.x_dim
        self.hidden_size = args.d_AE
        self.y_dims = args.n_targets       # was n_class (discrete); now regression output dim (11)
        self.n_sources = args.n_sources    # NEW: number of source domains this run (149 in your case)

        t_size = 8
        self.tsize = t_size

        self.share_encoder = CNNEncoder(t_size, x_dim=args.x_dim, seq_len=args.seq_len)
        self.regressor = RegressionHead(t_size, self.y_dims)                    # was self.classifier
        self.discriminator = MultiDomainDiscriminator(t_size, self.n_sources)   # was Discriminator(t_size+1), binary
        self.domain_classifier = MultiDomainClassifier(t_size, self.n_sources)  # was self.discriminator2, binary

        lr = 1e-3
        self.share_encoder_optimizer = torch.optim.AdamW(self.share_encoder.parameters(), lr=lr)
        self.regressor_optimizer = torch.optim.AdamW(self.regressor.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        self.domain_classifier_optimizer = torch.optim.AdamW(self.domain_classifier.parameters(), lr=lr)

        self.reg_loss_fn = nn.SmoothL1Loss(reduction='mean')  # Huber; swap to nn.MSELoss() if labels are clean
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrast_loss_fn = RegressionAwareContrastLoss(tau=getattr(args, 'contrast_tau', 1.0))

        # REMOVED vs. original:
        #   self.a           (one-hot eye matrix for class labels -- no discrete classes)
        #   self.c            (per-class hypersphere centers -- nothing to pool by class anymore)
        #   self.bce_loss     (was for binary source/target discrimination)
        #   reset()           (only existed to reinitialize after init_center_c's warm-up phase)
        #   init_center_c()   (computed self.c from labeled data -- no longer applicable)

    def train_step(self, x_src, y_src, dom_ids, sphere_w=1.0, dis_w=1.0, dom_w=1.0,
                    with_argumentation=True):
        """
        x_src: (B, seq_len, x_dim) -- SOURCE DOMAINS ONLY. The original x_real
               (target) argument is gone entirely -- target never enters training.
        y_src: (B, n_targets) continuous regression labels
        dom_ids: (B,) long tensor of remapped source-domain ids in [0, n_sources)
        """
        zsc, zs = self.share_encoder(x_src, with_argumentation=False)

        # --- Task loss: was ce_loss on Classifier; now regression loss ---
        reg_loss = self.reg_loss_fn(self.regressor(zsc), y_src)

        # --- Hypersphere replacement: was kl_loss against self.c[y_fake] ---
        sphere_loss = self.contrast_loss_fn(zsc, y_src)

        # --- Domain classifier on domain-specific features zs ---
        # Was: self.discriminator2 with binary bce_loss (source=0, target=1).
        # Now: M-way cross-entropy identifying which source domain zs came from.
        # Same non-adversarial role: encoder + domain_classifier co-optimize.
        dom_logits = self.domain_classifier(zs)
        domain_class_loss = self.ce_loss(dom_logits, dom_ids)

        if with_argumentation:
            _, zs2 = self.share_encoder(x_src, with_argumentation=True)
            dom_logits2 = self.domain_classifier(zs2)
            domain_class_loss2 = self.ce_loss(dom_logits2, dom_ids)
        else:
            domain_class_loss2 = torch.tensor(0.0, device=x_src.device)

        # --- Adversarial discriminator on domain-invariant features zsc ---
        # Was: BCE with source=fake_label(0), target=real_label(1), flipped
        # for the encoder's adversarial objective.
        # Now: with M sources and no target in the batch, "fool the
        # discriminator" becomes "make its output indistinguishable from
        # uniform over the M sources" -- push predicted distribution toward
        # uniform via KL divergence (domain-confusion loss, Tzeng et al. 2015).
        disc_logits = self.discriminator(zsc)
        log_probs = F.log_softmax(disc_logits, dim=1)
        uniform_target = torch.full_like(log_probs, 1.0 / self.n_sources)
        confusion_loss = F.kl_div(log_probs, uniform_target, reduction='batchmean')

        total_loss = (reg_loss
                      + sphere_w * sphere_loss
                      + dis_w * confusion_loss
                      + dom_w * (domain_class_loss + domain_class_loss2))

        self.share_encoder_optimizer.zero_grad()
        self.regressor_optimizer.zero_grad()
        self.domain_classifier_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.share_encoder.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=0.5)
        self.share_encoder_optimizer.step()
        self.regressor_optimizer.step()
        self.domain_classifier_optimizer.step()

        # --- Discriminator's own update: real objective, detached features ---
        # Mirrors the original's second .backward() pass, which trained
        # self.discriminator on the true source/target labels using detached
        # features (so this update doesn't also push the encoder).
        disc_logits2 = self.discriminator(zsc.detach())
        disc_loss = self.ce_loss(disc_logits2, dom_ids)
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        self.discriminator_optimizer.step()

        return {
            'reg_loss': reg_loss.item(),
            'sphere_loss': sphere_loss.item(),
            'confusion_loss': confusion_loss.item(),
            'domain_class_loss': (domain_class_loss + domain_class_loss2).item(),
            'disc_loss': disc_loss.item(),
        }

    def compute_loss(self, x, y):
        """
        Was: CE loss + argmax classification accuracy (correct/total).
        Now: sum of squared error, sum of absolute error, and element count --
        so main.py can aggregate MSE/MAE across batches the same way it
        aggregated correct/total before.
        """
        with torch.no_grad():
            zsc, zs = self.share_encoder(x)
            pred = self.regressor(zsc)
            sq_err = torch.sum((pred - y) ** 2).item()
            abs_err = torch.sum(torch.abs(pred - y)).item()
            total = y.numel()
        return sq_err, abs_err, total

    def compute_loss2(self, x, true_dom_ids):
        """Domain-classification eval accuracy, generalized from binary
        threshold-at-0.5 to M-way argmax."""
        with torch.no_grad():
            zsc, zs = self.share_encoder(x, with_argumentation=True)
            dom_logits = self.domain_classifier(zs)
            pred = torch.argmax(dom_logits, dim=1)
            total = true_dom_ids.size(0)
            correct = pred.eq(true_dom_ids).cpu().sum()
        return 0.0, correct, total

    @torch.no_grad()
    def predict_target(self, x_target):
        """
        Target-domain inference only. share_encoder structurally always
        computes both zsc and zs together (because of the linear1/linear2
        cross-branch interaction inside forward()) -- they can't be cleanly
        separated within one forward pass without restructuring the encoder.
        zs is discarded here and never passed to domain_classifier or
        discriminator, at train OR test time, satisfying the requirement
        that target skips the discriminator/domain-classifier entirely.
        """
        self.eval()
        zsc, zs = self.share_encoder(x_target, with_argumentation=False)
        return self.regressor(zsc)