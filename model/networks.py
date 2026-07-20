# networks.py
import sys
import torch
import torch.nn as nn
from zjm.cadt.utils.data_argumentation import data_argumentation


class CNNEncoder(nn.Module):
    def __init__(self, t_size, x_dim=9, seq_len=128):
        super(CNNEncoder, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.seq_len = seq_len

        self.feature_10 = nn.Sequential(
            nn.Conv1d(in_channels=x_dim, out_channels=128, kernel_size=(7,), padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(5,), padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.feature_11 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(5,), padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=seq_len),          # was hardcoded 128
        )

        self.feature_20 = nn.Sequential(
            nn.Conv1d(in_channels=x_dim, out_channels=128, kernel_size=(7,), padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(21,), stride=1, padding=10),  # length-preserving for any seq_len, unchanged
        )

        self.feature_22 = nn.AvgPool1d(kernel_size=seq_len)          # was hardcoded 128
        self.feature_21 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(5,), padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=seq_len - 2)                    # was hardcoded 126 (=128-2)
        )
        self.linear1 = nn.Sequential(nn.Linear(128, 128))
        self.linear2 = nn.Sequential(nn.Linear(128, 128))

        self.fc11 = nn.Sequential(nn.Linear(128, t_size))
        self.fc21 = nn.Sequential(nn.Linear(128, t_size))

    def forward(self, z, with_argumentation=False, with_connection=True):
        data = z.cuda()
        data = data.transpose(1, 2)
        if with_argumentation:
            data = data_argumentation(data)
            data = data_argumentation(data)
        c = self.feature_10(data)
        d = self.feature_20(data)
        e = self.feature_22(d)

        if with_connection:
            d2 = d + torch.transpose(self.linear1(torch.transpose(c, 1, 2)), 1, 2)
            c2 = c + torch.transpose(self.linear2(torch.transpose(d, 1, 2)), 1, 2)
        else:
            d2 = d
            c2 = c

        data1 = self.feature_11(c2)
        data1 = data1.flatten(1, 2)
        f1 = self.fc11(data1)

        data2 = self.feature_21(d2)
        data2 = data2 + e
        data2 = data2.flatten(1, 2)
        f2 = self.fc21(data2)

        return f1, f2


class RegressionHead(nn.Module):
    """Replaces Classifier. Same architecture (one Linear layer), just no
    softmax reference (Classifier.forward() never called self.softmax anyway
    -- it returned raw logits -- so this is a rename, not a behavior change)."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        return self.linear(inputs)


class MultiDomainDiscriminator(nn.Module):
    """
    Replaces Discriminator(t_size + 1).
    Two structural changes from the original:
      1. Input width is t_size, not t_size+1 -- the '+1' was for the
         distance-to-nearest-class-center feature (zsd/ztd) which no longer
         exists (no discrete classes, no self.c).
      2. Output is n_domains logits, not a single sigmoid probability --
         there's no "source vs target" binary distinction anymore, only
         "which of the M source domains".
    """
    def __init__(self, input_size, n_domains, hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(input_size), hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, inputs):
        feature = inputs.view(inputs.size(0), -1)
        return self.model(feature)   # logits, not sigmoid -- use CrossEntropyLoss on these


class MultiDomainClassifier(nn.Module):
    """Replaces Discriminator(t_size) used as self.discriminator2.
    Same non-adversarial role (identify which domain zs came from), just
    M-way instead of binary."""
    def __init__(self, input_size, n_domains, hidden=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(input_size), hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, inputs):
        feature = inputs.view(inputs.size(0), -1)
        return self.model(feature)