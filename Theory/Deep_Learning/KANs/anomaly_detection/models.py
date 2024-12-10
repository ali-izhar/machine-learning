"""Model implementations for KAN Anomaly Detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :].to(x.device)


class NaiveFourierKANLayer(nn.Module):
    """Implementation of a KAN layer using Fourier features."""

    def __init__(self, inputdim, outdim, gridsize=50, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2 * gridsize, inputdim, outdim)
            / (np.sqrt(inputdim) * np.sqrt(gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(outdim))

    def forward(self, x):
        batch_size, window_size, inputdim = x.size()
        k = torch.arange(1, self.gridsize + 1, device=x.device).float()
        k = k.view(1, 1, 1, self.gridsize)
        x_expanded = x.unsqueeze(-1)
        angles = x_expanded * k * np.pi
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        features = torch.cat([sin_features, cos_features], dim=-1)
        features = features.view(batch_size * window_size, inputdim, -1)
        y = torch.einsum("bik,kio->bo", features, self.fouriercoeffs)
        y = y.view(batch_size, window_size, self.outdim)
        if self.addbias:
            y += self.bias
        return y


class KAN(nn.Module):
    """Kolmogorov-Arnold Network for time series anomaly detection."""

    def __init__(
        self,
        in_feat,
        hidden_feat,
        out_feat,
        grid_feat,
        num_layers,
        use_bias=True,
        dropout=0.3,
    ):
        super(KAN, self).__init__()
        self.num_layers = num_layers
        self.positional_encoding = PositionalEncoding(hidden_feat)

        # Input layer
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.bn_in = nn.BatchNorm1d(hidden_feat)
        self.dropout = nn.Dropout(p=dropout)

        # KAN layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                NaiveFourierKANLayer(
                    hidden_feat, hidden_feat, grid_feat, addbias=use_bias
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_feat))

        # Output layer
        self.lin_out = nn.Linear(hidden_feat, out_feat, bias=use_bias)

    def forward(self, x):
        batch_size, window_size, _ = x.size()

        # Input transformation
        x = self.lin_in(x)
        x = self.bn_in(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
        x = F.leaky_relu(x, negative_slope=0.1)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # KAN layers
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)

        # Global pooling and output
        x = x.mean(dim=1)
        x = self.lin_out(x).squeeze()
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        return F_loss.mean()
