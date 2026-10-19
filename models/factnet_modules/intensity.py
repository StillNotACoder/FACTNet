"""Intensity evolution modules adapted from the supplied implementation."""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .common import ResidualBlock


class LocalSpatialBranch(nn.Module):
    """Per-frame 3 x 3 convolution, group normalization, and SiLU."""

    def __init__(self, in_channels, out_channels):
        super(LocalSpatialBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.conv(x)
        x = x.view(B, T, -1, H, W)
        return x


class FreMLP(nn.Module):
    """Two complex temporal linear maps in Fourier space, plus a temporal residual.

    The real and imaginary components use separate ReLU activations. This
    module is not an amplitude-only AE block; it can change spectral phase."""

    def __init__(self, t_in, t_out, size_factor=1):
        super().__init__()
        self.t_in, self.t_out = (t_in, t_out)
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, t_in, t_out * size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, 1, 1, 1, t_out * size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, t_out * size_factor, t_out))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, 1, 1, 1, t_out))

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        bias = self.tmlp(x)
        xf = torch.fft.rfft2(x, dim=[2, 3], norm="ortho")
        x1_real = (
            torch.einsum("bchwt,to->bchwo", xf.real, self.w1[0])
            - torch.einsum("bchwt,to->bchwo", xf.imag, self.w1[1])
            + self.b1[0]
        )
        x1_imag = (
            torch.einsum("bchwt,to->bchwo", xf.real, self.w1[1])
            + torch.einsum("bchwt,to->bchwo", xf.imag, self.w1[0])
            + self.b1[1]
        )
        x1_real, x1_imag = (F.relu(x1_real), F.relu(x1_imag))
        x2_real = (
            torch.einsum("bchwt,to->bchwo", x1_real, self.w2[0])
            - torch.einsum("bchwt,to->bchwo", x1_imag, self.w2[1])
            + self.b2[0]
        )
        x2_imag = (
            torch.einsum("bchwt,to->bchwo", x1_real, self.w2[1])
            + torch.einsum("bchwt,to->bchwo", x1_imag, self.w2[0])
            + self.b2[1]
        )
        x2 = torch.view_as_complex(torch.stack([x2_real, x2_imag], dim=-1))
        x = torch.fft.irfft2(x2, dim=[2, 3], norm="ortho")
        x = x + bias
        return x.permute(0, 4, 1, 2, 3)


class IEBlock(nn.Module):
    """Fourier temporal mixing followed by temporal and spatial residual paths."""

    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.t_in, self.t_out = (t_in, t_out)
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.amptime = FreMLP(t_in, t_out)
        self.conv = nn.Sequential(
            nn.Conv2d(dim * t_out, dim * t_out, kernel_size=3, padding=1),
            nn.GroupNorm(4, dim * t_out),
            nn.SiLU(),
            nn.Conv2d(dim * t_out, dim * t_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        residual = self.tmlp(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.amptime(x)
        x = x + residual
        residual = x
        x = rearrange(x, "b t c h w -> b (t c) h w")
        x = self.conv(x)
        x = rearrange(x, "b (t c) h w -> b t c h w", t=self.t_out)
        x = x + residual
        return x


class IENet(nn.Module):
    """Intensity branch with interacting Fourier and local spatial streams.

    Legacy child attribute names are retained for state-dict compatibility."""

    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=3, mlp_ratio=2):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = (pre_seq_length, aft_seq_length)
        self.dim, self.hidden_dim = (dim, hidden_dim)
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, int(aft_seq_length * mlp_ratio)),
            nn.SELU(True),
            nn.Linear(int(aft_seq_length * mlp_ratio), aft_seq_length),
        )
        self.convin = nn.Sequential(
            ResidualBlock(dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.interactions = n_layers
        self.amplist = nn.ModuleList(
            [
                IEBlock(pre_seq_length if i == 0 else aft_seq_length, aft_seq_length, hidden_dim)
                for i in range(self.interactions)
            ]
        )
        self.locallist = nn.ModuleList(
            [LocalSpatialBranch(hidden_dim, hidden_dim) for i in range(self.interactions)]
        )
        self.local_time_align = nn.Linear(pre_seq_length, aft_seq_length)
        self.fusion_list = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1), nn.SiLU())
                for _ in range(self.interactions)
            ]
        )
        self.convout = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.convin(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=self.pre_seq_length)
        x_ = x.permute(0, 2, 3, 4, 1)
        xr = self.tmlp(x_)
        xr = rearrange(xr, "b c h w t -> (b t) c h w")
        amp_feat = x
        local_feat = x
        for i in range(self.interactions):
            a_out = self.amplist[i](amp_feat)
            l_out = self.locallist[i](local_feat)
            if i == 0:
                l_out = l_out.permute(0, 2, 3, 4, 1)
                l_out = self.local_time_align(l_out)
                l_out = l_out.permute(0, 4, 1, 2, 3)
            B, T, C, H, W = a_out.shape
            cat_feat = torch.cat([a_out, l_out], dim=2)
            cat_feat = cat_feat.reshape(-1, C * 2, H, W)
            fused = self.fusion_list[i](cat_feat).view(B, T, C, H, W)
            amp_feat = a_out + fused
            local_feat = l_out + fused
        x_out = xr + rearrange(amp_feat, "b t c h w -> (b t) c h w")
        x_out = self.convout(x_out)
        x_out = rearrange(x_out, "(b t) c h w -> b t c h w", t=self.aft_seq_length)
        return x_out
