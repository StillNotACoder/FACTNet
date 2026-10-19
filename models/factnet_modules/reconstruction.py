"""Spatial-frequency reconstruction and learned fusion."""

import torch
from torch import nn
from einops import rearrange
from .common import ResidualBlock


class SpatialFrequencyReconstruction(nn.Module):
    """Truncated amplitude-phase reconstruction and learned spatial fusion.

    Three sequences are concatenated and processed by residual convolutions;
    this is not the simple three-way summation described in the manuscript."""

    def __init__(self, input_shape, spec_num, input_dim, hidden_dim, aft_seq_length) -> None:
        super().__init__()
        h, w = input_shape
        self.aft_seq_length = aft_seq_length
        self.spec_num = spec_num
        spec_mask = torch.zeros(h, w // 2 + 1)
        spec_mask[..., :spec_num, :spec_num] = 1.0
        spec_mask[..., -spec_num:, :spec_num] = 1.0
        self.register_buffer("spec_mask", spec_mask)
        self.out_mixer = nn.Sequential(
            ResidualBlock(3 * input_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1),
        )

    def forward(self, xas, xps, phas):
        xas_fft = torch.fft.rfft2(xas)
        amps = torch.abs(xas_fft)
        alpha_fft = amps * self.spec_mask * torch.exp(torch.tensor(1j) * phas)
        alpha = torch.fft.irfft2(alpha_fft)
        xap = torch.cat([xas, xps, alpha], dim=2)
        xap = rearrange(xap, "b t c h w -> (b t) c h w")
        xt = self.out_mixer(xap)
        xt = rearrange(xt, "(b t) c h w -> b t c h w", t=self.aft_seq_length)
        return xt
