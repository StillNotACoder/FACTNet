"""Convolutional building blocks used by both FACTNet branches."""

from torch import nn
import torch


def sampling_generator(N: int, reverse: bool = False):
    samplings = [False, True] * (N // 2)
    samplings = samplings[:N]
    return list(reversed(samplings)) if reverse else samplings


class SpatialConvBlock(nn.Module):
    """Spatial convolution with group normalization and LeakyReLU."""

    def __init__(
        self, c_in, c_out, kernel_size=3, downsampling=False, upsampling=False, act_inplace=True
    ):
        super().__init__()
        padding = kernel_size // 2
        if upsampling:
            self.conv = nn.ConvTranspose2d(
                c_in, c_out, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1
            )
        else:
            stride = 2 if downsampling else 1
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding
            )
        num_groups = 2 if c_out % 2 == 0 else 1
        self.norm = nn.GroupNorm(num_groups, c_out)
        self.act = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SpatialEncoder(nn.Module):
    """Spatial encoder with alternating full-resolution and downsampling blocks."""

    def __init__(self, c_in: int, c_hid: int, n_s: int, k: int, act_inplace: bool = False):
        super().__init__()
        samplings = sampling_generator(n_s)
        self.enc = nn.Sequential(
            SpatialConvBlock(c_in, c_hid, k, downsampling=samplings[0], act_inplace=act_inplace),
            *[
                SpatialConvBlock(c_hid, c_hid, k, downsampling=s, act_inplace=act_inplace)
                for s in samplings[1:]
            ],
        )

    def forward(self, x: torch.Tensor):
        x0 = self.enc[0](x)
        z = x0
        for i in range(1, len(self.enc)):
            z = self.enc[i](z)
        return (z, x0)


class SpatialDecoder(nn.Module):
    """Spatial decoder with a residual skip connection."""

    def __init__(self, c_hid: int, c_out: int, n_s: int, k: int, act_inplace: bool = False):
        super().__init__()
        samplings = sampling_generator(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[
                SpatialConvBlock(c_hid, c_hid, k, upsampling=s, act_inplace=act_inplace)
                for s in samplings[:-1]
            ],
            SpatialConvBlock(c_hid, c_hid, k, upsampling=samplings[-1], act_inplace=act_inplace),
        )
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1)

    def forward(self, z: torch.Tensor, skip: torch.Tensor):
        for i in range(len(self.dec) - 1):
            z = self.dec[i](z)
        y = self.dec[-1](z + skip)
        return self.readout(y)


class ConvNormActivation(nn.Module):
    """Convolution, normalization, and SiLU building block."""

    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode="zeros", groupnorm=True):
        super(ConvNormActivation, self).__init__()
        self.proj = nn.Conv2d(
            dim,
            dim_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
        )
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """Two convolutional blocks with a projection skip when needed."""

    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode="zeros"):
        super().__init__()
        self.block1 = ConvNormActivation(
            dim, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode
        )
        self.block2 = ConvNormActivation(
            dim_out, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)
