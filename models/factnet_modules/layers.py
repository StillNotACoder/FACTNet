"""Required layers retained verbatim from the supplied dependency."""

import torch
from torch import nn


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2 style)
    x: (B, C, H, W) -> y = x + gamma * (x / ||x||_2) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))  # learnable scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))  # learnable bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # L2 over spatial dims
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta


class _RepDWLite(nn.Module):
    """
    Lightweight re-parameterized depthwise composition:
    DW(1xK) -> DW(Kx1) + DW(3x3) + DW(1x1 ~ identity)
    """

    def __init__(self, dim: int, K: int, stride: int = 1):
        super().__init__()

        # separable large-kernel approximation
        self.dw_h = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, K),
            stride=(1, stride),
            padding=(0, K // 2),
            groups=dim,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=(K, 1),
            stride=(stride, 1),
            padding=(K // 2, 0),
            groups=dim,
            bias=False,
        )

        # complementary 3x3 + identity-like DW(1x1)
        self.dw_s = nn.Conv2d(
            dim, dim, kernel_size=3, stride=stride, padding=1, dilation=1, groups=dim, bias=False
        )
        self.dw_i = nn.Conv2d(dim, dim, kernel_size=1, stride=stride, groups=dim, bias=False)
        nn.init.dirac_(self.dw_i.weight)  # start as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_v(self.dw_h(x)) + self.dw_s(x) + self.dw_i(x)
