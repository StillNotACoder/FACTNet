"""Motion modeling with the spatial guidance found in the supplied source."""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from timm.layers import DropPath
from .common import SpatialEncoder, SpatialDecoder
from .layers import GRN, _RepDWLite


class MSCM(nn.Module):
    """Multi-scale context initialization with projected, concatenated branches.

    A remainder branch preserves output channel counts not divisible by the
    number of scales. The underlying reparameterizable block is a dependency."""

    def __init__(
        self, in_ch: int, out_ch: int, k_list=(3, 5, 7), stride: int = 1, use_gn: bool = True
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    _RepDWLite(in_ch, K=k, stride=stride),
                    nn.Conv2d(in_ch, out_ch // len(k_list), 1, bias=False),
                )
                for k in k_list
            ]
        )
        gap = out_ch - out_ch // len(k_list) * len(k_list)
        self.tail = (
            nn.Identity()
            if gap == 0
            else nn.Sequential(
                _RepDWLite(in_ch, K=k_list[0], stride=stride), nn.Conv2d(in_ch, gap, 1, bias=False)
            )
        )
        self.fuse = nn.Identity()
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches]
        if not isinstance(self.tail, nn.Identity):
            parts.append(self.tail(x))
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))


class MSK(nn.Module):
    """One separable multi-scale kernel branch with channel-wise tanh center suppression."""

    def __init__(self, dim: int, K: int, center_suppress: bool = True):
        super().__init__()
        self.center_suppress = center_suppress
        self.dw_h = nn.Conv2d(
            dim, dim, kernel_size=(1, K), padding=(0, K // 2), groups=dim, bias=False
        )
        self.dw_v = nn.Conv2d(
            dim, dim, kernel_size=(K, 1), padding=(K // 2, 0), groups=dim, bias=False
        )
        if self.center_suppress:
            self.dw_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.register_parameter("beta", None)
            self.dw_c = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw_v(self.dw_h(x))
        if self.center_suppress:
            center = self.dw_c(x)
            y = y - torch.tanh(self.beta) * center
        return y


class SpatialGuidanceUnit(nn.Module):
    """Fixed per-channel LoG, Scharr, and Sobel sigmoid response maps.

    FGA evaluates these maps without gradient tracking, matching the source."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
        scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        gauss_1d = torch.tensor([0.0545, 0.2442, 0.4026, 0.2442, 0.0545])
        gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("scharr_x", scharr_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("scharr_y", scharr_y.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("laplace", laplace.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("gauss5x5", gauss_2d.view(1, 1, 5, 5), persistent=False)

    def _depthwise_filter(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """通用的 Depthwise 物理算子运算器"""
        B, C, H, W = x.shape
        w = k.repeat(C, 1, 1, 1)
        pad = k.shape[-1] // 2
        return F.conv2d(x, w, padding=pad, groups=C)

    def forward(self, x: torch.Tensor):
        x_blur = self._depthwise_filter(x, self.gauss5x5)
        lap = self._depthwise_filter(x_blur, self.laplace)
        gate_micro = torch.sigmoid(lap.abs())
        sch_x = self._depthwise_filter(x, self.scharr_x)
        sch_y = self._depthwise_filter(x, self.scharr_y)
        gate_meso = torch.sigmoid(torch.sqrt(sch_x.pow(2) + sch_y.pow(2) + 1e-06))
        sx = self._depthwise_filter(x, self.sobel_x)
        sy = self._depthwise_filter(x, self.sobel_y)
        gate_macro = torch.sigmoid(torch.sqrt(sx.pow(2) + sy.pow(2) + 1e-06))
        return [gate_micro, gate_meso, gate_macro]


class FGA(nn.Module):
    """Multi-scale kernel responses fused using fixed spatial guidance.

    The supplied implementation uses LoG, Scharr, and Sobel sigmoid gates,
    not the manuscript's Fourier high-pass and softmax routing formulation."""

    def __init__(
        self, dim: int, K_list=(9, 15, 31), use_grn: bool = False, center_suppress: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.K_list = sorted(K_list)
        self.branches = nn.ModuleList(
            [MSK(dim, K, center_suppress=center_suppress) for K in self.K_list]
        )
        self.guidance = SpatialGuidanceUnit()
        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        peris = [b(x) for b in self.branches]
        with torch.no_grad():
            gates = self.guidance(x)
        Y = 0.0
        for y, g in zip(peris, gates):
            Y = Y + y * g
        if self.use_grn:
            Y = self.grn(Y)
        return Y

    Branch = MSK


class MotionEvolutionBlock(nn.Module):
    """Residual FGA token mixing and gated depthwise channel mixing.

    Retains the supplied normalization, GRN, LayerScale, and DropPath operations."""

    def __init__(
        self,
        dim: int,
        groups_pw: int = 1,
        layerscale_init: float = 1e-06,
        act_layer=nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        pfga_K=(9, 15, 31),
        mlp_ratio: float = 4.0,
        dw_kernel: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.norm_dw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.norm_pw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.tm = FGA(dim, K_list=pfga_K, use_grn=False)
        self.grn_dw = GRN(dim)
        self.grn_pw = GRN(dim)
        self.mlp_ratio = mlp_ratio
        self.dw_kernel = dw_kernel
        E = max(dim, int(dim * self.mlp_ratio))
        self.pw_in = nn.Conv2d(dim, 2 * E, kernel_size=1, bias=True, groups=groups_pw)
        self.dw_v = nn.Conv2d(E, E, kernel_size=self.dw_kernel, padding=1, groups=E, bias=False)
        self.pw_out = nn.Conv2d(E, dim, kernel_size=1, bias=True, groups=groups_pw)
        self.act = act_layer()
        self.gamma_dw = nn.Parameter(torch.ones(dim) * layerscale_init)
        self.gamma_pw = nn.Parameter(torch.ones(dim) * layerscale_init)
        self.dropout_dw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.dropout_pw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self._init_params()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"gamma_dw", "gamma_pw"}

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm_dw(x)
        y = self.tm(y)
        y = self.act(y)
        y = self.grn_dw(y)
        y = self.dropout_dw(y)
        x = x + self.drop_path(y * self.gamma_dw.view(1, self.dim, 1, 1))
        z = self.norm_pw(x)
        uv = self.pw_in(z)
        u, v = torch.chunk(uv, 2, dim=1)
        v = self.dw_v(v)
        z = F.silu(u) * v
        z = self.pw_out(z)
        z = self.grn_pw(z)
        z = self.dropout_pw(z)
        x = x + self.drop_path(z * self.gamma_pw.view(1, self.dim, 1, 1))
        return x


class MotionEvolutionStage(nn.Module):
    """MSCM followed by stacked motion evolution blocks."""

    def __init__(
        self,
        in_ch: int,
        depth: int,
        groups_pw: int = 1,
        layerscale_init: float = 1e-06,
        cel_k=(3, 5, 7),
        drop: float = 0.0,
        drop_path: float = 0.0,
        pfga_K=(9, 15, 31),
    ):
        super().__init__()
        self.cel = MSCM(in_ch, in_ch, k_list=cel_k, use_gn=True)
        if drop_path > 0 and depth > 0:
            dpr = [x.item() for x in torch.linspace(0.01, drop_path, depth)]
        else:
            dpr = [0.0] * depth
        self.blocks = nn.Sequential(
            *[
                MotionEvolutionBlock(
                    in_ch,
                    groups_pw=groups_pw,
                    layerscale_init=layerscale_init,
                    act_layer=nn.GELU,
                    drop=drop,
                    drop_path=drop_path if i == depth - 1 else dpr[i],
                    pfga_K=pfga_K,
                )
                for i in range(depth)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.view(b, t * c, h, w)
        x = self.cel(x)
        x = self.blocks(x)
        return x.view(b, t, c, h, w)


class MDNet(nn.Module):
    """Encoder-evolution-decoder motion branch with phase from predicted frames.

    No historical-phase addition is performed by this implementation."""

    def __init__(
        self,
        input_shape,
        pre_seq_length,
        aft_seq_length,
        input_dim,
        hidden_dim,
        n_layers,
        kernel_size,
        bias=1,
    ):
        super().__init__()
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        n_s = 2
        self.enc = SpatialEncoder(
            c_in=input_dim * pre_seq_length, c_hid=hidden_dim, n_s=n_s, k=kernel_size
        )
        self.mid_pfg = MotionEvolutionStage(in_ch=hidden_dim, depth=n_layers, groups_pw=1)
        self.dec = SpatialDecoder(
            c_hid=hidden_dim, c_out=input_dim * aft_seq_length, n_s=n_s, k=kernel_size
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> b (t c) h w")
        z, skip = self.enc(x_flat)
        z_pfg_in = rearrange(z, "b c h w -> b 1 c h w")
        z_pfg_out = self.mid_pfg(z_pfg_in)
        z_pfg_flat = rearrange(z_pfg_out, "b 1 c h w -> b c h w")
        xt_flat = self.dec(z_pfg_flat, skip)
        xt = rearrange(xt_flat, "b (t c) h w -> b t c h w", t=self.aft_seq_length)
        xt_fft = torch.fft.rfft2(xt)
        x_phas_t = torch.angle(xt_fft)
        x_amps = torch.abs(xt_fft)
        return (xt, x_phas_t, x_amps)
