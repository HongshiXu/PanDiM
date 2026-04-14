import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from inspect import isfunction
from timm.models.vision_transformer import Mlp


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    padding = (3 - 1) // 2 * dilation
    return nn.Sequential(
        nn.ReplicationPad2d(padding),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
        ),
    )


class TimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        return torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_patch, hidden_size):
        super().__init__()
        self.num_patch = num_patch
        self.hidden_size = hidden_size

    def forward(self, level):
        position = torch.arange(self.num_patch, dtype=torch.float32, device=level.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32, device=level.device)
            * (-math.log(10000.0) / self.hidden_size)
        )
        pe = torch.zeros(self.num_patch, self.hidden_size, device=level.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) * level


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DiscrepancyCommonInjctionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, num_heads, mode):
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads
        hidden_dim = in_channels * patch_size * patch_size
        self.position_encoder = PositionalEncoding(
            num_patch=(img_size // patch_size) * (img_size // patch_size),
            hidden_size=hidden_dim,
        )
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LayerNorm([out_channels, img_size, img_size], eps=1e-6, elementwise_affine=False),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.depth = hidden_dim // num_heads
        self.seg = Rearrange("b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)", s1=patch_size, s2=patch_size)
        self.comb = Rearrange(
            "b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
            s1=patch_size,
            s2=patch_size,
        )

    def forward(self, k, q, v):
        b, n, e = k.shape
        scale = e ** -0.5
        position_emb = repeat(self.position_encoder(self.p), "1 n e -> b n e", b=b)
        k, q, v = k + position_emb, q + position_emb, v + position_emb
        q = q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        t = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
        t = t.transpose(1, 2).reshape(b, n, e)
        v = v.transpose(1, 2).reshape(b, n, e)
        q = q.transpose(1, 2).reshape(b, n, e)
        v = v - t if self.mode == "discrepancy" else t
        q = self.linear(v) + q
        q = self.comb(q)
        q = self.body(q)
        return self.seg(q)


class ModalAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, num_heads):
        super().__init__()
        self.diim = DiscrepancyCommonInjctionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            mode="discrepancy",
        )
        self.aciim1 = DiscrepancyCommonInjctionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            mode="common",
        )
        self.aciim2 = DiscrepancyCommonInjctionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            mode="common",
        )
        self.reg = Rearrange("b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)", s1=patch_size, s2=patch_size)
        self.comb = Rearrange(
            "b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
            s1=patch_size,
            s2=patch_size,
        )

    def forward(self, ms, pan):
        pan_k, pan_q, pan_v = map(self.reg, pan.chunk(3, dim=1))
        ms_k, ms_q, ms_v = map(self.reg, ms.chunk(3, dim=1))
        q1 = self.diim(pan_k, pan_v, ms_q)
        q2 = self.aciim1(ms_k, ms_v, q1)
        q2 = self.aciim2(pan_k, pan_v, q2)
        return self.comb(q1 + q2)


class CondBlock(nn.Module):
    def __init__(self, cond_dim, hidden_dim, patch_size, num_chunk, groups):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 8, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 8),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 8, hidden_dim * num_chunk, 1, bias=True),
        )
        cond_size = cond_dim * patch_size * patch_size
        hidden_size = hidden_dim * patch_size * patch_size
        self.global_linear = nn.Linear(cond_size, hidden_size)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.reg = Rearrange("b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)", s1=patch_size, s2=patch_size)

    def forward(self, cond):
        out = self.body(cond)
        global_token = self.global_linear(self.reg(cond)).mean(dim=1)
        global_token = self.mlp(global_token)
        return out, global_token


class ConvLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, img_size, need_residual=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReplicationPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False),
            nn.SiLU(),
            conv3x3(hidden_channels, out_channels),
        )
        if need_residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        if hasattr(self, "res_conv"):
            return self.body(x) + self.res_conv(x)
        return self.body(x)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class WaveletBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.num_heads = num_heads
        self.ms_conv = ConvLayer(
            in_channels=ms_channels * 2,
            hidden_channels=hidden_dim * 4,
            out_channels=hidden_dim * 6,
            kernel_size=3,
            img_size=img_size,
            need_residual=True,
        )
        self.pan_scale_conv = nn.Sequential(
            conv3x3(pan_channels, hidden_dim * 4),
            nn.LeakyReLU(0.1),
            conv3x3(hidden_dim * 4, hidden_dim * 6),
        )
        self.pan_shift_conv = nn.Sequential(
            conv3x3(pan_channels, hidden_dim * 4),
            nn.LeakyReLU(0.1),
            conv3x3(hidden_dim * 4, hidden_dim * 6),
        )
        self.pan_body = ConvLayer(
            in_channels=pan_channels * 3,
            hidden_channels=hidden_dim * 4,
            out_channels=hidden_dim * 6,
            kernel_size=3,
            img_size=img_size,
            need_residual=True,
        )
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            patch_size=patch_size,
            img_size=img_size,
            num_heads=num_heads * 2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_chunk=num_chunks,
            groups=hidden_dim,
        )
        self.hf_compress = nn.Conv2d(hidden_dim * 6, hidden_dim, kernel_size=1)

    def forward(self, cond):
        ms = torch.cat(
            [
                cond[:, : self.ms_channels],
                cond[:, self.ms_channels + self.pan_channels : self.ms_channels * 2 + self.pan_channels],
            ],
            dim=1,
        )
        ms = self.ms_conv(ms)
        pan = cond[:, self.ms_channels : self.ms_channels + self.pan_channels]
        pan_wavelet = cond[:, self.ms_channels * 2 + self.pan_channels :]
        pan_scale = self.pan_scale_conv(pan)
        pan_shift = self.pan_shift_conv(pan)
        pan_hf_feat = modulate(self.pan_body(pan_wavelet), pan_shift, pan_scale)
        fusion = self.fusion_net(ms, pan_hf_feat)
        cond_base, global_token = self.cond_inj(fusion)
        cond_hf = self.hf_compress(pan_hf_feat)
        return cond_base, cond_hf, global_token


class PyramidalFuseBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ms_conv = nn.Sequential(
            conv3x3(in_channels, in_channels * 2),
            nn.LeakyReLU(0.1),
            conv3x3(in_channels * 2, in_channels * 2),
        )
        self.pan_conv = nn.Sequential(
            conv3x3(in_channels, in_channels * 2),
            nn.LeakyReLU(0.1),
            conv3x3(in_channels * 2, in_channels * 2),
        )

    def forward(self, ms, pan):
        ms_scale, ms_shift = self.ms_conv(ms).chunk(2, dim=1)
        pan_scale, pan_shift = self.pan_conv(pan).chunk(2, dim=1)
        ms = modulate(ms, ms_shift, ms_scale)
        pan = modulate(pan, pan_shift, pan_scale)
        return ms, pan


class PyramidalUpConvs(nn.Module):
    def __init__(self, in_channels, n_levels, img_size):
        super().__init__()
        self.up_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels // (2 ** i),
                        in_channels // (2 ** (i + 1)),
                        4,
                        2,
                        1,
                    ),
                    nn.LeakyReLU(0.1),
                    nn.LayerNorm(
                        [
                            in_channels // (2 ** (i + 1)),
                            img_size // (2 ** (n_levels - 1 - i)),
                            img_size // (2 ** (n_levels - 1 - i)),
                        ],
                        eps=1e-6,
                        elementwise_affine=False,
                    ),
                )
                for i in range(n_levels)
            ]
        )

    def forward(self, x):
        for up_conv in self.up_convs:
            x = up_conv(x)
        return x


class PyramidalSpatialBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(
            in_channels=ms_channels,
            hidden_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=3,
            img_size=img_size,
            need_residual=False,
        )
        self.pan_conv = ConvLayer(
            in_channels=pan_channels,
            hidden_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=3,
            img_size=img_size,
            need_residual=False,
        )
        self.ms_pan_fuse = PyramidalFuseBlock(hidden_dim * 2)
        self.ms_dw_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim * (2 ** (i + 1)), hidden_dim * (2 ** (i + 2)), 3, 2, 1),
                    nn.LeakyReLU(0.1),
                    nn.LayerNorm(
                        [
                            hidden_dim * (2 ** (i + 2)),
                            img_size // (2 ** (i + 1)),
                            img_size // (2 ** (i + 1)),
                        ],
                        eps=1e-6,
                        elementwise_affine=False,
                    ),
                )
                for i in range(2)
            ]
        )
        self.pan_dw_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim * (2 ** (i + 1)), hidden_dim * (2 ** (i + 2)), 3, 2, 1),
                    nn.LeakyReLU(0.1),
                    nn.LayerNorm(
                        [
                            hidden_dim * (2 ** (i + 2)),
                            img_size // (2 ** (i + 1)),
                            img_size // (2 ** (i + 1)),
                        ],
                        eps=1e-6,
                        elementwise_affine=False,
                    ),
                )
                for i in range(2)
            ]
        )
        self.fuse_nets = nn.ModuleList([PyramidalFuseBlock(hidden_dim * (2 ** (i + 2))) for i in range(2)])
        self.ms_up_convs = nn.ModuleList([PyramidalUpConvs(hidden_dim * (2 ** (i + 2)), i + 1, img_size) for i in range(2)])
        self.pan_up_convs = nn.ModuleList([PyramidalUpConvs(hidden_dim * (2 ** (i + 2)), i + 1, img_size) for i in range(2)])
        self.ms_refine_conv = nn.Conv2d(hidden_dim * 6, hidden_dim * 6, 1)
        self.pan_refine_conv = nn.Conv2d(hidden_dim * 6, hidden_dim * 6, 1)
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            patch_size=patch_size,
            img_size=img_size,
            num_heads=num_heads * 2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_chunk=num_chunks,
            groups=hidden_dim,
        )
        self.hf_compress = nn.Conv2d(hidden_dim * 6, hidden_dim, kernel_size=1)

    def forward(self, cond):
        ms = [self.ms_conv(cond[:, : self.ms_channels])]
        pan = [self.pan_conv(cond[:, self.ms_channels : self.ms_channels + self.pan_channels])]
        for i in range(2):
            ms.append(self.ms_dw_convs[i](ms[i]))
            pan.append(self.pan_dw_convs[i](pan[i]))
        ms[0], pan[0] = self.ms_pan_fuse(ms[0], pan[0])
        for i in range(2):
            ms[i + 1], pan[i + 1] = self.fuse_nets[i](ms[i + 1], pan[i + 1])
            ms[i + 1] = self.ms_up_convs[i](ms[i + 1])
            pan[i + 1] = self.pan_up_convs[i](pan[i + 1])
        ms = self.ms_refine_conv(torch.cat(ms, dim=1))
        pan = self.pan_refine_conv(torch.cat(pan, dim=1))
        fusion = self.fusion_net(ms, pan)
        cond_base, global_token = self.cond_inj(fusion)
        cond_hf = self.hf_compress(pan)
        return cond_base, cond_hf, global_token


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, inner_channel, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.position_encoder = PositionalEncoding(
            num_patch=(img_size // patch_size) * (img_size // patch_size),
            hidden_size=patch_size * patch_size * inner_channel,
        )
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.reg = Rearrange("b N (c s1 s2) -> b N c s1 s2", s1=patch_size, s2=patch_size)
        self.seg = Rearrange("b c (n1 s1) (n2 s2) -> b (n1 n2) c s1 s2", s1=patch_size, s2=patch_size)
        self.comb = Rearrange(
            "b (n1 n2) c s1 s2 -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
        )

    def forward(self, noisy):
        noisy_patches = self.seg(noisy)
        position_emb = repeat(self.position_encoder(self.p), "1 n e -> b n e", b=noisy_patches.size(0))
        noisy_patches = noisy_patches + self.reg(position_emb)
        return self.comb(noisy_patches)


class FinalLayer(nn.Module):
    def __init__(self, ms_channels, pan_channels, img_size, hidden_channels, patch_size=16, out_channels=3, num_heads=16):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(
            in_channels=ms_channels * 2,
            hidden_channels=hidden_channels * 4,
            out_channels=hidden_channels * 6,
            kernel_size=3,
            img_size=img_size,
            need_residual=True,
        )
        self.pan_conv = ConvLayer(
            in_channels=pan_channels * 4,
            hidden_channels=hidden_channels * 4,
            out_channels=hidden_channels * 6,
            kernel_size=3,
            img_size=img_size,
            need_residual=True,
        )
        hidden_size = hidden_channels * patch_size * patch_size
        self.norm_final = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 2,
            patch_size=patch_size,
            img_size=img_size,
            num_heads=num_heads * 2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_channels * 2,
            hidden_dim=hidden_channels,
            patch_size=patch_size,
            num_chunk=2,
            groups=hidden_channels,
        )
        self.seg = Rearrange("b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)", s1=patch_size, s2=patch_size)
        self.comb = Rearrange(
            "b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
            s1=patch_size,
            s2=patch_size,
        )

    def forward(self, x, cond):
        ms = torch.cat(
            [
                cond[:, : self.ms_channels],
                cond[:, self.ms_channels + self.pan_channels : self.ms_channels * 2 + self.pan_channels],
            ],
            dim=1,
        )
        pan = torch.cat(
            [
                cond[:, self.ms_channels : self.ms_channels + self.pan_channels],
                cond[:, self.ms_channels * 2 + self.pan_channels :],
            ],
            dim=1,
        )
        ms = self.ms_conv(ms)
        pan = self.pan_conv(pan)
        fusion = self.fusion_net(ms, pan)
        cond, _ = self.cond_inj(fusion)
        shift, scale = cond.chunk(2, dim=1)
        x = self.linear(self.seg(modulate(self.norm_final(x), shift, scale)))
        return self.comb(x)


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, device=None, dtype=None):
        super().__init__()
        try:
            from mamba_ssm import Mamba

            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                device=device,
                dtype=dtype,
            )
            self.is_available = True
        except ImportError:
            self.mamba = nn.Identity()
            self.is_available = False

    def forward(self, x):
        if not self.is_available:
            return x

        b, c, h, w = x.shape
        xs_1 = rearrange(x, "b c h w -> b (h w) c")
        xs_2 = rearrange(torch.flip(x, dims=[-1, -2]), "b c h w -> b (h w) c")
        xs_3 = rearrange(x.transpose(-1, -2), "b c w h -> b (w h) c")
        xs_4 = rearrange(torch.flip(x.transpose(-1, -2), dims=[-1, -2]), "b c w h -> b (w h) c")
        xs = torch.cat([xs_1, xs_2, xs_3, xs_4], dim=0).contiguous()
        out = self.mamba(xs)
        out_1, out_2, out_3, out_4 = out.chunk(4, dim=0)
        y_1 = rearrange(out_1, "b (h w) c -> b c h w", h=h, w=w)
        y_2 = torch.flip(rearrange(out_2, "b (h w) c -> b c h w", h=h, w=w), dims=[-1, -2])
        y_3 = rearrange(out_3, "b (w h) c -> b c w h", h=h, w=w).transpose(-1, -2)
        y_4 = torch.flip(rearrange(out_4, "b (w h) c -> b c w h", h=h, w=w), dims=[-1, -2]).transpose(-1, -2)
        return y_1 + y_2 + y_3 + y_4


class PhysicalDetailSeed(nn.Module):
    def __init__(self, pan_channels, cond_dim):
        super().__init__()
        kernel_small = self._get_gaussian_kernel(3, sigma=1.0).view(1, 1, 3, 3).repeat(pan_channels, 1, 1, 1)
        kernel_large = self._get_gaussian_kernel(5, sigma=2.0).view(1, 1, 5, 5).repeat(pan_channels, 1, 1, 1)
        self.register_buffer("weight_small", kernel_small)
        self.register_buffer("weight_large", kernel_large)
        self.pad_small = nn.ReflectionPad2d(1)
        self.pad_large = nn.ReflectionPad2d(2)
        num_groups = 32 if cond_dim % 32 == 0 else (16 if cond_dim % 16 == 0 else (8 if cond_dim % 8 == 0 else 1))
        self.proj = nn.Sequential(
            nn.Conv2d(pan_channels * 2, cond_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=cond_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1),
        )

    @staticmethod
    def _get_gaussian_kernel(kernel_size, sigma):
        x = torch.arange(kernel_size).float() - kernel_size // 2
        y = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def forward(self, pan_raw):
        b, c, _, _ = pan_raw.shape
        blur_small = F.conv2d(self.pad_small(pan_raw), self.weight_small, groups=c)
        blur_large = F.conv2d(self.pad_large(pan_raw), self.weight_large, groups=c)
        d_small = pan_raw - blur_small
        d_large = pan_raw - blur_large
        sigma_small = d_small.view(b, c, -1).std(dim=-1).view(b, c, 1, 1) + 1e-6
        sigma_large = d_large.view(b, c, -1).std(dim=-1).view(b, c, 1, 1) + 1e-6
        d_small = d_small / sigma_small
        d_large = d_large / sigma_large
        feat = torch.cat([d_small, d_large], dim=1)
        phys_feat = self.proj(feat)
        phys_energy = (d_small.abs() + d_large.abs()).mean(dim=1, keepdim=True)
        phys_energy = phys_energy / (phys_energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return phys_feat, phys_energy


def _pick_group_count(channels):
    if channels % 32 == 0:
        return 32
    if channels % 16 == 0:
        return 16
    if channels % 8 == 0:
        return 8
    return 1


class DegradationPosteriorEncoder(nn.Module):
    def __init__(self, pan_channels, cond_dim, time_dim, posterior_dim=None):
        super().__init__()
        posterior_dim = time_dim if posterior_dim is None else posterior_dim
        self.posterior_dim = posterior_dim
        self.cond_dim = cond_dim
        self.physical_detail_seed = PhysicalDetailSeed(pan_channels, cond_dim)
        self.learned_align = nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv2d(1, cond_dim, kernel_size=1)
        self.global_proj = nn.Linear(time_dim, cond_dim)
        self.time_proj = nn.Linear(time_dim, cond_dim)
        self.posterior_fuse = nn.Sequential(
            nn.Conv2d(cond_dim * 5, cond_dim * 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_pick_group_count(cond_dim * 2), num_channels=cond_dim * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(cond_dim * 2, cond_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=_pick_group_count(cond_dim), num_channels=cond_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1, bias=True),
        )
        self.map_head = nn.Conv2d(cond_dim, 1, kernel_size=1)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, posterior_dim),
            nn.SiLU(),
            nn.Linear(posterior_dim, posterior_dim),
        )

    def forward(self, cond_hf, pan_raw, global_token, time_token):
        posterior_seed, phys_energy = self.physical_detail_seed(pan_raw)
        learned_feat = self.learned_align(cond_hf)
        energy_feat = self.energy_proj(phys_energy)
        b, _, h, w = posterior_seed.shape
        global_feat = self.global_proj(global_token).unsqueeze(-1).unsqueeze(-1).expand(b, self.cond_dim, h, w)
        time_feat = self.time_proj(time_token).unsqueeze(-1).unsqueeze(-1).expand(b, self.cond_dim, h, w)
        fused = torch.cat([posterior_seed, learned_feat, energy_feat, global_feat, time_feat], dim=1)
        posterior_feat = self.posterior_fuse(fused) + posterior_seed
        learned_map = torch.sigmoid(self.map_head(posterior_feat))
        d_map = 0.5 * learned_map + 0.5 * phys_energy
        pooled_mean = F.adaptive_avg_pool2d(posterior_feat * (1 + d_map), 1).flatten(1)
        pooled_std = posterior_feat.flatten(2).std(dim=-1)
        z_deg = self.posterior_mlp(torch.cat([pooled_mean, pooled_std], dim=1))
        return posterior_feat, d_map, z_deg, phys_energy


class PosteriorControlMapper(nn.Module):
    def __init__(self, cond_dim, out_dim, time_dim, pan_channels=1, adapter_rank=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.adapter_rank = max(4, min(8, cond_dim)) if adapter_rank is None else adapter_rank
        self.posterior_encoder = DegradationPosteriorEncoder(
            pan_channels=pan_channels,
            cond_dim=cond_dim,
            time_dim=time_dim,
            posterior_dim=time_dim,
        )
        self.posterior_to_feat = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim * 2, cond_dim * 2))
        self.norm = nn.GroupNorm(num_groups=_pick_group_count(cond_dim), num_channels=cond_dim, affine=False)
        self.act = nn.SiLU()
        self.zero_conv = nn.Conv2d(cond_dim, out_dim * 2, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        self.gamma_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2))
        nn.init.zeros_(self.gamma_mlp[1].weight)
        nn.init.zeros_(self.gamma_mlp[1].bias)
        self.hyper_ssm = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_dim * 2))
        nn.init.zeros_(self.hyper_ssm[1].weight)
        nn.init.zeros_(self.hyper_ssm[1].bias)
        self.hyper_in_left = nn.Linear(time_dim, cond_dim * self.adapter_rank)
        self.hyper_in_right = nn.Parameter(torch.zeros(self.adapter_rank, out_dim * 2))
        self.hyper_out_left = nn.Linear(time_dim, out_dim * self.adapter_rank)
        self.hyper_out_right = nn.Parameter(torch.zeros(self.adapter_rank, cond_dim))
        self.hyper_in_scale = nn.Parameter(torch.tensor(1e-3))
        self.hyper_out_scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, cond_hf, pan_raw, global_token, time_token):
        posterior_feat, d_map, z_deg, phys_energy = self.posterior_encoder(
            cond_hf=cond_hf,
            pan_raw=pan_raw,
            global_token=global_token,
            time_token=time_token,
        )
        posterior_bias = self.posterior_to_feat(z_deg).unsqueeze(-1).unsqueeze(-1)
        x = posterior_feat + posterior_bias
        time_params = self.time_mlp(torch.cat([time_token, z_deg], dim=1))
        t_scale, t_shift = time_params.chunk(2, dim=1)
        x = self.norm(x)
        x = x * (1 + t_scale.unsqueeze(-1).unsqueeze(-1)) + t_shift.unsqueeze(-1).unsqueeze(-1)
        x = self.act(x)
        params = self.zero_conv(x)
        scale, shift = params.chunk(2, dim=1)
        posterior_gate = 0.25 + 0.75 * d_map
        scale = scale * posterior_gate
        shift = shift * posterior_gate
        delta_gamma = 0.1 * torch.tanh(self.gamma_mlp(z_deg))
        delta_ssm = self.hyper_ssm(z_deg)
        in_left = self.hyper_in_left(z_deg).view(z_deg.size(0), self.cond_dim, self.adapter_rank)
        out_left = self.hyper_out_left(z_deg).view(z_deg.size(0), self.out_dim, self.adapter_rank)
        hyper = {
            "z_deg": z_deg,
            "d_map": d_map,
            "delta_gamma": delta_gamma,
            "delta_ssm": delta_ssm,
            "in_left": in_left,
            "out_left": out_left,
            "in_scale": torch.tanh(self.hyper_in_scale),
            "out_scale": torch.tanh(self.hyper_out_scale),
        }
        with torch.no_grad():
            self._last_scale_abs_mean = float(scale.abs().detach().mean().item())
            self._last_shift_abs_mean = float(shift.abs().detach().mean().item())
            self._last_phys_energy = float(phys_energy.detach().mean().item())
            self._last_alpha_aux = float(z_deg.detach().norm(dim=1).mean().item())
            self._last_z_deg_norm = float(z_deg.detach().norm(dim=1).mean().item())
            self._last_posterior_energy = float(d_map.detach().mean().item())
            self._last_d_map_mean = float(d_map.detach().mean().item())
            self._last_delta_gamma_abs = float(delta_gamma.detach().abs().mean().item())
            self._last_delta_ssm_abs = float(delta_ssm.detach().abs().mean().item())
            self._last_hyper_in_scale = float(torch.tanh(self.hyper_in_scale).detach().item())
            self._last_hyper_out_scale = float(torch.tanh(self.hyper_out_scale).detach().item())
        return scale, shift, hyper


class CondDiffusionMambaBlock(nn.Module):
    def __init__(self, img_size, hidden_channels, patch_size, num_heads=16, mlp_ratio=4, expand=2, d_conv=3, pan_channels=1, **_kwargs):
        super().__init__()
        dim = hidden_channels
        cond_dim = hidden_channels
        self.expand = expand
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        time_dim = hidden_channels * patch_size * patch_size
        self.posterior_control_mapper = PosteriorControlMapper(
            cond_dim=cond_dim,
            out_dim=dim * expand,
            time_dim=time_dim,
            pan_channels=pan_channels,
            adapter_rank=max(4, min(8, hidden_channels)),
        )
        self.global_proj = nn.Linear(time_dim, dim)
        self.posterior_global_proj = nn.Linear(time_dim, dim)
        nn.init.zeros_(self.posterior_global_proj.weight)
        nn.init.zeros_(self.posterior_global_proj.bias)
        self.gamma_gate = nn.Parameter(torch.full((1,), 0.1))
        self.gamma_main = nn.Parameter(torch.full((1,), 0.1))
        self.ssm_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim * expand * 2))
        nn.init.zeros_(self.ssm_time_mlp[1].weight)
        nn.init.zeros_(self.ssm_time_mlp[1].bias)
        self.in_proj = nn.Linear(dim, dim * expand * 2)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            in_channels=dim * expand,
            out_channels=dim * expand,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=dim * expand,
        )
        self.ss2d_core = SS2D(d_model=dim * expand)
        self.out_proj = nn.Linear(dim * expand, dim)
        self.norm2 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio, out_features=dim)

    @staticmethod
    def _apply_low_rank_adapter(x, left, right):
        return torch.einsum("bhwd,bdr,ro->bhwo", x, left, right)

    def forward(self, x, cond_base, cond_hf, pan_raw, global_token, time_token):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, cond_feat, _ = cond_base.chunk(8, dim=1)
        scale, shift, hyper = self.posterior_control_mapper(
            cond_hf=cond_hf,
            pan_raw=pan_raw,
            global_token=global_token,
            time_token=time_token,
        )
        z_deg = hyper["z_deg"]
        d_map = hyper["d_map"]
        b, _, _, _ = x.shape
        shortcut = x
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        global_bias = self.global_proj(global_token).unsqueeze(-1).unsqueeze(-1)
        posterior_global_bias = self.posterior_global_proj(z_deg).unsqueeze(-1).unsqueeze(-1)
        cond_feat_enriched = cond_feat + global_bias + posterior_global_bias
        cond_bias_nhwc = cond_feat_enriched.permute(0, 2, 3, 1).repeat(1, 1, 1, self.expand)
        d_map_nhwc = d_map.permute(0, 2, 3, 1)
        x_nhwc = x_norm.permute(0, 2, 3, 1)
        x_proj = self.in_proj(x_nhwc)
        delta_in = self._apply_low_rank_adapter(
            x_nhwc,
            hyper["in_left"],
            self.posterior_control_mapper.hyper_in_right,
        )
        x_proj = x_proj + hyper["in_scale"] * delta_in
        x_ss2d, x_gate = x_proj.chunk(2, dim=-1)
        delta_gamma_gate = hyper["delta_gamma"][:, 0].view(b, 1, 1, 1)
        delta_gamma_main = hyper["delta_gamma"][:, 1].view(b, 1, 1, 1)
        x_gate_modulated = self.act(x_gate + (self.gamma_gate + delta_gamma_gate) * cond_bias_nhwc)
        x_gate_modulated = x_gate_modulated * (1 + 0.1 * d_map_nhwc)
        x_ss2d = self.act(self.conv2d(x_ss2d.permute(0, 3, 1, 2)))
        ssm_scale, ssm_shift = self.ssm_time_mlp(time_token).chunk(2, dim=1)
        delta_ssm_scale, delta_ssm_shift = hyper["delta_ssm"].chunk(2, dim=1)
        ssm_scale = (ssm_scale + delta_ssm_scale).unsqueeze(-1).unsqueeze(-1)
        ssm_shift = (ssm_shift + delta_ssm_shift).unsqueeze(-1).unsqueeze(-1)
        cond_bias_nchw = cond_bias_nhwc.permute(0, 3, 1, 2)
        x_ss2d = (x_ss2d + (self.gamma_main + delta_gamma_main) * cond_bias_nchw) * (1 + ssm_scale) + ssm_shift
        x_ss2d = x_ss2d * (1 + 0.1 * d_map)
        x_ss2d = self.ss2d_core(x_ss2d)
        x_fused = x_ss2d * x_gate_modulated.permute(0, 3, 1, 2)
        x_sft = x_fused * (1 + scale) + shift
        x_sft_nhwc = x_sft.permute(0, 2, 3, 1)
        x_out = self.out_proj(x_sft_nhwc)
        delta_out = self._apply_low_rank_adapter(
            x_sft_nhwc,
            hyper["out_left"],
            self.posterior_control_mapper.hyper_out_right,
        )
        x_out = x_out + hyper["out_scale"] * delta_out
        with torch.no_grad():
            base_in_abs = float(self.in_proj(x_nhwc).detach().abs().mean().item())
            delta_in_abs = float((hyper["in_scale"] * delta_in).detach().abs().mean().item())
            base_out_abs = float(self.out_proj(x_sft_nhwc).detach().abs().mean().item())
            delta_out_abs = float((hyper["out_scale"] * delta_out).detach().abs().mean().item())
            eps = 1e-8
            self._last_delta_in_ratio = float(delta_in_abs / (base_in_abs + eps))
            self._last_delta_out_ratio = float(delta_out_abs / (base_out_abs + eps))
            self._last_delta_in_abs = float(delta_in_abs)
            self._last_delta_out_abs = float(delta_out_abs)
            self._last_base_in_abs = float(base_in_abs)
            self._last_base_out_abs = float(base_out_abs)
            self._last_gate_mean = float(x_gate_modulated.detach().abs().mean().item())
            self._last_ssm_mean = float(x_ss2d.detach().abs().mean().item())
        x = shortcut + x_out.permute(0, 3, 1, 2) * gate_msa
        shortcut = x
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_norm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return shortcut + x_mlp * gate_mlp


class DiMWithCond(nn.Module):
    def __init__(
        self,
        hidden_channels,
        img_size,
        mode,
        ms_channels,
        pan_channels,
        patch_size=16,
        noise_level_emb_dim=None,
        time_hidden_ratio=4,
        num_heads=16,
        mlp_ratio=4,
        **_kwargs,
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        if mode == "WAVE":
            self.cond_inj = WaveletBlock(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_heads=num_heads,
                num_chunks=8,
            )
        elif mode == "PYRAMID":
            self.cond_inj = PyramidalSpatialBlock(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_heads=num_heads,
                num_chunks=8,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.time_mlp = nn.Sequential(
            nn.Linear(noise_level_emb_dim, patch_size * patch_size * hidden_channels // time_hidden_ratio),
            nn.SiLU(),
            nn.Linear(
                patch_size * patch_size * hidden_channels // time_hidden_ratio,
                patch_size * patch_size * hidden_channels,
            ),
        )
        self.dit = CondDiffusionMambaBlock(
            img_size=img_size,
            hidden_channels=hidden_channels,
            patch_size=patch_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pan_channels=pan_channels,
        )

    def forward(self, x, time_emb, cond):
        cond_base, cond_hf, global_token = self.cond_inj(cond)
        time_token = self.time_mlp(time_emb)
        pan_raw = cond[:, self.ms_channels : self.ms_channels + self.pan_channels]
        return self.dit(x, cond_base, cond_hf, pan_raw, global_token, time_token)


class DiM(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        hidden_channels,
        ms_channels,
        pan_channels,
        noise_level_emb_dim,
        time_hidden_ratio=4,
        num_layers=6,
        num_heads=16,
        mlp_ratio=4,
        neck_type="hybrid",
        **_kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if neck_type == "hybrid":
                mode = "PYRAMID" if i < num_layers // 2 else "WAVE"
            elif neck_type == "wave":
                mode = "WAVE"
            elif neck_type == "pyramid":
                mode = "PYRAMID"
            else:
                raise ValueError(f"Unknown neck_type: {neck_type}")
            self.layers.append(
                DiMWithCond(
                    hidden_channels=hidden_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    mode=mode,
                    ms_channels=ms_channels,
                    pan_channels=pan_channels,
                    noise_level_emb_dim=noise_level_emb_dim,
                    time_hidden_ratio=time_hidden_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
            )

    def forward(self, x, time_emb, cond, **_kwargs):
        for layer in self.layers:
            x = layer(x, time_emb, cond)
        return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
