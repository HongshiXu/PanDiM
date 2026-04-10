import math
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def pick_group_count(channels):
    if channels % 32 == 0:
        return 32
    if channels % 16 == 0:
        return 16
    if channels % 8 == 0:
        return 8
    return 1


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


class CrossModalInjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, num_heads, mode):
        super().__init__()
        hidden_dim = in_channels * patch_size * patch_size
        self.mode = mode
        self.num_heads = num_heads
        self.depth = hidden_dim // num_heads
        self.position_encoder = PositionalEncoding(
            num_patch=(img_size // patch_size) * (img_size // patch_size),
            hidden_size=patch_size * patch_size * in_channels,
        )
        self.level = nn.Parameter(torch.tensor([1.0]))
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LayerNorm([out_channels, img_size, img_size], eps=1e-6, elementwise_affine=False),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.seg = Rearrange(
            "b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)",
            s1=patch_size,
            s2=patch_size,
        )
        self.comb = Rearrange(
            "b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
            s1=patch_size,
            s2=patch_size,
        )

    def forward(self, k, q, v):
        batch, token_count, embed_dim = k.shape
        scale = embed_dim ** -0.5
        position = self.position_encoder(self.level)
        position = repeat(position, "1 n e -> b n e", b=batch)
        k, q, v = k + position, q + position, v + position
        q = q.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
        mixed = mixed.transpose(1, 2).reshape(batch, token_count, embed_dim)
        v = v.transpose(1, 2).reshape(batch, token_count, embed_dim)
        q = q.transpose(1, 2).reshape(batch, token_count, embed_dim)
        if self.mode == "discrepancy":
            mixed = v - mixed
        q = self.linear(mixed) + q
        q = self.comb(q)
        q = self.body(q)
        return self.seg(q)


class ModalAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, num_heads):
        super().__init__()
        self.diff_branch = CrossModalInjectionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            mode="discrepancy",
        )
        self.common_branch_1 = CrossModalInjectionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            mode="common",
        )
        self.common_branch_2 = CrossModalInjectionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            mode="common",
        )
        self.reg = Rearrange(
            "b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)",
            s1=patch_size,
            s2=patch_size,
        )
        self.comb = Rearrange(
            "b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
            s1=patch_size,
            s2=patch_size,
        )

    def forward(self, ms, pan):
        pan_k, pan_q, pan_v = pan.chunk(3, dim=1)
        pan_k, pan_q, pan_v = self.reg(pan_k), self.reg(pan_q), self.reg(pan_v)
        ms_k, ms_q, ms_v = ms.chunk(3, dim=1)
        ms_k, ms_q, ms_v = self.reg(ms_k), self.reg(ms_q), self.reg(ms_v)
        q1 = self.diff_branch(pan_k, pan_v, ms_q)
        q2 = self.common_branch_1(ms_k, ms_v, q1)
        q2 = self.common_branch_2(pan_k, pan_v, q2)
        return self.comb(q1 + q2)


class ConditionProjector(nn.Module):
    def __init__(self, cond_dim, hidden_dim, patch_size, img_size, num_chunk, groups):
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
        self.reg = Rearrange(
            "b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)",
            s1=patch_size,
            s2=patch_size,
        )

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
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if need_residual else None

    def forward(self, x):
        if self.res_conv is None:
            return self.body(x)
        return self.body(x) + self.res_conv(x)


class WaveletConditionNeck(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(ms_channels * 2, hidden_dim * 4, hidden_dim * 6, 3, img_size, need_residual=True)
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
        self.pan_body = ConvLayer(pan_channels * 3, hidden_dim * 4, hidden_dim * 6, 3, img_size, need_residual=True)
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            patch_size=patch_size,
            img_size=img_size,
            num_heads=num_heads * 2,
        )
        self.cond_projector = ConditionProjector(
            cond_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_chunk=num_chunks,
            groups=hidden_dim,
        )
        self.hf_compress = nn.Conv2d(hidden_dim * 6, hidden_dim, kernel_size=1)

    def forward(self, cond):
        ms = torch.cat(
            [cond[:, : self.ms_channels], cond[:, self.ms_channels + self.pan_channels : self.ms_channels * 2 + self.pan_channels]],
            dim=1,
        )
        ms = self.ms_conv(ms)
        pan = cond[:, self.ms_channels : self.ms_channels + self.pan_channels]
        pan_wavelet = cond[:, self.ms_channels * 2 + self.pan_channels :]
        pan_scale, pan_shift = self.pan_scale_conv(pan), self.pan_shift_conv(pan)
        pan_hf_feat = modulate(self.pan_body(pan_wavelet), pan_shift, pan_scale)
        fusion = self.fusion_net(ms, pan_hf_feat)
        cond_base, global_token = self.cond_projector(fusion)
        cond_hf = self.hf_compress(pan_hf_feat)
        return cond_base, cond_hf, global_token


class PyramidFuseBlock(nn.Module):
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
        return modulate(ms, ms_shift, ms_scale), modulate(pan, pan_shift, pan_scale)


class PyramidUpSampler(nn.Module):
    def __init__(self, in_channels, n_levels, img_size):
        super().__init__()
        self.up_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels // (2 ** i), in_channels // (2 ** (i + 1)), 4, 2, 1),
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


class PyramidConditionNeck(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(ms_channels, hidden_dim * 2, hidden_dim * 2, 3, img_size, need_residual=False)
        self.pan_conv = ConvLayer(pan_channels, hidden_dim * 2, hidden_dim * 2, 3, img_size, need_residual=False)
        self.ms_pan_fuse = PyramidFuseBlock(hidden_dim * 2)
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
        self.fuse_nets = nn.ModuleList([PyramidFuseBlock(hidden_dim * (2 ** (i + 2))) for i in range(2)])
        self.ms_up_convs = nn.ModuleList([PyramidUpSampler(hidden_dim * (2 ** (i + 2)), i + 1, img_size) for i in range(2)])
        self.pan_up_convs = nn.ModuleList([PyramidUpSampler(hidden_dim * (2 ** (i + 2)), i + 1, img_size) for i in range(2)])
        self.ms_refine_conv = nn.Conv2d(hidden_dim * 6, hidden_dim * 6, 1)
        self.pan_refine_conv = nn.Conv2d(hidden_dim * 6, hidden_dim * 6, 1)
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            patch_size=patch_size,
            img_size=img_size,
            num_heads=num_heads * 2,
        )
        self.cond_projector = ConditionProjector(
            cond_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            img_size=img_size,
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
        cond_base, global_token = self.cond_projector(fusion)
        cond_hf = self.hf_compress(pan)
        return cond_base, cond_hf, global_token


class DiMStage(nn.Module):
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
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        if mode == "WAVE":
            self.condition_neck = WaveletConditionNeck(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_chunks=8,
                num_heads=num_heads,
            )
        elif mode == "PYRAMID":
            self.condition_neck = PyramidConditionNeck(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_heads=num_heads,
                num_chunks=8,
            )
        else:
            raise ValueError(f"Unknown neck_type mode: {mode}")
        token_dim = patch_size * patch_size * hidden_channels
        hidden_time_dim = token_dim // time_hidden_ratio
        self.time_mlp = nn.Sequential(
            nn.Linear(noise_level_emb_dim, hidden_time_dim),
            nn.SiLU(),
            nn.Linear(hidden_time_dim, token_dim),
        )
        self.mamba_block = ConditionModulatedMambaBlock(
            img_size=img_size,
            hidden_channels=hidden_channels,
            patch_size=patch_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pan_channels=pan_channels,
        )

    def forward(self, x, time_emb, cond):
        cond_base, cond_hf, global_token = self.condition_neck(cond)
        time_token = self.time_mlp(time_emb)
        pan_raw = cond[:, self.ms_channels : self.ms_channels + self.pan_channels]
        return self.mamba_block(x, cond_base, cond_hf, pan_raw, global_token, time_token)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, inner_channel, patch_size=16):
        super().__init__()
        self.position_encoder = PositionalEncoding(
            num_patch=(img_size // patch_size) * (img_size // patch_size),
            hidden_size=patch_size * patch_size * inner_channel,
        )
        self.level = nn.Parameter(torch.tensor([1.0]))
        self.reg = Rearrange("b n (c s1 s2) -> b n c s1 s2", s1=patch_size, s2=patch_size)
        self.seg = Rearrange("b c (n1 s1) (n2 s2) -> b (n1 n2) c s1 s2", s1=patch_size, s2=patch_size)
        self.comb = Rearrange(
            "b (n1 n2) c s1 s2 -> b c (n1 s1) (n2 s2)",
            n1=img_size // patch_size,
            n2=img_size // patch_size,
        )

    def forward(self, noisy):
        noisy_patches = self.seg(noisy)
        position = self.position_encoder(self.level)
        position = repeat(position, "1 n e -> b n e", b=noisy_patches.size(0))
        position = self.reg(position)
        return self.comb(noisy_patches + position)


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads, patch_size):
        super().__init__()
        hidden_size = hidden_channels * patch_size * patch_size
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = hidden_size // num_heads
        self.scale = hidden_size ** -0.5
        self.kv_conv = conv3x3(hidden_channels, hidden_channels * 2)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.reg = Rearrange(
            "b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)",
            s1=patch_size,
            s2=patch_size,
        )

    def forward(self, x, cond):
        batch, _, height, width = x.shape
        q = self.q_linear(self.reg(x))
        k, v = self.reg(self.kv_conv(cond)).chunk(2, dim=2)
        q = q.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        return out.transpose(1, 2).reshape(batch, -1, height, width)


class FinalLayer(nn.Module):
    def __init__(self, ms_channels, pan_channels, img_size, hidden_channels, patch_size=16, out_channels=3, num_heads=16):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(ms_channels * 2, hidden_channels * 4, hidden_channels * 6, 3, img_size, need_residual=True)
        self.pan_conv = ConvLayer(pan_channels * 4, hidden_channels * 4, hidden_channels * 6, 3, img_size, need_residual=True)
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
        self.cond_projector = ConditionProjector(
            cond_dim=hidden_channels * 2,
            hidden_dim=hidden_channels,
            patch_size=patch_size,
            img_size=img_size,
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
            [cond[:, : self.ms_channels], cond[:, self.ms_channels + self.pan_channels : self.ms_channels * 2 + self.pan_channels]],
            dim=1,
        )
        pan = torch.cat(
            [cond[:, self.ms_channels : self.ms_channels + self.pan_channels], cond[:, self.ms_channels * 2 + self.pan_channels :]],
            dim=1,
        )
        ms = self.ms_conv(ms)
        pan = self.pan_conv(pan)
        fusion = self.fusion_net(ms, pan)
        cond, _ = self.cond_projector(fusion)
        shift, scale = cond.chunk(2, dim=1)
        x = self.linear(self.seg(modulate(self.norm_final(x), shift, scale)))
        return self.comb(x)


class DiMBackbone(nn.Module):
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
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(num_layers):
            if neck_type == "hybrid":
                mode = "PYRAMID" if index < num_layers // 2 else "WAVE"
            elif neck_type == "wave":
                mode = "WAVE"
            elif neck_type == "pyramid":
                mode = "PYRAMID"
            else:
                raise ValueError(f"Unknown neck_type: {neck_type}")
            self.layers.append(
                DiMStage(
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

    def forward(self, x, time_emb, cond):
        for layer in self.layers:
            x = layer(x, time_emb, cond)
        return x


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
        batch, _, height, width = x.shape
        seq_1 = rearrange(x, "b c h w -> b (h w) c")
        seq_2 = rearrange(torch.flip(x, dims=[-1, -2]), "b c h w -> b (h w) c")
        seq_3 = rearrange(x.transpose(-1, -2), "b c w h -> b (w h) c")
        seq_4 = rearrange(torch.flip(x.transpose(-1, -2), dims=[-1, -2]), "b c w h -> b (w h) c")
        stacked = torch.cat([seq_1, seq_2, seq_3, seq_4], dim=0).contiguous()
        out = self.mamba(stacked)
        out_1, out_2, out_3, out_4 = out.chunk(4, dim=0)
        y_1 = rearrange(out_1, "b (h w) c -> b c h w", h=height, w=width)
        y_2 = torch.flip(rearrange(out_2, "b (h w) c -> b c h w", h=height, w=width), dims=[-1, -2])
        y_3 = rearrange(out_3, "b (w h) c -> b c w h", h=height, w=width).transpose(-1, -2)
        y_4 = torch.flip(rearrange(out_4, "b (w h) c -> b c w h", h=height, w=width), dims=[-1, -2]).transpose(-1, -2)
        return y_1 + y_2 + y_3 + y_4


class PhysicsDetailPrior(nn.Module):
    def __init__(self, pan_channels, cond_dim):
        super().__init__()
        kernel_small = self._get_gaussian_kernel(3, sigma=1.0).view(1, 1, 3, 3).repeat(pan_channels, 1, 1, 1)
        kernel_large = self._get_gaussian_kernel(5, sigma=2.0).view(1, 1, 5, 5).repeat(pan_channels, 1, 1, 1)
        self.register_buffer("weight_small", kernel_small)
        self.register_buffer("weight_large", kernel_large)
        self.pad_small = nn.ReflectionPad2d(1)
        self.pad_large = nn.ReflectionPad2d(2)
        groups = pick_group_count(cond_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(pan_channels * 2, cond_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=cond_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1),
        )

    def _get_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(kernel_size).float() - kernel_size // 2
        y = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def forward(self, pan_raw):
        batch, channels, _, _ = pan_raw.shape
        blur_small = F.conv2d(self.pad_small(pan_raw), self.weight_small, groups=channels)
        blur_large = F.conv2d(self.pad_large(pan_raw), self.weight_large, groups=channels)
        d_small = pan_raw - blur_small
        d_large = pan_raw - blur_large
        sigma_small = d_small.view(batch, channels, -1).std(dim=-1).view(batch, channels, 1, 1) + 1e-6
        sigma_large = d_large.view(batch, channels, -1).std(dim=-1).view(batch, channels, 1, 1) + 1e-6
        d_small = d_small / sigma_small
        d_large = d_large / sigma_large
        feat = torch.cat([d_small, d_large], dim=1)
        phys_feat = self.proj(feat)
        phys_energy = (d_small.abs() + d_large.abs()).mean(dim=1, keepdim=True)
        phys_energy = phys_energy / (phys_energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return phys_feat, phys_energy


class SeedConfidenceAffineModulator(nn.Module):
    def __init__(self, cond_dim, out_dim, time_dim, pan_channels=1):
        super().__init__()
        groups = pick_group_count(cond_dim)
        self.physics_prior = PhysicsDetailPrior(pan_channels, cond_dim)
        self.learned_align = nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1)
        self.alpha_aux = nn.Parameter(torch.tensor(0.0))
        self.energy_proj = nn.Conv2d(1, cond_dim, kernel_size=1)
        self.seed_fuse = nn.Sequential(
            nn.Conv2d(cond_dim * 3, cond_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=cond_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=cond_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1, bias=True),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, cond_dim * 2))
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=cond_dim, affine=False)
        self.act = nn.SiLU()
        self.param_head = nn.Conv2d(cond_dim, out_dim * 3, kernel_size=1)
        nn.init.zeros_(self.param_head.weight)
        nn.init.zeros_(self.param_head.bias)

    def forward(self, cond_hf, pan_raw, time_token):
        phys_feat, phys_energy = self.physics_prior(pan_raw)
        learned_feat = self.learned_align(cond_hf)
        energy_feat = self.energy_proj(phys_energy)
        beta = torch.tanh(self.alpha_aux)
        detail_seed = self.seed_fuse(torch.cat([phys_feat, beta * learned_feat, energy_feat], dim=1)) + phys_feat
        if not self.training or not hasattr(self, "_last_phys_energy"):
            self._last_phys_energy = phys_energy.detach().mean().item()
            self._last_alpha_aux = self.alpha_aux.detach().item()
        t_scale, t_shift = self.time_mlp(time_token).chunk(2, dim=1)
        x = self.norm(detail_seed)
        x = x * (1 + t_scale.unsqueeze(-1).unsqueeze(-1)) + t_shift.unsqueeze(-1).unsqueeze(-1)
        x = self.act(x)
        scale_raw, shift_raw, conf_logit = self.param_head(x).chunk(3, dim=1)
        gate = 0.3 + 0.7 * torch.sigmoid(conf_logit)
        scale = scale_raw * gate
        shift = shift_raw * gate
        if not self.training or not hasattr(self, "_last_scale_abs_mean"):
            self._last_scale_abs_mean = scale.abs().detach().mean().item()
            self._last_phys_energy = phys_energy.detach().mean().item()
            self._last_alpha_aux = self.alpha_aux.detach().item()
        return scale, shift


class ConditionModulatedMambaBlock(nn.Module):
    def __init__(self, img_size, hidden_channels, patch_size, num_heads=8, mlp_ratio=4, expand=2, d_conv=3, pan_channels=1):
        super().__init__()
        dim = hidden_channels
        time_dim = hidden_channels * patch_size * patch_size
        self.expand = expand
        self.norm1 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.affine_modulator = SeedConfidenceAffineModulator(
            cond_dim=hidden_channels,
            out_dim=dim * expand,
            time_dim=time_dim,
            pan_channels=pan_channels,
        )
        self.global_proj = nn.Linear(time_dim, dim)
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

    def forward(self, x, cond_base, cond_hf, pan_raw, global_token, time_token):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, cond_feat, _ = cond_base.chunk(8, dim=1)
        shortcut = x
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        cond_bias = cond_feat + self.global_proj(global_token).unsqueeze(-1).unsqueeze(-1)
        cond_bias_nhwc = cond_bias.permute(0, 2, 3, 1).repeat(1, 1, 1, self.expand)
        x_ss2d, x_gate = self.in_proj(x_norm.permute(0, 2, 3, 1)).chunk(2, dim=-1)
        x_gate = self.act(x_gate + self.gamma_gate * cond_bias_nhwc)
        x_ss2d = self.act(self.conv2d(x_ss2d.permute(0, 3, 1, 2)))
        ssm_scale, ssm_shift = self.ssm_time_mlp(time_token).chunk(2, dim=1)
        cond_bias_nchw = cond_bias_nhwc.permute(0, 3, 1, 2)
        x_ss2d = (x_ss2d + self.gamma_main * cond_bias_nchw) * (1 + ssm_scale.unsqueeze(-1).unsqueeze(-1)) + ssm_shift.unsqueeze(-1).unsqueeze(-1)
        x_ss2d = self.ss2d_core(x_ss2d)
        x_fused = x_ss2d * x_gate.permute(0, 3, 1, 2)
        scale, shift = self.affine_modulator(cond_hf, pan_raw, time_token)
        x_out = self.out_proj((x_fused * (1 + scale) + shift).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = shortcut + x_out * gate_msa
        shortcut = x
        x_mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return shortcut + x_mlp * gate_mlp
