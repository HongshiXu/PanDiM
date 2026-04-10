import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft
from einops import repeat
from einops.layers.torch import Rearrange
import math
from timm.models.vision_transformer import Attention, Mlp

class ModalAttentionFusionBlock(nn.Module):
    def __init__(self, img_size, patch_size, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size // patch_size, n2= img_size // patch_size, s1=patch_size, s2=patch_size)

    def forward(self, k, q, v):
        b, N, e = k.shape
        scale = e ** -0.5
        q = q.view(b, -1, self.num_heads, e // self.num_heads).transpose(1, 2)  
        k = k.view(b, -1, self.num_heads, e // self.num_heads).transpose(1, 2)  
        v = v.view(b, -1, self.num_heads, e // self.num_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
        out = out.transpose(1, 2).reshape(b, N, e)
        return self.comb(out)
 
class SpectralAttention(nn.Module):  
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels // ratio),
            nn.ReLU(),
            nn.Linear(out_channels // ratio, out_channels),
        )
         
    def forward(self, x):
        x = self.conv(x)
        avg_out = self.avg_pool(x)  
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.mlp(out.flatten(1))
        return out.unsqueeze(2).unsqueeze(3)
 
class SpatialAttention(nn.Module):
    def __init__(self, cond_channels, pan_channels, hidden_channels, ratio=2):
        super().__init__()
        self.k_conv = nn.Conv2d(cond_channels, hidden_channels // ratio, 5, 1, 2, bias=False)
        self.q_conv = nn.Conv2d(pan_channels, hidden_channels // ratio, 7, 1, 3, bias=False)
        self.v_conv = nn.Conv2d(pan_channels, hidden_channels // ratio, 7, 1, 3, bias=False)
        self.refine_conv = nn.Conv2d(hidden_channels // ratio, 1, 1)
        
        self.inner_dim = hidden_channels // ratio
        self.reg = Rearrange('b c h w -> b (c h) w')
        
    def forward(self, x, cond):
        b , c, h, w = x.size()
        k_cond = self.reg(self.k_conv(cond))
        q_pan = self.reg(self.q_conv(x)).permute(0, 2, 1)
        out = torch.bmm(q_pan, k_cond)
        v_pan = self.reg(self.v_conv(x))
        out = torch.bmm(v_pan, out.permute(0, 2, 1))
        out = self.refine_conv(out.view(b, self.inner_dim, h, w))
        
        return out
 
class SSAFBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_chunks=7, groups=32):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.body = nn.Sequential(
            conv3x3(ms_channels+pan_channels, hidden_dim),
            nn.LayerNorm([hidden_dim, img_size, img_size], eps=1e-6, elementwise_affine=False),
        )  
        self.spectrial_attention = SpectralAttention(ms_channels, hidden_dim, ratio=2) 
        self.spatial_attention = SpatialAttention(ms_channels+pan_channels, pan_channels, hidden_dim, ratio=4)
        self.cond_inj = CondBlock(
            cond_dim=hidden_dim*2, 
            hidden_dim=hidden_dim, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_chunk=num_chunks, 
        )
        
    def forward(self, cond):
        ms = cond[:, :self.ms_channels]
        pan = cond[:, self.ms_channels:self.ms_channels+self.pan_channels]
        out = self.body(cond[:, :self.ms_channels+self.pan_channels])
        spec_attention = self.spectrial_attention(ms)
        spa_attention = self.spatial_attention(pan, cond[:, :self.ms_channels+self.pan_channels])
        out = modulate(out, spa_attention, spec_attention)
        out, global_token = self.cond_inj(out)
        
        return out, global_token
    
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0.0,
        act="relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        return x

class LiteLA(Attention):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        norm_eps=1e-5,
    ):
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__(in_dim, num_heads=heads, qkv_bias=use_bias)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads  # TODO: need some change
        self.eps = eps
        
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        
        self.kernel_func = nn.ReLU(inplace=False)

    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, 3, C --> B, N, C
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()

class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=True,
            act=nn.SiLU(),
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            use_bias=True,
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=False,
            act=None
        )

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x

class SanaMSAdaLNBlock(nn.Module):
    """
    A Sana block with layer-wise adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        img_size,
        hidden_channels,
        patch_size,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        hidden_size = hidden_channels * patch_size * patch_size
        self.norm1 = nn.LayerNorm([hidden_channels, img_size, img_size], elementwise_affine=False, eps=1e-6)
        # linear self attention
        self.attn = LiteLA(hidden_size, hidden_size, heads=num_heads, eps=1e-8)

        self.norm2 = nn.LayerNorm([hidden_channels, img_size, img_size], elementwise_affine=False, eps=1e-6)

        self.mlp = GLUMBConv(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )

        self.scale_shift_table = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 6, hidden_channels * 6, 1),
        )
        
        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size//patch_size, n2=img_size//patch_size, s1=patch_size, s2=patch_size)
        
    def forward(self, x, cond, global_token, time_token):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.scale_shift_table(cond).chunk(6, dim=1)

        res = self.seg(modulate(self.norm1(x), shift_msa, scale_msa))
        res = torch.cat((global_token.unsqueeze(1), time_token.unsqueeze(1), res), dim=1)
        res = self.attn(res)
        x = x + self.comb(res[:, 2:]) * gate_msa
        x = x + self.comb(self.mlp(self.seg(modulate(self.norm2(x), shift_mlp, scale_mlp)))) * gate_mlp

        return x

class FinalDenoiseBlock(nn.Module):
    def __init__(self, hidden_channels, ms_channels, nb=3):
        super().__init__()
        self.body = nn.Sequential(*[
            GroupResidualDenseBlock(nf=hidden_channels, gc=hidden_channels//2) for _ in range(nb)
        ])
        self.conv = conv3x3(hidden_channels, ms_channels)
        
    def forward(self, x):
        return self.conv(self.body(x))