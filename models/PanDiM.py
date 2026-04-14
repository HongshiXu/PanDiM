import os

import torch
import torch.nn as nn

from .PanDiM_modules import DiM, FinalLayer, PatchEmbedding, TimeEncoding, default, exists

py_path = os.path.abspath(__file__)
file_dir = os.path.dirname(py_path)


class PanDiM(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        image_size=128,
        patch_size=16,
        inner_channel=8,
        noise_level_channel=128,
        lms_channel=8,
        pan_channel=1,
        time_hidden_ratio=4,
        num_dit_layers=6,
        num_heads=16,
        mlp_ratio=4,
        with_noise_level_emb=True,
        self_condition=False,
        use_gated_block=False,
        all_gated=False,
        local_type="resblock",
        use_channel_swap=False,
        use_pure_mamba=False,
        gate_type="spatial",
        use_cond_mamba=False,
        neck_type="hybrid",
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.self_condition = self_condition
        self.use_cond_mamba = use_cond_mamba

        if with_noise_level_emb:
            self.noise_level_emb = TimeEncoding(noise_level_channel)
        else:
            noise_level_channel = None
            self.noise_level_emb = None

        if self_condition:
            in_channel += out_channel

        self.prev_conv = nn.Conv2d(in_channel, inner_channel, 1, 1)
        self.noise_embedding = PatchEmbedding(
            img_size=image_size,
            patch_size=patch_size,
            inner_channel=inner_channel,
        )
        self.dit = DiM(
            hidden_channels=inner_channel,
            img_size=image_size,
            patch_size=patch_size,
            ms_channels=lms_channel,
            pan_channels=pan_channel,
            noise_level_emb_dim=noise_level_channel,
            time_hidden_ratio=time_hidden_ratio,
            num_layers=num_dit_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_gated_block=use_gated_block,
            all_gated=all_gated,
            local_type=local_type,
            use_channel_swap=use_channel_swap,
            use_pure_mamba=use_pure_mamba,
            gate_type=gate_type,
            use_cond_mamba=use_cond_mamba,
            neck_type=neck_type,
        )
        self.final_layer = FinalLayer(
            ms_channels=lms_channel,
            pan_channels=pan_channel,
            img_size=image_size,
            hidden_channels=inner_channel,
            patch_size=patch_size,
            out_channels=out_channel,
            num_heads=num_heads,
        )

    def forward(self, x, time, cond, self_cond=None, current_iter=None, lock_threshold=16000):
        if self.self_condition:
            self_cond = default(self_cond, x)
            x = torch.cat([x, self_cond], dim=1)

        x = self.prev_conv(x)
        x = self.noise_embedding(x)
        t = self.noise_level_emb(time) if exists(self.noise_level_emb) else None
        x = self.dit(x, t, cond, current_iter=current_iter, lock_threshold=lock_threshold)
        x = self.final_layer(x, cond)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "zero_conv" in name or "hyper_" in name or "posterior_global_proj" in name:
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if "ssm_time_mlp" in name or "zero_conv" in name or "hyper_" in name or "posterior_global_proj" in name:
                    continue
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
