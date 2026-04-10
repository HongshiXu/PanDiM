import torch
import torch.nn as nn

from models.PanVIM_modules import DiMBackbone, FinalLayer, PatchEmbedding, TimeEncoding, default, exists


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
        num_dim_layers=6,
        num_heads=16,
        mlp_ratio=4,
        with_noise_level_emb=True,
        self_condition=False,
        neck_type="hybrid",
    ):
        super().__init__()
        self.self_condition = self_condition

        if with_noise_level_emb:
            self.noise_level_emb = TimeEncoding(noise_level_channel)
        else:
            noise_level_channel = None
            self.noise_level_emb = None

        if self_condition:
            in_channel += out_channel

        self.input_proj = nn.Conv2d(in_channel, inner_channel, 1, 1)
        self.vim_embedding = PatchEmbedding(
            img_size=image_size,
            patch_size=patch_size,
            inner_channel=inner_channel,
        )
        self.dim_backbone = DiMBackbone(
            hidden_channels=inner_channel,
            img_size=image_size,
            patch_size=patch_size,
            ms_channels=lms_channel,
            pan_channels=pan_channel,
            noise_level_emb_dim=noise_level_channel,
            time_hidden_ratio=time_hidden_ratio,
            num_layers=num_dim_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
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
        del current_iter, lock_threshold
        if self.self_condition:
            self_cond = default(self_cond, x)
            x = torch.cat([x, self_cond], dim=1)
        x = self.input_proj(x)
        x = self.vim_embedding(x)
        time_token = self.noise_level_emb(time) if exists(self.noise_level_emb) else None
        x = self.dim_backbone(x, time_token, cond)
        return self.final_layer(x, cond)

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "param_head" in name:
                    continue
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if "ssm_time_mlp" in name or "param_head" in name:
                    continue
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
