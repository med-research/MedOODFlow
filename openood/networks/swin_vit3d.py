from typing import Any, Sequence

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep


class SwinViT3D(SwinTransformer):
    """SwinViT3D built on monai.networks.nets.SwinTransformer with feature
    extraction support.

    Forward returns:
    - logits by default
    - (logits, feature) if return_feature=True, where feature is the final
    stage output (pre-logits)
    - (logits, feature_list) if return_feature_list=True, where feature_list
    is a list of features from all stages
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        patch_size: Sequence[int] | int = (2, 2, 2),
        window_size: Sequence[int] | int = (7, 7, 7),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        embed_dim: int = 24,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = 'merging',
        use_v2: bool = False,
        **kwargs: Any,
    ) -> None:
        window_size = ensure_tuple_rep(window_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)

        super().__init__(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )

        self.num_classes = num_classes
        if spatial_dims == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif spatial_dims == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise ValueError(
                f'spatial_dims must be 2 or 3, got {spatial_dims}')

        self.classification_head = nn.Linear(self.num_features * 2,
                                             num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_feature: bool = False,
        return_feature_list: bool = False,
    ):
        # Get features from all stages:
        # [x0_out, x1_out, x2_out, x3_out, x4_out]
        stage_features = super().forward(x, normalize=True)

        # Use the final stage feature for classification
        final_feat = self.avg_pool(stage_features[-1])
        feature = final_feat.view(final_feat.size(0), -1)
        logits_cls = self.classification_head(feature)

        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, stage_features
        return logits_cls

    def forward_threshold(self, x: torch.Tensor,
                          threshold: float) -> torch.Tensor:
        """Forward pass with feature clipping at threshold."""
        stage_features = super().forward(x, normalize=True)
        final_feat = self.avg_pool(stage_features[-1])
        final_feat = final_feat.clip(max=threshold)
        feature = final_feat.view(final_feat.size(0), -1)
        logits_cls = self.classification_head(feature)
        return logits_cls

    def get_fc(self):
        """Get fully connected layer weights and biases."""
        fc = self.classification_head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        """Get the classification head layer."""
        return self.classification_head
