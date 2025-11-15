from typing import Any, Sequence

import torch
from monai.networks.nets.vit import ViT


class ViT3D(ViT):
    """ViT3D built on monai.networks.nets.ViT with feature extraction support.

    Forward returns:
    - logits by default
    - (logits, feature) if return_feature=True, where feature is CLS token at
    the last block (pre-logits)
    - (logits, feature_list) if return_feature_list=True, where feature_list
    is a list of CLS tokens from all blocks
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        img_size: Sequence[int] | int = (96, 96, 96),
        patch_size: Sequence[int] | int = (16, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = 'conv',
        pos_embed_type: str = 'learnable',
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation='',  # no Tanh for logits
            qkv_bias=qkv_bias,
            **kwargs,
        )

    @staticmethod
    def _cls_pool(tokens):  # (B, 1+N, hidden_size)
        # Get the CLS token feature
        return tokens[:, 0, :]  # (B, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        return_feature: bool = False,
        return_feature_list: bool = False,
    ):
        x, blocks_feats = super().forward(x)
        final_feat = self._cls_pool(self.norm(blocks_feats[-1]))
        if return_feature:
            return x, final_feat
        elif return_feature_list:
            feature_list = [
                self._cls_pool(feat) for feat in blocks_feats[:-1]
            ] + [final_feat]
            return x, feature_list
        return x

    def forward_threshold(self, x: torch.Tensor,
                          threshold: float) -> torch.Tensor:
        _, blocks_feats = super().forward(x)
        final_feat = self._cls_pool(self.norm(blocks_feats[-1]))
        final_feat = final_feat.clip(max=threshold)
        logits_cls = self.classification_head(final_feat)
        return logits_cls

    def get_fc(self):
        fc = self.classification_head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.classification_head
