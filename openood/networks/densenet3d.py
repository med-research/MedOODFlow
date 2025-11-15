from typing import Any

import torch
from monai.networks.nets import DenseNet

from .pool3d import DeterministicPool3d


class DenseNet3D(DenseNet):
    """DenseNet3D for 3D medical image classification.

    Extends MONAI's DenseNet with additional forward methods for feature
    extraction and compatibility with OpenOOD framework.
    """
    def __init__(self, use_deterministic_pool=False, **kwargs):
        super(DenseNet3D, self).__init__(spatial_dims=3, **kwargs)
        # Calculate feature size from the last layer before classification
        self.feature_size = self.class_layers.out.in_features
        self.use_deterministic_pool = use_deterministic_pool

        if self.use_deterministic_pool:
            self._replace_pool_layers()

    def _replace_pool_layers(self):
        """Replace PyTorch pooling layers with deterministic equivalents."""
        # Replace initial max pool in features.pool0
        if hasattr(self.features, 'pool0'):
            original_pool = self.features.pool0
            self.features.pool0 = DeterministicPool3d(
                pool_type='max',
                kernel_size=original_pool.kernel_size,
                stride=original_pool.stride,
                padding=original_pool.padding)

        # Replace average pooling layers in transitions
        for name, module in self.features.named_children():
            if 'transition' in name and hasattr(module, 'pool'):
                original_pool = module.pool
                module.pool = DeterministicPool3d(
                    pool_type='avg',
                    kernel_size=original_pool.kernel_size,
                    stride=original_pool.stride,
                    padding=original_pool.padding)

    def forward(self,
                x: torch.Tensor,
                return_feature: bool = False,
                return_feature_list: bool = False) -> torch.Tensor:
        """Forward pass with optional feature extraction.

        Args:
            x: Input tensor
            return_feature: If True, return (logits, features)
            return_feature_list: If True, return (logits, feature_list)

        Returns:
            logits or (logits, features) or (logits, feature_list)
        """
        if return_feature_list:
            feature_list = []
            # Extract features at exactly after each pooling layer
            # (pool0, 3 last pool of transitions, final pool before cls layer)
            x_inter = x
            for name, module in self.features.named_children():
                x_inter = module(x_inter)
                if 'pool' in name or 'transition' in name:
                    feature_list.append(x_inter)

            # Get final features before classification
            features = self.class_layers.relu(x_inter)
            features = self.class_layers.pool(features)
            feature_list.append(features)
            features = self.class_layers.flatten(features)
            logits_cls = self.class_layers.out(features)

            return logits_cls, feature_list

        # Extract features
        x = self.features(x)
        x = self.class_layers.relu(x)
        x = self.class_layers.pool(x)
        features = self.class_layers.flatten(x)
        logits_cls = self.class_layers.out(features)

        if return_feature:
            return logits_cls, features
        else:
            return logits_cls

    def forward_threshold(self, x: torch.Tensor,
                          threshold: float) -> torch.Tensor:
        """Forward pass with threshold clipping on features.

        Args:
            x: Input tensor
            threshold: Maximum value for feature clipping

        Returns:
            Classification logits
        """
        x = self.features(x)
        x = self.class_layers.relu(x)
        x = self.class_layers.pool(x)
        x = x.clip(max=threshold)
        x = self.class_layers.flatten(x)
        logits_cls = self.class_layers.out(x)
        return logits_cls

    def get_fc(self):
        """Get fully connected layer weights and biases."""
        fc = self.class_layers.out
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        """Get the fully connected layer."""
        return self.class_layers.out


def DenseNet3D_121(num_classes: int,
                   in_channels: int = 1,
                   **kwargs: Any) -> DenseNet3D:
    """DenseNet-121 for 3D medical images.

    Based on: `Densely Connected Convolutional Networks
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        num_classes: Number of classes for the classification head.
        in_channels: Number of input channels.
        use_deterministic_pool: If True, use deterministic pooling layers.
        **kwargs: Additional arguments for DenseNet3D

    Returns:
        DenseNet3D-121 model
    """
    return DenseNet3D(growth_rate=32,
                      block_config=(6, 12, 24, 16),
                      init_features=64,
                      in_channels=in_channels,
                      out_channels=num_classes,
                      use_deterministic_pool=True,
                      **kwargs)


def DenseNet3D_169(num_classes: int,
                   in_channels: int = 1,
                   **kwargs: Any) -> DenseNet3D:
    """DenseNet-169 for 3D medical images.

    Based on: `Densely Connected Convolutional Networks
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        num_classes: Number of classes for the classification head.
        in_channels: Number of input channels.
        use_deterministic_pool: If True, use deterministic pooling layers.
        **kwargs: Additional arguments for DenseNet3D

    Returns:
        DenseNet3D-169 model
    """
    return DenseNet3D(growth_rate=32,
                      block_config=(6, 12, 32, 32),
                      init_features=64,
                      in_channels=in_channels,
                      out_channels=num_classes,
                      use_deterministic_pool=True,
                      **kwargs)
