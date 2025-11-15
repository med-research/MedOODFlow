import torch
from torch import nn
import torch.nn.functional as F


class FeatureConcatNetwork(nn.Module):
    def __init__(self, encoder, layers, processing=None):
        super(FeatureConcatNetwork, self).__init__()
        self.encoder = encoder
        self.layers = layers
        if not processing:
            self.process = nn.Identity()
            return
        processing_modules = []
        for proc in processing:
            if proc == 'avg_pool_2d':
                processing_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
            elif proc == 'avg_pool_3d':
                processing_modules.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
            elif proc == 'l2_normalize':
                processing_modules.append(Normalize(p=2, dim=1))
            else:
                raise ValueError(f'Unsupported processing: {proc}')
        if len(processing_modules) == 1:
            self.process = processing_modules[0]
        else:
            self.process = nn.Sequential(*processing_modules)

    def forward(self, x, return_feature=False):
        if not return_feature:
            return self.encoder(x, return_feature=False)
        logits_cls, features_list = self.encoder(x, return_feature_list=True)
        features_to_aggregate = [
            f for i, f in enumerate(features_list) if (i + 1) in self.layers
        ]
        concatenated_features = torch.cat(
            [self.process(f).flatten(1) for f in features_to_aggregate], dim=1)
        return logits_cls, concatenated_features


class Normalize(nn.Module):
    def __init__(self, p, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)
