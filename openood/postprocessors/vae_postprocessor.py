from typing import Any

import torch

from .base_postprocessor import BasePostprocessor


class VAEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(VAEPostprocessor, self).__init__(config)

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        """Postprocess using VAE ELBO as density estimate.

        Higher ELBO (less negative) indicates higher likelihood of being ID.
        """
        # Images input
        if data.shape[-1] > 1 and data.shape[1] in {1, 3}:
            data = torch.Tensor(data)  # fix the issue when data is MetaTensor
            output, feats = net['backbone'](data, return_feature=True)
            score = torch.softmax(output, dim=1)
            _, pred = torch.max(score, dim=1)

            if 'feat_agg' in net:
                feats = net['feat_agg'](feats)

            # Get per-sample log probability (ELBO)
            log_prob = net['vae'].log_prob(feats)
            conf = log_prob.reshape(-1).detach().cpu()

        # Feature input
        elif data.shape[-1] == 1 and data.shape[-2] == 1:
            feats = data.flatten(1)

            if 'feat_agg' in net:
                feats = net['feat_agg'](feats)

            # Get per-sample log probability (ELBO)
            log_prob = net['vae'].log_prob(feats)
            conf = log_prob.reshape(-1).detach().cpu()
            pred = torch.ones_like(conf)  # dummy predictions

        else:
            raise ValueError('Unsupported input type!')

        return pred, conf
