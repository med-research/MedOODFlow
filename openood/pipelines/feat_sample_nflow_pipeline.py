import os

import numpy as np
import torch

from openood.networks import get_network
from openood.utils import setup_logger, comm


class FeatSampleNormalizingFlowPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def sample_from_flow(self, net, num_samples, save_path):
        """Generate samples using the normalizing flow model.

        Args:
            net: Network containing the normalizing flow model
            num_samples: Number of samples to generate
            save_path: Path to save the generated samples
        """
        nflow = net['nflow']
        # feat_agg = net.get('feat_agg', nn.Identity())

        print(f'\nGenerating {num_samples} samples from normalizing flow...')

        # Generate samples using nflow.sample
        with torch.no_grad():
            x, _ = nflow.sample(num_samples)

        # If there's a feature aggregator in reverse order (optional)
        # x = feat_agg(x)  # Uncomment if needed

        np.savez(save_path, feat_list=x.cpu().numpy())

        print(f'Samples saved to {save_path}')
        return x

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set deterministic behavior
        comm.set_deterministic(self.config.seed,
                               self.config.nondeterministic_operators)

        # init network
        net = get_network(self.config.network)

        # Setup save directory
        num_samples = self.config.pipeline.num_samples
        save_name = self.config.pipeline.save_name
        if not save_name.endswith('.npz'):
            save_name += '.npz'
        save_path = os.path.join(self.config.output_dir, save_name)

        # Generate samples
        self.sample_from_flow(net, num_samples, save_path)
