import normflows as nf
import torch
from normflows.flows import Flow


class ClampedMLP(nf.nets.MLP):
    def __init__(self, clamp: float, method: str = 'HARD', **kwargs):
        super().__init__(**kwargs)
        self.clamp = clamp
        clamp_activations = {
            'HARD': lambda u: torch.clamp(u, min=-clamp, max=clamp),
            'TANH': lambda u: clamp * torch.tanh(u / clamp),
            'SIGMOID': lambda u: clamp * 2. * (torch.sigmoid(u / clamp) - 0.5),
            'ATAN': lambda u: clamp * 0.636 * torch.atan(u / clamp)
        }
        if not method or not clamp:
            self.clamp_fn = lambda u: u
        elif method in clamp_activations:
            self.clamp_fn = clamp_activations[method]
        else:
            raise ValueError(f'Unknown clamp method: {method}')

    def forward(self, x):
        x = super().forward(x)
        x = self.clamp_fn(x)
        return x


class L2Norm(Flow):
    def __init__(self, eps=1e-12, adjust_volume=False):
        """
        L2 normalization flow, i.e., z / ||z||_2
        Args:
            eps: small value to avoid division by zero. Default: 1e-12
            adjust_volume: if True, the log determinant of the Jacobian
             is returned. When False, the module acts as a normalization
             layer, and the log determinant to account for the volume
             change is not computed. Default: False
        """
        super().__init__()
        self.adjust_volume = adjust_volume
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        # Assuming z has shape (batch_size, channels, n1, n2, ...)
        norms = torch.norm(z, p=2, dim=1, keepdim=True)
        z_ = z / (norms + self.eps)
        if self.adjust_volume:
            n = z.shape[1] + 1
            log_det = (torch.log(self.eps) -
                       n * torch.log(norms + self.eps)).squeeze()
        else:
            log_det = 0
        return z_, log_det


def get_normalizing_flow(network_config):
    latent_size = network_config.latent_size
    hidden_size = network_config.hidden_size
    n_flows = network_config.n_flows
    l2_normalize = network_config.l2_normalize
    flow_arch = network_config.arch
    flow_archs = {
        'RealNVP': _get_realnvp,
        'Glow': _get_glow,
        'NSF': _get_nsf,
        'ResFlow': _get_resflow
    }

    if flow_arch not in flow_archs:
        raise ValueError(f'Unknown flow architecture: {flow_arch}')

    flows = flow_archs[flow_arch](latent_size, hidden_size, n_flows,
                                  network_config)
    if l2_normalize:
        flows += [L2Norm(adjust_volume=False)]

    q0 = nf.distributions.DiagGaussian(latent_size)
    # Note that in inverse method which is applied to the features
    # extracted from the backbone, the order of the flows is reversed.
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm


def _get_realnvp(latent_size, hidden_size, n_flows, network_config):
    """
    RealNVP: Density Estimation using Real NVP,
    [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)
    """
    hidden_layers = network_config.hidden_layers
    clamp_value = network_config.clamp_value
    clamp_method = network_config.clamp_method
    clamp_t = network_config.clamp_t
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    layers = [latent_size] + [hidden_size] * hidden_layers + [latent_size]
    for i in range(n_flows):
        s = ClampedMLP(clamp=clamp_value,
                       method=clamp_method,
                       layers=layers,
                       init_zeros=True)
        t = ClampedMLP(clamp=clamp_value,
                       method=clamp_method if clamp_t else None,
                       layers=layers,
                       init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        # ActNorm z-score normalizes (zero mean and unit variance) the input,
        # using two learnable parameters "mean" and "std" which are initialized
        # by the statistics of the first batch.
        # It's not in the original RealNVP paper, but commonly used in practice
        flows += [nf.flows.ActNorm(latent_size)]
    return flows


def _get_glow(latent_size, hidden_size, n_flows, network_config):
    """
    Glow: Generative Flow with Invertible 1×1 Convolutions,
    [arXiv: 1807.03039](https://arxiv.org/abs/1807.03039)
    """
    hidden_layers = network_config.hidden_layers
    clamp_value = network_config.clamp_value
    clamp_method = network_config.clamp_method
    clamp_t = network_config.clamp_t
    flows = []
    b = torch.zeros(latent_size)
    b[:latent_size // 2] = 1
    layers = [latent_size] + [hidden_size] * hidden_layers + [latent_size]
    for i in range(n_flows):
        s = ClampedMLP(clamp=clamp_value,
                       method=clamp_method,
                       layers=layers,
                       init_zeros=True)
        t = ClampedMLP(clamp=clamp_value,
                       method=clamp_method if clamp_t else None,
                       layers=layers,
                       init_zeros=True)
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        flows += [nf.flows.mixing.InvertibleAffine(latent_size)]
        flows += [nf.flows.ActNorm(latent_size)]
    return flows


def _get_nsf(latent_size, hidden_size, n_flows, network_config):
    """
    NSF: Neural Spline Flows,
    [arXiv: 1906.04032](https://arxiv.org/abs/1906.04032)
    """
    num_blocks = network_config.num_blocks
    flows = []
    for i in range(n_flows):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                num_input_channels=latent_size,
                num_blocks=num_blocks,
                num_hidden_channels=hidden_size)
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]
    return flows


def _get_resflow(latent_size, hidden_size, n_flows, network_config):
    """
    ResFlow: Residual Flows for Invertible Generative Modeling,
    [arXiv: 1906.02735](https://arxiv.org/abs/1906.02735)
    """
    hidden_layers = network_config.hidden_layers
    lipschitz_const = network_config.lipschitz_const or 0.97
    reduce_memory = bool(network_config.reduce_memory)
    flows = []
    channels = [latent_size] + [hidden_size] * hidden_layers + [latent_size]
    for i in range(n_flows):
        net = nf.nets.LipschitzMLP(channels=channels,
                                   lipschitz_const=lipschitz_const,
                                   init_zeros=True)
        flows += [nf.flows.Residual(net, reduce_memory=reduce_memory)]
        flows += [nf.flows.ActNorm(latent_size)]
    return flows
