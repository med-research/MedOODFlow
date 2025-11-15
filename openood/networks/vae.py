import torch
import torch.nn as nn


class FeatureL2Norm(nn.Module):
    """Applies L2 normalization over feature dimension: x / ||x||_2."""
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, x):
        # x: (batch, feat_dim)
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norms + self.eps)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, h_dim),
                 nn.BatchNorm1d(h_dim),
                 nn.ReLU()])
            prev_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.network(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend(
                [nn.Linear(prev_dim, h_dim),
                 nn.BatchNorm1d(h_dim),
                 nn.ReLU()])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 latent_dim,
                 l2_normalize: bool = False):
        """Variational Autoencoder for density estimation using ELBO.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.l2norm = FeatureL2Norm() if l2_normalize else nn.Identity()

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through VAE.

        Returns reconstruction, mu, and logvar.
        """
        # L2-normalize input features if enabled
        x = self.l2norm(x)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def _compute_elbo(x, recon, mu, logvar, beta=1.0):
        """Compute negative ELBO loss.

        ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

        Args:
            x: Input data
            recon: Reconstructed data
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence term (beta-VAE)

        Returns:
            Negative ELBO (total loss), reconstruction loss, KL divergence
        """
        # Reconstruction loss (negative log likelihood) - per sample
        recon_loss = nn.functional.mse_loss(recon, x,
                                            reduction='none').sum(dim=1)

        # KL div per sample: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Negative ELBO per sample
        neg_elbo = recon_loss + beta * kl_div
        return neg_elbo, recon_loss, kl_div

    def forward_elbo(self, x, beta=1.0):
        """Compute negative ELBO loss for training (mean over batch).

        This is the loss function to minimize during training.
        Also returns mean reconstruction and KL losses for logging.

        Args:
            x: Input data batch
            beta: Weight for KL divergence term (beta-VAE)

        Returns:
            Tuple of (mean negative ELBO, mean recon loss, mean KL loss)
        """
        recon, mu, logvar = self.forward(x)
        neg_elbo, recon_loss, kl_div = self._compute_elbo(x,
                                                          recon,
                                                          mu,
                                                          logvar,
                                                          beta=beta)
        return torch.mean(neg_elbo), torch.mean(recon_loss), torch.mean(kl_div)

    def log_prob(self, x, beta=1.0):
        """
        Estimate log probability using ELBO as a proxy.
        Returns per-sample ELBO, negated so higher values = higher density.

        This matches the normalizing flow interface where log_prob returns
        unreduced scores.

        Args:
            x: Input data batch
            beta: Weight for KL divergence term (beta-VAE)

        Returns:
            Per-sample ELBO (higher is better for ID data)
        """
        recon, mu, logvar = self.forward(x)
        neg_elbo, _, _ = self._compute_elbo(x, recon, mu, logvar, beta=beta)

        # Return negative of neg_elbo per sample (so higher = more ID-like)
        return -neg_elbo

    def sample(self, num_samples, device='cuda'):
        """Sample from the VAE by sampling from prior p(z) and decoding."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples


def get_vae(network_config):
    """Factory function to create VAE model.

    Expected config attributes:
        - input_dim or latent_size: Input feature dimension
        - hidden_dims: List of hidden dimensions
        - latent_dim: Latent space dimension
        - l2_normalize: Whether to L2-normalize input features
    """
    input_dim = network_config.input_dim
    hidden_dims = network_config.hidden_dims
    latent_dim = network_config.latent_dim
    l2_normalize = network_config.l2_normalize

    vae = VAE(input_dim=input_dim,
              hidden_dims=hidden_dims,
              latent_dim=latent_dim,
              l2_normalize=l2_normalize)
    return vae
