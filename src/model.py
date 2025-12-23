import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional VAE for 28 x 28 grayscale images (Fashion-MNIST)

    Encoder learns q_phi(z | x) = N(mu(x), diag(sigma(x)^2)).
    Decoder models p_theta(x | z) as Bernoulli with logits output.

    We output logits (not probabilities) from the decoder so we can use
    BCEWithLogits for numerical stability.
    """

    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim

        # Encoder: x -> features -> (mu, logvar)
        # Input x shape: (B, 1, 28, 28)

        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (B,32,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (B,64,7,7)
            nn.ReLU(inplace=True),
        )

        # Flatten to a vector and map to a small hidden representation
        self.enc_fc = nn.Linear(64 * 7 * 7, 128)

        # Two Heads: mean and log-variance of q(z|x)
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

        # Decoder: z -> features -> x logits
        # Outputs logits shape: (B, 1, 28, 28)

        self.dec_fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1),# -> (B,32,14,14)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size = 4, stride = 2, padding = 1), # -> (B,1,28,28)
            # No sigmoid here; we keep logits and apply sigmoid only for visualization.
        )

    def encode(self, x: torch.Tensor):
        """
        Returns parameters of q(z|x): mu, logvar
        """
        h = self.enc(x)  # (B,64,7,7)
        h = h.view(h.size(0), -1)  # (B, 64*7*7)
        h = F.relu(self.enc_fc(h))  # (B,128)
        return self.mu(h), self.logvar(h)  # each (B,z_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick:
            z = mu + sigma * eps
            eps ~ N(0, I), sigma = exp(0.5 * logvar)
        This makes sampling differentiable w.r.t. mu/logvar.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_logits(self, z: torch.Tensor):
        """
        Maps latent z to decoder output logits for x.
        """
        h = self.dec_fc(z)  # (B, 64*7*7)
        h = h.view(z.size(0), 64, 7, 7)  # (B,64,7,7)
        return self.dec(h)  # (B,1,28,28) logits

    def forward(self, x: torch.Tensor):
        """
        Full VAE forward pass:
            x -> (mu,logvar) -> z -> logits
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

def vae_loss_bce_logits(
        x: torch.Tensor,
        logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
):
    """
    Negative ELBO (minimize):
        recon(x, logits) + beta * KL(q(z|x) || p(z))

    recon: Bernoulli likelihood -> BCEWithLogits summed over pixels, mean over batch
    KL: closed form for diagonal Gaussians:
        KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )

    Returns:
        total_loss, recon_loss, kl_loss  (each scalar, batch-averaged)
    """
    # Reconstruction term: sum over pixels then mean over batch
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon = recon.view(recon.size(0), -1).sum(dim=1).mean()

    # KL divergence term
    kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
    kl = kl.sum(dim=1).mean()

    total = recon + beta * kl
    return total, recon, kl