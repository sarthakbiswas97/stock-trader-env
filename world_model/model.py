"""Neural market dynamics model (V-M-C architecture).

Encoder: OHLCV sequence → latent vector
Dynamics: GRU + MDN predicting next-day distribution (3 Gaussian mixture)
Decoder: latent → reconstructed features (training regularization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.data import INPUT_DIM, N_FEATURES, N_PRICE_FEATURES



@dataclass(frozen=True)
class WorldModelConfig:
    """Configuration for the neural market model."""

    # Encoder
    latent_dim: int = 64
    encoder_channels: list[int] = (32, 64, 64)
    encoder_kernel_size: int = 3

    # Dynamics (GRU)
    gru_hidden_dim: int = 128
    gru_layers: int = 1

    # MDN (Mixture Density Network) output
    n_gaussians: int = 3

    # Input
    input_dim: int = INPUT_DIM
    n_features: int = N_FEATURES
    n_price_features: int = N_PRICE_FEATURES
    seq_len: int = 50

    # Training
    dropout: float = 0.1



class MarketEncoder(nn.Module):
    """1D-CNN encoder: (batch, seq_len, input_dim) → (batch, latent_dim)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        channels = list(config.encoder_channels)
        layers = []
        in_channels = config.input_dim

        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, config.encoder_kernel_size, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.MaxPool1d(2),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        # After 3 rounds of MaxPool1d(2): seq_len=50 → 25 → 12 → 6
        conv_out_len = config.seq_len
        for _ in channels:
            conv_out_len = conv_out_len // 2

        self.fc = nn.Linear(channels[-1] * conv_out_len, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects (batch, channels, length)
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = h.flatten(1)
        h = self.dropout(h)
        return self.fc(h)



class MarketDynamics(nn.Module):
    """GRU + MDN: (latent, action) → mixture distribution over next-day features."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.latent_dim + 1,  # +1 for action
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )

        # MDN output heads
        n_out = config.n_price_features
        n_g = config.n_gaussians

        self.pi_head = nn.Linear(config.gru_hidden_dim, n_g)
        self.mu_head = nn.Linear(config.gru_hidden_dim, n_g * n_out)
        self.log_sigma_head = nn.Linear(config.gru_hidden_dim, n_g * n_out)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pi, mu, sigma, hidden) — MDN parameters + new GRU state."""
        n_out = self.config.n_price_features
        n_g = self.config.n_gaussians

        # Combine latent + action
        x = torch.cat([latent, action], dim=-1)
        x = x.unsqueeze(1)  # (batch, 1, latent_dim + 1)

        gru_out, hidden = self.gru(x, hidden)
        h = gru_out.squeeze(1)  # (batch, gru_hidden_dim)

        # MDN parameters
        pi = F.log_softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, n_g, n_out)
        log_sigma = self.log_sigma_head(h)
        sigma = torch.exp(log_sigma).view(-1, n_g, n_out)
        sigma = torch.clamp(sigma, min=1e-6, max=0.5)

        return pi, mu, sigma, hidden

    def sample(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample next-day features from the MDN mixture distribution."""
        # Select mixture component
        pi_temp = pi / temperature
        pi_probs = torch.softmax(pi_temp, dim=-1)
        component = torch.multinomial(pi_probs, 1).squeeze(-1)

        # Get mu and sigma for selected component
        batch_idx = torch.arange(mu.size(0), device=mu.device)
        selected_mu = mu[batch_idx, component]
        selected_sigma = sigma[batch_idx, component]

        # Sample from Gaussian
        eps = torch.randn_like(selected_mu) * temperature
        return selected_mu + selected_sigma * eps



class MarketDecoder(nn.Module):
    """Decoder: latent → full feature vector (reconstruction regularization)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, config.n_features),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)



class MarketWorldModel(nn.Module):
    """Complete V-M-C world model for market dynamics (~140K params)."""

    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        self.encoder = MarketEncoder(self.config)
        self.dynamics = MarketDynamics(self.config)
        self.decoder = MarketDecoder(self.config)

    def forward(
        self,
        sequence: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward: returns (latent, pi, mu, sigma, reconstruction)."""
        batch_size = sequence.size(0)
        device = sequence.device

        if action is None:
            action = torch.zeros(batch_size, 1, device=device)

        latent = self.encoder(sequence)
        pi, mu, sigma, _ = self.dynamics(latent, action)
        reconstruction = self.decoder(latent)

        return latent, pi, mu, sigma, reconstruction

    def predict_next(
        self,
        sequence: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference: sample next-day features and return updated hidden state."""
        latent = self.encoder(sequence)
        pi, mu, sigma, hidden = self.dynamics(latent, action, hidden)
        next_features = self.dynamics.sample(pi, mu, sigma, temperature)
        return next_features, hidden

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def mdn_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for the Gaussian mixture."""
    target = target.unsqueeze(1)  # (batch, 1, n_features)

    # Log probability for each Gaussian component
    log_probs = -0.5 * (
        math.log(2 * math.pi)
        + 2 * torch.log(sigma)
        + ((target - mu) / sigma) ** 2
    )
    # Sum over features, add mixture weight
    log_component_probs = log_probs.sum(dim=-1) + pi  # (batch, n_gaussians)

    # Log-sum-exp over components
    log_likelihood = torch.logsumexp(log_component_probs, dim=-1)

    return -log_likelihood.mean()


def reconstruction_loss(
    predicted: torch.Tensor,
    sequence: torch.Tensor,
) -> torch.Tensor:
    """MSE between decoder output and last timestep's features."""
    target = sequence[:, -1, :N_FEATURES]
    return F.mse_loss(predicted, target)
