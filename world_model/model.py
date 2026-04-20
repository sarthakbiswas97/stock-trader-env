"""Neural market dynamics model — V-M-C architecture.

Vision (Encoder): Sequence of OHLCV features → latent vector
Memory (Dynamics): GRU that maintains hidden state, predicts next latent
Decoder: Latent → predicted OHLCV features (returns space)

The model learns market dynamics from historical data and generates
realistic, stochastic market sequences. Agent actions are an input,
so the model learns market impact implicitly.

Inspired by Ha & Schmidhuber's World Models, adapted for financial
time series with Mixture Density Network output for multi-modal
regime transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.data import INPUT_DIM, N_FEATURES, N_PRICE_FEATURES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Encoder (V) — Sequence → Latent
# ---------------------------------------------------------------------------

class MarketEncoder(nn.Module):
    """1D-CNN encoder that compresses a sequence of market features into a latent vector.

    Input: (batch, seq_len, input_dim)
    Output: (batch, latent_dim)
    """

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
        """Encode sequence to latent vector.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, latent_dim)
        """
        # Conv1d expects (batch, channels, length)
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = h.flatten(1)
        h = self.dropout(h)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Dynamics (M) — GRU + MDN
# ---------------------------------------------------------------------------

class MarketDynamics(nn.Module):
    """GRU-based dynamics model with Mixture Density Network output.

    Takes the latent state and produces parameters for a mixture of
    Gaussians over the next day's market features. This captures
    multi-modal market behavior (bull/bear/sideways regimes).

    Input: (batch, latent_dim + action_dim)
    Output: MDN parameters (pi, mu, sigma) for n_gaussians components
    """

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
        """Predict next state distribution.

        Args:
            latent: (batch, latent_dim)
            action: (batch, 1) — encoded action
            hidden: (n_layers, batch, gru_hidden_dim) or None

        Returns:
            pi: (batch, n_gaussians) — mixture weights (log-softmax)
            mu: (batch, n_gaussians, n_price_features) — means
            sigma: (batch, n_gaussians, n_price_features) — std devs
            hidden: (n_layers, batch, gru_hidden_dim) — new hidden state
        """
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
        """Sample from the mixture distribution.

        Args:
            pi: (batch, n_gaussians) — log mixture weights
            mu: (batch, n_gaussians, n_features) — means
            sigma: (batch, n_gaussians, n_features) — std devs
            temperature: Sampling temperature (higher = more random)

        Returns:
            (batch, n_features) — sampled next-day features
        """
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


# ---------------------------------------------------------------------------
# Decoder — Latent → Features (for reconstructing full feature vector)
# ---------------------------------------------------------------------------

class MarketDecoder(nn.Module):
    """Decodes latent vector back to full feature vector.

    Used for reconstruction loss during training.
    At inference time, we use MDN samples directly for price features
    and this decoder for derived features.
    """

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
        """Decode latent to feature vector.

        Args:
            latent: (batch, latent_dim)
        Returns:
            (batch, n_features)
        """
        return self.net(latent)


# ---------------------------------------------------------------------------
# Full World Model
# ---------------------------------------------------------------------------

class MarketWorldModel(nn.Module):
    """Complete V-M-C world model for market dynamics.

    Combines encoder, dynamics (GRU+MDN), and decoder into a single
    model that can:
    1. Encode a market history into a latent state
    2. Predict the next day's market distribution given an action
    3. Sample realistic next-day OHLCV features

    Total params: ~900K-1.2M depending on config.
    """

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
        """Forward pass for training.

        Args:
            sequence: (batch, seq_len, input_dim) — market history
            action: (batch, 1) — action for next step. Defaults to 0 (HOLD).

        Returns:
            latent: (batch, latent_dim) — encoded state
            pi, mu, sigma: MDN parameters for next-day prediction
            reconstruction: (batch, n_features) — decoded features
        """
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
        """Predict and sample next day's features.

        Args:
            sequence: (batch, seq_len, input_dim) — market history
            action: (batch, 1) — agent's action
            hidden: GRU hidden state (for sequential generation)
            temperature: Sampling temperature

        Returns:
            next_features: (batch, n_price_features) — sampled next-day features
            hidden: Updated GRU hidden state
        """
        latent = self.encoder(sequence)
        pi, mu, sigma, hidden = self.dynamics(latent, action, hidden)
        next_features = self.dynamics.sample(pi, mu, sigma, temperature)
        return next_features, hidden

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mdn_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for Mixture Density Network.

    Args:
        pi: (batch, n_gaussians) — log mixture weights
        mu: (batch, n_gaussians, n_features) — means
        sigma: (batch, n_gaussians, n_features) — std devs
        target: (batch, n_features) — true next-day features
    """
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
    """MSE reconstruction loss on the last timestep's features.

    Args:
        predicted: (batch, n_features) — decoder output
        sequence: (batch, seq_len, input_dim) — input sequence
    """
    target = sequence[:, -1, :N_FEATURES]
    return F.mse_loss(predicted, target)
