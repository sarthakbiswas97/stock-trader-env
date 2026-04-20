"""Tests for the neural market world model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from world_model.data import (
    N_FEATURES,
    N_PRICE_FEATURES,
    MarketSequenceDataset,
    features_to_ohlcv,
    ohlcv_to_features,
)
from world_model.model import (
    CausalTransformerWorldModel,
    MarketDecoder,
    MarketDynamics,
    MarketEncoder,
    MarketWorldModel,
    TransformerConfig,
    WorldModelConfig,
    mdn_loss,
    reconstruction_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return WorldModelConfig(seq_len=20)


@pytest.fixture
def model(config):
    return MarketWorldModel(config)


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    n_days = 100
    rng = np.random.default_rng(42)
    close = 1000.0 + np.cumsum(rng.normal(0, 10, n_days))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days),
        "open": close + rng.normal(0, 2, n_days),
        "high": close + np.abs(rng.normal(5, 3, n_days)),
        "low": close - np.abs(rng.normal(5, 3, n_days)),
        "close": close,
        "volume": rng.integers(100_000, 10_000_000, n_days),
    })


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------

class TestOhlcvToFeatures:
    def test_output_shape(self, sample_ohlcv):
        features = ohlcv_to_features(sample_ohlcv)
        assert features.shape == (len(sample_ohlcv) - 1, N_FEATURES)

    def test_returns_are_bounded(self, sample_ohlcv):
        features = ohlcv_to_features(sample_ohlcv)
        assert np.all(features >= -0.5)
        assert np.all(features <= 0.5)

    def test_dtype_is_float32(self, sample_ohlcv):
        features = ohlcv_to_features(sample_ohlcv)
        assert features.dtype == np.float32

    def test_handles_zero_volume(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [100] * 10,
            "volume": [0] * 10,
        })
        features = ohlcv_to_features(df)
        assert not np.any(np.isnan(features))


class TestFeaturesToOhlcv:
    def test_reconstruction_keys(self):
        features = np.array([0.01, 0.02, -0.01, 0.005, 0.1])
        result = features_to_ohlcv(features, 1000.0, 1_000_000.0)
        assert set(result.keys()) == {"open", "high", "low", "close", "volume"}

    def test_ohlc_consistency(self):
        features = np.array([0.01, 0.03, -0.02, 0.005, 0.1])
        result = features_to_ohlcv(features, 1000.0, 1_000_000.0)
        assert result["high"] >= result["open"]
        assert result["high"] >= result["close"]
        assert result["low"] <= result["open"]
        assert result["low"] <= result["close"]
        assert result["low"] > 0

    def test_volume_positive(self):
        features = np.array([0.0, 0.0, 0.0, 0.0, -5.0])
        result = features_to_ohlcv(features, 1000.0, 1_000_000.0)
        assert result["volume"] >= 100


class TestMarketSequenceDataset:
    def test_creates_sequences(self, sample_ohlcv):
        dataset = MarketSequenceDataset({"TEST": sample_ohlcv}, seq_len=20)
        assert len(dataset) > 0

    def test_sequence_shape(self, sample_ohlcv):
        dataset = MarketSequenceDataset({"TEST": sample_ohlcv}, seq_len=20)
        seq, target = dataset[0]
        assert seq.shape == (20, N_FEATURES)
        assert target.shape == (N_PRICE_FEATURES,)

    def test_skips_short_symbols(self):
        short_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [100] * 10,
            "volume": [1_000_000] * 10,
        })
        dataset = MarketSequenceDataset({"SHORT": short_df}, seq_len=50)
        assert len(dataset) == 0

    def test_compute_stats(self, sample_ohlcv):
        dataset = MarketSequenceDataset({"TEST": sample_ohlcv}, seq_len=20)
        stats = dataset.compute_stats()
        assert stats.feature_means.shape == (N_FEATURES,)
        assert stats.feature_stds.shape == (N_FEATURES,)
        assert np.all(stats.feature_stds > 0)


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestMarketEncoder:
    def test_output_shape(self, config):
        encoder = MarketEncoder(config)
        x = torch.randn(4, config.seq_len, config.n_features)
        out = encoder(x)
        assert out.shape == (4, config.latent_dim)


class TestMarketDynamics:
    def test_forward_shapes(self, config):
        dynamics = MarketDynamics(config)
        latent = torch.randn(4, config.latent_dim)
        pi, mu, sigma, hidden = dynamics(latent)

        assert pi.shape == (4, config.n_gaussians)
        assert mu.shape == (4, config.n_gaussians, config.n_price_features)
        assert sigma.shape == (4, config.n_gaussians, config.n_price_features)
        assert hidden.shape == (config.gru_layers, 4, config.gru_hidden_dim)

    def test_pi_sums_to_one(self, config):
        dynamics = MarketDynamics(config)
        latent = torch.randn(4, config.latent_dim)
        pi, _, _, _ = dynamics(latent)
        probs = torch.exp(pi)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_sigma_positive(self, config):
        dynamics = MarketDynamics(config)
        latent = torch.randn(4, config.latent_dim)
        _, _, sigma, _ = dynamics(latent)
        assert torch.all(sigma > 0)

    def test_sample_shape(self, config):
        dynamics = MarketDynamics(config)
        latent = torch.randn(4, config.latent_dim)
        pi, mu, sigma, _ = dynamics(latent)
        sample = dynamics.sample(pi, mu, sigma)
        assert sample.shape == (4, config.n_price_features)


class TestMarketDecoder:
    def test_output_shape(self, config):
        decoder = MarketDecoder(config)
        latent = torch.randn(4, config.latent_dim)
        out = decoder(latent)
        assert out.shape == (4, config.n_features)


class TestMarketWorldModel:
    def test_forward_shapes(self, model, config):
        seq = torch.randn(4, config.seq_len, config.n_features)
        latent, pi, mu, sigma, recon = model(seq)

        assert latent.shape == (4, config.latent_dim)
        assert pi.shape == (4, config.n_gaussians)
        assert mu.shape == (4, config.n_gaussians, config.n_price_features)
        assert recon.shape == (4, config.n_features)

    def test_predict_next(self, model, config):
        seq = torch.randn(2, config.seq_len, config.n_features)
        features, hidden = model.predict_next(seq)

        assert features.shape == (2, config.n_price_features)
        assert hidden.shape == (config.gru_layers, 2, config.gru_hidden_dim)

    def test_temperature_affects_variance(self, model, config):
        torch.manual_seed(42)
        seq = torch.randn(1, config.seq_len, config.n_features)

        samples_low = [model.predict_next(seq, temperature=0.1)[0] for _ in range(20)]
        samples_high = [model.predict_next(seq, temperature=2.0)[0] for _ in range(20)]

        var_low = torch.stack(samples_low).var(dim=0).mean()
        var_high = torch.stack(samples_high).var(dim=0).mean()
        assert var_high > var_low

    def test_param_count(self, model):
        n = model.count_parameters()
        assert 50_000 < n < 500_000


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def test_mdn_loss_finite(self):
        pi = torch.log_softmax(torch.randn(4, 3), dim=-1)
        mu = torch.randn(4, 3, 5)
        sigma = torch.ones(4, 3, 5) * 0.1
        target = torch.randn(4, 5)
        loss = mdn_loss(pi, mu, sigma, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_mdn_loss_lower_for_closer_target(self):
        pi = torch.log_softmax(torch.zeros(4, 3), dim=-1)
        mu = torch.zeros(4, 3, 5)
        sigma = torch.ones(4, 3, 5) * 0.1

        close_target = torch.zeros(4, 5)
        far_target = torch.ones(4, 5) * 10

        loss_close = mdn_loss(pi, mu, sigma, close_target)
        loss_far = mdn_loss(pi, mu, sigma, far_target)
        assert loss_close < loss_far

    def test_reconstruction_loss_finite(self):
        predicted = torch.randn(4, N_FEATURES)
        sequence = torch.randn(4, 20, N_FEATURES)
        loss = reconstruction_loss(predicted, sequence)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_mdn_loss_causal_shape(self):
        """MDN loss works with multi-position (causal) output."""
        pi = torch.log_softmax(torch.randn(4, 20, 3), dim=-1)
        mu = torch.randn(4, 20, 3, 5)
        sigma = torch.ones(4, 20, 3, 5) * 0.1
        target = torch.randn(4, 20, 5)
        loss = mdn_loss(pi, mu, sigma, target)
        assert torch.isfinite(loss)


class TestCausalTransformer:
    @pytest.fixture
    def t_config(self):
        return TransformerConfig(seq_len=20)

    @pytest.fixture
    def t_model(self, t_config):
        return CausalTransformerWorldModel(t_config)

    def test_forward_shapes(self, t_model, t_config):
        seq = torch.randn(2, t_config.seq_len, t_config.n_features)
        pi, mu, sigma = t_model(seq)
        assert pi.shape == (2, t_config.seq_len, t_config.n_gaussians)
        assert mu.shape == (2, t_config.seq_len, t_config.n_gaussians, t_config.n_price_features)
        assert sigma.shape == (2, t_config.seq_len, t_config.n_gaussians, t_config.n_price_features)

    def test_predict_next(self, t_model, t_config):
        seq = torch.randn(2, t_config.seq_len, t_config.n_features)
        features, hidden = t_model.predict_next(seq)
        assert features.shape == (2, t_config.n_price_features)
        assert hidden is None

    def test_causal_masking(self, t_model, t_config):
        """Changing future tokens should not affect past predictions."""
        t_model.eval()
        seq1 = torch.randn(1, t_config.seq_len, t_config.n_features)
        seq2 = seq1.clone()
        seq2[0, -1, :] = 999.0

        with torch.no_grad():
            pi1, _, _ = t_model(seq1)
            pi2, _, _ = t_model(seq2)

        assert torch.allclose(pi1[0, 0], pi2[0, 0], atol=1e-4)

    def test_param_count(self, t_model):
        n = t_model.count_parameters()
        assert 500_000 < n < 2_000_000
