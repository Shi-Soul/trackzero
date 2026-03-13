"""Tests for the MLP inverse dynamics policy."""

import numpy as np
import pytest
import torch

from trackzero.config import Config
from trackzero.policy.mlp import (
    InverseDynamicsMLP,
    MLPPolicy,
    save_checkpoint,
    load_checkpoint,
)
from trackzero.policy.train import extract_pairs, compute_normalization, train, TrainingConfig
from trackzero.sim.simulator import Simulator


@pytest.fixture
def cfg():
    return Config()


class TestInverseDynamicsMLP:
    def test_forward_shape(self):
        model = InverseDynamicsMLP()
        x = torch.randn(32, 8)
        y = model(x)
        assert y.shape == (32, 2)

    def test_single_sample(self):
        model = InverseDynamicsMLP()
        x = torch.randn(1, 8)
        y = model(x)
        assert y.shape == (1, 2)

    def test_normalization_applied(self):
        model = InverseDynamicsMLP()
        x = torch.ones(1, 8) * 100
        y1 = model(x).detach()

        # Change normalization
        model.set_normalization(np.ones(8) * 100, np.ones(8))
        y2 = model(x).detach()

        # Different normalization should give different outputs
        assert not torch.allclose(y1, y2)

    def test_configurable_architecture(self):
        model = InverseDynamicsMLP(hidden_dim=64, n_hidden=2)
        x = torch.randn(10, 8)
        y = model(x)
        assert y.shape == (10, 2)

        n_params = sum(p.numel() for p in model.parameters())
        # 8*64 + 64 + 64*64 + 64 + 64*2 + 2 = 512+64+4096+64+128+2 = 4866
        assert n_params < 5000


class TestMLPPolicy:
    def test_callable_interface(self):
        model = InverseDynamicsMLP()
        policy = MLPPolicy(model, tau_max=5.0)
        state = np.array([0.1, -0.2, 0.5, -0.3])
        next_state = np.array([0.11, -0.19, 0.45, -0.28])
        action = policy(state, next_state)
        assert action.shape == (2,)
        assert np.all(np.abs(action) <= 5.0)

    def test_output_clipping(self):
        """Policy should clip outputs to tau_max."""
        model = InverseDynamicsMLP()
        # Set weights to produce large outputs
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(10.0)
        policy = MLPPolicy(model, tau_max=5.0)
        action = policy(np.zeros(4), np.zeros(4))
        assert np.all(np.abs(action) <= 5.0)


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        model = InverseDynamicsMLP(hidden_dim=64, n_hidden=2)
        model.set_normalization(np.random.randn(8), np.abs(np.random.randn(8)) + 0.1)

        path = tmp_path / "model.pt"
        save_checkpoint(model, path)
        loaded = load_checkpoint(path)

        x = torch.randn(5, 8)
        with torch.no_grad():
            y1 = model(x)
            y2 = loaded(x)
        torch.testing.assert_close(y1, y2)


class TestDataExtraction:
    def test_extract_pairs_shape(self):
        states = np.random.randn(10, 51, 4)
        actions = np.random.randn(10, 50, 2)
        inputs, targets = extract_pairs(states, actions)
        assert inputs.shape == (500, 8)
        assert targets.shape == (500, 2)

    def test_extract_pairs_content(self):
        """First pair should be (state[0,0,:], state[0,1,:]) -> action[0,0,:]."""
        states = np.random.randn(2, 5, 4)
        actions = np.random.randn(2, 4, 2)
        inputs, targets = extract_pairs(states, actions)
        # First pair
        np.testing.assert_array_equal(inputs[0, :4], states[0, 0, :])
        np.testing.assert_array_equal(inputs[0, 4:], states[0, 1, :])
        np.testing.assert_array_equal(targets[0], actions[0, 0, :])

    def test_normalization(self):
        inputs = np.random.randn(1000, 8) * 5 + 3
        mean, std = compute_normalization(inputs)
        assert mean.shape == (8,)
        assert std.shape == (8,)
        normalized = (inputs - mean) / std
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-2)


class TestTraining:
    def test_tiny_training_reduces_loss(self, cfg):
        """Training on a tiny dataset should reduce loss."""
        sim = Simulator(cfg)
        rng = np.random.default_rng(42)

        # Generate tiny dataset
        N, T = 20, 50
        states = np.zeros((N, T + 1, 4))
        actions = np.zeros((N, T, 2))
        for i in range(N):
            acts = rng.uniform(-3, 3, size=(T, 2))
            q0 = rng.uniform(-np.pi, np.pi, size=2)
            v0 = rng.uniform(-1, 1, size=2)
            states[i] = sim.rollout(acts, q0=q0, v0=v0)
            actions[i] = acts

        train_cfg = TrainingConfig(
            hidden_dim=32, n_hidden=2, batch_size=128,
            lr=1e-3, epochs=5, output_dir="/tmp/test_train",
        )

        model, logs = train(
            states[:15], actions[:15],
            states[15:], actions[15:],
            cfg=train_cfg, tau_max=cfg.pendulum.tau_max,
            device="cpu",
        )

        # Loss should decrease
        assert logs[-1].train_loss < logs[0].train_loss
        # Model should produce reasonable outputs
        policy = MLPPolicy(model, tau_max=cfg.pendulum.tau_max, device="cpu")
        action = policy(states[0, 0], states[0, 1])
        assert action.shape == (2,)
