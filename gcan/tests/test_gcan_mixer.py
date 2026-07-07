"""Unit tests for GCANMixer."""

import copy
import torch
import pytest

import sys
sys.path.insert(0, "D:/科研/GNN-LLM-MARL协同优化框架")
from gcan.gcan_mixer import GCANMixer


class FakeArgs:
    n_agents = 5
    obs_shape = 30
    state_shape = (150,)
    mixing_embed_dim = 32
    gat_n_heads = 4
    use_cuda = False


@pytest.fixture
def args():
    return FakeArgs()


@pytest.fixture
def mixer(args):
    return GCANMixer(args)


@pytest.fixture
def batch_data():
    """Return (agent_qs, states, obs) with bs*T=8, bs=4, T=2."""
    agent_qs = torch.randn(8, 1, 5)
    states = torch.randn(8, 150)
    obs = torch.randn(4, 2, 5, 30)
    return agent_qs, states, obs


class TestForwardShapes:
    def test_basic_forward(self, mixer, batch_data):
        agent_qs, states, obs = batch_data
        out = mixer(agent_qs, states, obs=obs)
        assert out.shape == (8, 1, 1)

    def test_fallback_no_obs(self, mixer, batch_data):
        agent_qs, states, _ = batch_data
        out = mixer(agent_qs, states, obs=None)
        assert out.shape == (8, 1, 1)

    def test_different_bs_T(self, mixer):
        for bs_T in [1, 4, 16, 32]:
            agent_qs = torch.randn(bs_T, 1, 5)
            states = torch.randn(bs_T, 150)
            obs = torch.randn(bs_T, 1, 5, 30)
            out = mixer(agent_qs, states, obs=obs)
            assert out.shape == (bs_T, 1, 1)

    def test_flattened_agent_qs(self, mixer, batch_data):
        """agent_qs may arrive as (bs, T, n_agents) without the extra dim."""
        _, states, obs = batch_data
        agent_qs = torch.randn(4, 2, 5)
        out = mixer(agent_qs, states, obs=obs)
        assert out.shape == (4, 2, 1)  # (bs, T, 1)


class TestDeepCopy:
    def test_target_network(self, mixer, batch_data):
        agent_qs, states, obs = batch_data
        target = copy.deepcopy(mixer)
        with torch.no_grad():
            out = mixer(agent_qs, states, obs=obs)
            target_out = target(agent_qs, states, obs=obs)
        assert torch.allclose(out, target_out)

    def test_gradient_flows(self, mixer, batch_data):
        """Backward pass should compute gradients for all parameters."""
        agent_qs, states, obs = batch_data
        out = mixer(agent_qs, states, obs=obs)
        loss = out.sum()
        loss.backward()
        for name, p in mixer.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


class TestAttentionWeights:
    def test_attention_extractable(self, mixer, batch_data):
        agent_qs, states, obs = batch_data
        mixer(agent_qs, states, obs=obs)
        edge_idx, att = mixer.get_attention_weights()
        assert edge_idx is not None
        assert att is not None
        assert att.ndim == 2  # (N_edges, 1) for single head or (E, H)

    def test_attention_edge_count(self, mixer, batch_data):
        agent_qs, states, obs = batch_data
        mixer(agent_qs, states, obs=obs)
        edge_idx, att = mixer.get_attention_weights()
        # total edges = total_ts * N * (N-1), with N=5: 8*20=160
        assert edge_idx.shape[1] > 0
        assert att.shape[0] > 0


class TestParameterCount:
    def test_parameter_count(self, mixer):
        n_params = sum(p.numel() for p in mixer.parameters())
        # Should be modest (< 100K)
        assert n_params < 100000
        assert n_params > 1000
