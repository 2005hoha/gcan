"""Unit tests for GraphBuilder."""

import torch
import pytest
import time
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

import sys
sys.path.insert(0, "D:/科研/GNN-LLM-MARL协同优化框架")
from gcan.graph_builder import GraphBuilder


# ── helpers ──────────────────────────────────────────────────────────

def make_obs(B, N, obs_dim=30):
    """Create synthetic obs with recognizable position/velocity patterns."""
    obs = torch.randn(B, N, obs_dim)
    # Set positions [2:4] to distinguishable values
    for i in range(N):
        obs[:, i, 2] = float(i)
        obs[:, i, 3] = float(i * 0.5)
    # Set velocities [0:2] to small values
    obs[:, :, 0:2] = torch.randn(B, N, 2) * 0.1
    return obs


# ── Test 1: passthrough mode shapes ──────────────────────────────────

class TestPassthroughMode:
    def test_single_graph(self):
        builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=False)
        obs = make_obs(1, 5)
        qs = torch.randn(1, 5)
        g = builder.build(obs, qs)

        assert isinstance(g, Data)
        assert g.x.shape == (5, 31)  # 1 (q) + 30 (obs)
        assert g.edge_index.shape == (2, 20)  # 5*4 = 20
        # No self-loops
        for i in range(g.edge_index.shape[1]):
            assert g.edge_index[0, i] != g.edge_index[1, i]

    def test_batch(self):
        builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=False)
        obs = make_obs(4, 5)
        qs = torch.randn(4, 5)
        g = builder.build(obs, qs)

        assert isinstance(g, Batch)
        assert g.x.shape == (20, 31)   # 4*5 nodes
        assert g.batch.shape == (20,)  # batch assignment
        assert (g.batch[:5] == 0).all()
        assert (g.batch[5:10] == 1).all()


# ── Test 2: structured features ──────────────────────────────────────

class TestStructuredMode:
    def test_node_feature_dim(self):
        builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=True)
        assert builder.node_feat_dim == 38  # 8 + 30

    def test_geometric_features_valid(self):
        builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=True)
        obs = make_obs(1, 5)
        g = builder.build(obs)
        assert g.x.shape == (5, 38)
        # Indices: vel(0:2), pos(2:4), dists(4:8), raw_obs(8:38)
        vel = g.x[:, 0:2]
        pos = g.x[:, 2:4]
        pos_raw = g.x[:, 10:12]  # raw obs positions shifted by 8
        assert torch.allclose(vel, obs[0, :, 0:2])
        assert torch.allclose(pos, obs[0, :, 2:4])
        assert torch.allclose(pos_raw, obs[0, :, 2:4])

    def test_distance_features_non_negative(self):
        builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=True)
        obs = make_obs(1, 5)
        g = builder.build(obs)
        # Distance features at indices 4:8 (after vel(0:2), pos(2:4))
        dist_feats = g.x[:, 4:8]
        assert (dist_feats >= 0).all()


# ── Test 3: edge counts for various N ────────────────────────────────

@pytest.mark.parametrize("N,expected_edges", [
    (3, 6),
    (5, 20),
    (8, 56),
    (12, 132),
])
def test_edge_counts(N, expected_edges):
    builder = GraphBuilder(n_agents=N, obs_dim=2 * N + (N - 1) * 2 + 2 + (N - 1) * 2)
    obs = torch.randn(1, N, 2 * N + (N - 1) * 2 + 2 + (N - 1) * 2)
    g = builder.build(obs)
    assert g.edge_index.shape == (2, expected_edges)


# ── Test 4: distance edges ───────────────────────────────────────────

class TestDistanceEdges:
    def test_threshold(self):
        builder = GraphBuilder(n_agents=3, obs_dim=30, graph_type="distance",
                               distance_threshold=2.0)
        obs = torch.zeros(1, 3, 30)
        obs[0, 0, 2:4] = torch.tensor([0.0, 0.0])
        obs[0, 1, 2:4] = torch.tensor([0.5, 0.5])
        obs[0, 2, 2:4] = torch.tensor([10.0, 10.0])

        g = builder.build(obs)
        # Agent 0 and 1 within threshold (dist ~0.71 < 2.0) → connected
        # Agent 2 too far from both → only agents 0,1 should connect
        edges = g.edge_index.t().tolist()
        assert [0, 1] in edges or [1, 0] in edges
        assert [0, 2] not in edges
        assert [2, 0] not in edges
        assert [1, 2] not in edges
        assert [2, 1] not in edges

    def test_no_edges_below_threshold(self):
        builder = GraphBuilder(n_agents=3, obs_dim=30, graph_type="distance",
                               distance_threshold=0.001)
        obs = torch.zeros(1, 3, 30)
        obs[0, 0, 2:4] = torch.tensor([0.0, 0.0])
        obs[0, 1, 2:4] = torch.tensor([5.0, 5.0])
        obs[0, 2, 2:4] = torch.tensor([10.0, 10.0])
        g = builder.build(obs)
        assert g.edge_index.shape[1] == 0  # no edges


# ── Test 5: no agent_qs ──────────────────────────────────────────────

def test_no_agent_qs():
    builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=False)
    obs = make_obs(1, 5)
    g = builder.build(obs)  # no qs
    assert g.x.shape == (5, 31)
    # Q-value slot should be zeros
    assert (g.x[:, 0] == 0).all()


# ── Test 6: performance ──────────────────────────────────────────────

def test_performance():
    builder = GraphBuilder(n_agents=5, obs_dim=30)
    obs = make_obs(32, 5)  # batch_size=32
    qs = torch.randn(32, 5)

    # Warmup
    for _ in range(10):
        builder.build(obs, qs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(100):
        builder.build(obs, qs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - t0) / 100 * 1000  # ms

    assert elapsed < 50.0, f"Build took {elapsed:.2f}ms, expected < 50ms"


# ── Test 7: device consistency ───────────────────────────────────────

def test_device_cpu():
    builder = GraphBuilder(n_agents=5, obs_dim=30, device="cpu")
    obs = make_obs(1, 5)
    g = builder.build(obs)
    assert g.x.device.type == "cpu"
    assert g.edge_index.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    builder = GraphBuilder(n_agents=5, obs_dim=30, device="cuda")
    obs = make_obs(1, 5).cuda()
    qs = torch.randn(1, 5).cuda()
    g = builder.build(obs, qs)
    assert g.x.device.type == "cuda"
    assert g.edge_index.device.type == "cuda"


# ── Test 8: PyG GATConv integration ──────────────────────────────────

def test_gat_integration():
    builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=False)
    obs = make_obs(2, 5)
    qs = torch.randn(2, 5)
    g = builder.build(obs, qs)

    gat = GATConv(31, 64, heads=4, concat=True)
    out = gat(g.x, g.edge_index)
    assert out.shape == (10, 256)  # 2*5 nodes, 64*4 heads


# ── Test 9: structured GAT integration ───────────────────────────────

def test_structured_gat_integration():
    builder = GraphBuilder(n_agents=5, obs_dim=30, structured_features=True)
    obs = make_obs(1, 5)
    g = builder.build(obs)

    gat = GATConv(38, 64, heads=4, concat=True)
    out = gat(g.x, g.edge_index)
    assert out.shape == (5, 256)
    # GATConv adds self-loops by default → edge_index grows
    _, (edge_index_out, att_weights) = gat(g.x, g.edge_index, return_attention_weights=True)
    assert edge_index_out.shape[1] == att_weights.shape[0]  # edges match weights


# ── Test 10: edge_index cache ────────────────────────────────────────

def test_edge_index_cache():
    builder = GraphBuilder(n_agents=5, obs_dim=30)
    assert builder._cached_edge_index is None
    g1 = builder.build(make_obs(1, 5))
    assert builder._cached_edge_index is not None
    g2 = builder.build(make_obs(1, 5))
    assert g1.edge_index is g2.edge_index  # same object (cached)
