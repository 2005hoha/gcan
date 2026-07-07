import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .graph_builder import GraphBuilder


class GCANMixer(nn.Module):
    """GAT-driven credit assignment mixer.

    Replaces QMIX's monotonic hypernetwork with a 2-layer GAT that produces
    per-agent credit scores via attention over a fully connected agent graph.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        self.hidden_dim = args.mixing_embed_dim
        self.n_heads = getattr(args, "gat_n_heads", 4)

        device = "cuda" if getattr(args, "use_cuda", False) else "cpu"

        # Graph builder
        self.graph_builder = GraphBuilder(
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            structured_features=True,
            graph_type="full",
            device=device,
        )

        node_feat_dim = self.graph_builder.node_feat_dim

        # 2-layer GAT
        self.gat1 = GATConv(node_feat_dim, self.hidden_dim, heads=self.n_heads,
                            concat=True, add_self_loops=False)
        self.gat2 = GATConv(self.hidden_dim * self.n_heads, self.hidden_dim,
                            heads=1, concat=False, add_self_loops=False)

        # Per-agent credit head (Softplus → positive, then Softmax across agents)
        self.credit_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus(),
        )

        # State-value baseline V(s)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Attention storage for visualization
        self._last_edge_index = None
        self._last_attention = None

    def forward(self, agent_qs, states, obs=None):
        """
        Args:
            agent_qs: (bs, T, n_agents) or (bs*T, 1, n_agents).
            states:   (bs, T, state_dim) or (bs*T, state_dim)
            obs:      (bs, T, n_agents, obs_dim) or None — falls back to state.

        Returns:
            q_tot: (bs, T, 1) matching QMixer output shape convention.
        """
        n_agents = self.n_agents
        orig_bs = agent_qs.shape[0]  # preserve for output reshape

        # Normalize agent_qs shape to (total_timesteps, n_agents)
        if agent_qs.dim() == 3 and agent_qs.shape[1] == 1:
            agent_qs_flat = agent_qs.squeeze(1)        # (bs*T, n_agents)
        elif agent_qs.dim() == 3:
            agent_qs_flat = agent_qs.reshape(-1, n_agents)  # (bs, T, n) -> (bs*T, n)
        else:
            agent_qs_flat = agent_qs                    # already 2D

        total_ts = agent_qs_flat.shape[0]  # bs*T

        # Reshape obs for GraphBuilder
        if obs is None:
            # Fallback: state = concat(all obs) in MPE
            obs_flat = states.view(total_ts, n_agents, -1)
        else:
            # obs from batch: (bs, T, n_agents, obs_dim)
            obs_flat = obs.reshape(total_ts, n_agents, -1)

        # Build graph(s)
        graph = self.graph_builder.build(obs_flat, agent_qs_flat)

        # GAT forward
        x = F.elu(self.gat1(graph.x, graph.edge_index))
        # Store attention from layer 1 for visualization
        if hasattr(self.gat1, 'att_src') and self.gat1.att_src is not None:
            self._last_attention = self.gat1.att_src.detach().mean(dim=1)
        self._last_edge_index = graph.edge_index.clone()

        x = F.elu(self.gat2(x, graph.edge_index))

        # Credit per node
        credits = self.credit_head(x)                  # (total_ts*N, 1)
        credits = credits.view(total_ts, n_agents)     # (total_ts, n_agents)

        # Softmax across agents
        credits = F.softmax(credits, dim=1)            # sum=1 per graph

        # Weighted sum of Q-values
        q_tot = (credits * agent_qs_flat).sum(dim=1)   # (total_ts,)

        # Add state-value baseline
        states_2d = states.reshape(-1, self.state_dim)  # (total_ts, state_dim)
        v = self.V(states_2d).squeeze(-1)               # (total_ts,)
        q_tot = q_tot + v

        return q_tot.view(orig_bs, -1, 1)  # (bs, T, 1)

    def get_attention_weights(self):
        """Return last forward-pass attention for visualization."""
        return self._last_edge_index, self._last_attention
