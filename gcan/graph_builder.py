import torch
import torch.nn as nn
from typing import Optional, Union
from torch_geometric.data import Data, Batch


class GraphBuilder:
    """Converts multi-agent observations into PyG graphs for GAT-based credit assignment.

    Zero EPyMARL dependency — takes plain tensors, returns PyG Data/Batch.

    Args:
        n_agents: Number of agents.
        obs_dim: Dimension of per-agent observation vector.
        structured_features: If True, parse MPE-format obs for position/velocity features.
        graph_type: "full" for fully connected, "distance" for threshold-based.
        distance_threshold: Max distance for edge (only when graph_type="distance").
        device: Tensor device.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        structured_features: bool = True,
        graph_type: str = "full",
        distance_threshold: float = 1.0,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.structured_features = structured_features
        self.graph_type = graph_type
        self.distance_threshold = distance_threshold
        self.device = device

        # Pre-compute fully connected edge_index (no self-loops)
        self._cached_edge_index: Optional[torch.Tensor] = None

    @property
    def node_feat_dim(self) -> int:
        if self.structured_features:
            return 8 + self.obs_dim  # vel(2) + pos(2) + 4 distance stats + raw obs
        else:
            return 1 + self.obs_dim  # q-value + raw obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        obs: torch.Tensor,
        agent_qs: Optional[torch.Tensor] = None,
    ) -> Union[Data, Batch]:
        """Build PyG graph(s) from observations.

        Args:
            obs: [B, N, obs_dim] per-agent observations.
            agent_qs: [B, N] per-agent Q-values (optional).

        Returns:
            Data (B=1) or Batch (B>1) with attributes .x, .edge_index.
        """
        B, N, _ = obs.shape
        assert N == self.n_agents, f"Expected {self.n_agents} agents, got {N}"

        if agent_qs is None:
            agent_qs = torch.zeros(B, N, device=self.device)

        graphs = []
        for b in range(B):
            g = self._build_single(obs[b], agent_qs[b])
            graphs.append(g)

        if B == 1:
            return graphs[0]
        return Batch.from_data_list(graphs)

    # ------------------------------------------------------------------
    # Internal: single graph
    # ------------------------------------------------------------------

    def _build_single(self, obs: torch.Tensor, agent_qs: torch.Tensor) -> Data:
        """Build one graph from [N, obs_dim] and [N] q-values."""
        x = self._build_node_features(obs, agent_qs)
        edge_index = self._build_edges(obs)
        return Data(x=x, edge_index=edge_index)

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _build_node_features(
        self, obs: torch.Tensor, agent_qs: torch.Tensor
    ) -> torch.Tensor:
        """Construct node features [N, node_feat_dim]."""
        N = obs.shape[0]

        if not self.structured_features:
            qs = agent_qs.view(N, 1)
            return torch.cat([qs, obs], dim=-1)

        # Structured: parse geometric features from MPE-format observations
        velocity = self._parse_agent_velocities(obs)          # [N, 2]
        position = self._parse_agent_positions(obs)            # [N, 2]

        n_landmarks = N  # SimpleSpread has N landmarks
        lm_rel_start = 4
        lm_rel_end = 4 + n_landmarks * 2
        landmark_rels = obs[:, lm_rel_start:lm_rel_end].view(N, n_landmarks, 2)

        other_rel_start = lm_rel_end
        other_rel_end = other_rel_start + (N - 1) * 2
        other_rels = obs[:, other_rel_start:other_rel_end].view(N, N - 1, 2)

        # Distance features
        landmark_dists = torch.norm(landmark_rels, dim=-1)        # [N, N_lm]
        other_dists = torch.norm(other_rels, dim=-1)              # [N, N-1]

        nearest_landmark = landmark_dists.min(dim=-1).values.unsqueeze(-1)   # [N, 1]
        mean_landmark = landmark_dists.mean(dim=-1).unsqueeze(-1)            # [N, 1]
        nearest_agent = other_dists.min(dim=-1).values.unsqueeze(-1)         # [N, 1]
        mean_agent = other_dists.mean(dim=-1).unsqueeze(-1)                  # [N, 1]

        geometric = torch.cat([
            velocity, position, nearest_landmark, mean_landmark,
            nearest_agent, mean_agent,
        ], dim=-1)  # [N, 8]

        return torch.cat([geometric, obs], dim=-1)  # [N, 8 + obs_dim]

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _build_edges(self, obs: torch.Tensor) -> torch.Tensor:
        """Build edge_index [2, E] for N agents."""
        N = obs.shape[0]

        if self.graph_type == "full":
            return self._fully_connected_edges(N)
        elif self.graph_type == "distance":
            positions = self._parse_agent_positions(obs)
            return self._distance_edges(positions)
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

    def _fully_connected_edges(self, N: int) -> torch.Tensor:
        """Edge_index for fully connected graph on N nodes, no self-loops."""
        if self._cached_edge_index is not None and self._cached_edge_index.shape[1] == N * (N - 1):
            return self._cached_edge_index.to(self.device)

        src = torch.arange(N, device=self.device).repeat_interleave(N - 1)
        dst_list = []
        for i in range(N):
            others = torch.cat([
                torch.arange(i, device=self.device),
                torch.arange(i + 1, N, device=self.device),
            ])
            dst_list.append(others)
        dst = torch.cat(dst_list)
        self._cached_edge_index = torch.stack([src, dst], dim=0)
        return self._cached_edge_index

    def _distance_edges(self, positions: torch.Tensor) -> torch.Tensor:
        """Edge_index for edges where |p_i - p_j| < distance_threshold."""
        dist = torch.cdist(positions, positions)  # [N, N]
        mask = (dist < self.distance_threshold) & ~torch.eye(
            positions.shape[0], dtype=torch.bool, device=positions.device
        )
        src, dst = mask.nonzero(as_tuple=True)
        if src.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=positions.device)
        return torch.stack([src, dst], dim=0)

    # ------------------------------------------------------------------
    # Position / velocity parsers (MPE-specific)
    # ------------------------------------------------------------------

    def _parse_agent_positions(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract absolute agent positions from MPE observations.

        MPE SimpleSpread: obs[2:4] = self absolute position.
        Returns [N, 2].
        """
        return obs[:, 2:4]

    def _parse_agent_velocities(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract absolute agent velocities from MPE observations.

        MPE SimpleSpread: obs[0:2] = self velocity.
        Returns [N, 2].
        """
        return obs[:, 0:2]
