"""
NEURL Engine — Cell 7: GNN Architecture Suite

Implements five complete GNN architectures as nn.Module subclasses, a
multi-task prediction head, and a factory for instantiation. All models
share the NEURLBaseGNN interface so they're interchangeable in the training
pipeline.

Design rationale: Supporting five architectures allows empirical comparison
on any given dataset. The factory pattern means Cell 8's trainer never
needs architecture-specific code — just config.gnn.architecture.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Guard: if torch_geometric is not available, provide stub classes
if not TORCH_GEOMETRIC_AVAILABLE:
    raise ImportError(
        "torch_geometric is required for Cell 7. "
        "Install with: pip install torch_geometric"
    )


# =============================================================================
# NEURLBaseGNN — abstract base class for all NEURL GNN architectures
# =============================================================================

class NEURLBaseGNN(nn.Module):
    """Abstract base class defining the NEURL GNN interface.

    All NEURL GNN architectures implement encode() and optionally override
    decode(). The forward() method calls encode then decode, providing a
    complete link-prediction-capable model out of the box.

    The inner-product decoder is a well-established choice for knowledge
    graph link prediction (see: Kipf & Welling, 2016 VGAE paper).

    Args:
        in_channels: Input node feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Output embedding dimension.
        num_layers: Number of message-passing layers.
        dropout: Dropout probability.
        config: NEURLConfig for architecture-specific params.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        config: "NEURLConfig",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.config = config
        self._logger = NEURLLogger.get_logger(self.__class__.__name__)

    @abstractmethod
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features into embeddings via message passing.

        Args:
            x: Node feature matrix, shape [N, in_channels].
            edge_index: Edge connectivity, shape [2, E].

        Returns:
            Node embedding matrix, shape [N, out_channels].
        """
        ...

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode node embeddings to edge scores via inner product.

        The inner-product decoder: score(i, j) = sigmoid(z_i · z_j)
        is parameter-free and scales well to large edge sets.

        Args:
            z: Node embeddings, shape [N, out_channels].
            edge_index: Edge index for which to compute scores, shape [2, E].

        Returns:
            Edge prediction logits, shape [E].
        """
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Args:
            x: Node feature matrix.
            edge_index: Edge index.

        Returns:
            Tuple of (node_embeddings [N, out_channels], edge_logits [E]).
        """
        z = self.encode(x, edge_index)
        edge_logits = self.decode(z, edge_index)
        return z, edge_logits

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """Get node embeddings as a detached numpy array.

        Args:
            x: Node feature matrix.
            edge_index: Edge index.

        Returns:
            Numpy array of shape [N, out_channels].
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x, edge_index)
        return z.cpu().numpy()

    def count_parameters(self) -> int:
        """Count the total trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# NEURLGraphConvNet — GCN implementation
# =============================================================================

class NEURLGraphConvNet(NEURLBaseGNN):
    """Graph Convolutional Network (Kipf & Welling, 2017).

    Stack of GCNConv layers with BatchNorm, ReLU activation, and Dropout.
    GCN averages neighbor features with normalized adjacency — fast and
    effective for homophilic graphs where connected nodes share features.

    Architecture:
        [in_channels] → GCNConv → BN → ReLU → Dropout → ... → [out_channels]
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float,
                 config: "NEURLConfig") -> None:
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_layers, dropout, config)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_dim, out_dim, normalize=True, add_self_loops=True))
            if i < num_layers - 1:
                self.bns.append(BatchNorm(out_dim))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GCN message passing with residual-style progressive refinement.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge index [2, E].

        Returns:
            Node embeddings [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# =============================================================================
# NEURLGraphSAGE — GraphSAGE implementation
# =============================================================================

class NEURLGraphSAGE(NEURLBaseGNN):
    """Graph SAGE — Inductive Representation Learning (Hamilton et al., 2017).

    SAGEConv learns a separate aggregation function (mean/max/lstm) for
    neighbor features and concatenates with self-features. Inductive:
    generalizes to unseen nodes at inference time.

    Architecture:
        [in_channels] → SAGEConv(aggr) → BN → ReLU → Dropout → ... → [out_channels]
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float,
                 config: "NEURLConfig") -> None:
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_layers, dropout, config)
        aggr = config.gnn.sage_aggregation

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_dim, out_dim, aggr=aggr))
            if i < num_layers - 1:
                self.bns.append(BatchNorm(out_dim))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GraphSAGE message passing.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge index [2, E].

        Returns:
            Node embeddings [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# =============================================================================
# NEURLGraphAttention — GAT implementation
# =============================================================================

class NEURLGraphAttention(NEURLBaseGNN):
    """Graph Attention Network v2 (Brody et al., 2022).

    GATv2Conv fixes the static attention problem of GATv1 by computing
    dynamic attention that depends on both source and target features.
    Multi-head attention with concatenation in hidden layers, mean-pooling
    in the final layer.

    Architecture:
        [in_channels] → GATv2Conv(heads=H) → BN → ELU → Dropout → ... → [out_channels]
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float,
                 config: "NEURLConfig") -> None:
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_layers, dropout, config)
        heads = config.gnn.gat_heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            is_last = (i == num_layers - 1)
            # Last layer: single head, mean aggregation → out_channels
            # Hidden layers: multi-head concat → hidden_channels * heads
            out_dim = out_channels if is_last else hidden_channels
            n_heads = 1 if is_last else heads
            concat = not is_last
            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=n_heads,
                          concat=concat, dropout=dropout)
            )
            if not is_last:
                self.bns.append(BatchNorm(out_dim * n_heads if concat else out_dim))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GAT message passing with attention-weighted aggregation.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge index [2, E].

        Returns:
            Node embeddings [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# =============================================================================
# NEURLGraphIsomorphism — GIN implementation
# =============================================================================

class NEURLGraphIsomorphism(NEURLBaseGNN):
    """Graph Isomorphism Network (Xu et al., 2019).

    GIN is maximally expressive among message-passing GNNs (equivalent to
    the Weisfeiler-Lehman graph isomorphism test). Uses a learnable MLP
    as the aggregation function and a learnable epsilon parameter.

    Architecture:
        GINConv(MLP([in → hidden → hidden])) × (num_layers-1) + final layer
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float,
                 config: "NEURLConfig") -> None:
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_layers, dropout, config)
        eps_init = config.gnn.gin_eps

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            # MLP inside GIN: two-layer with ReLU
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_dim),
            )
            self.convs.append(GINConv(mlp, eps=eps_init, train_eps=True))
            if i < num_layers - 1:
                self.bns.append(BatchNorm(out_dim))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GIN message passing with MLP aggregation.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge index [2, E].

        Returns:
            Node embeddings [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# =============================================================================
# NEURLVariationalGAE — VGAE implementation
# =============================================================================

class NEURLVariationalGAE(NEURLBaseGNN):
    """Variational Graph Autoencoder (Kipf & Welling, 2016).

    Encoder produces mean (mu) and log-standard-deviation (logstd) of a
    Gaussian latent distribution per node. The reparameterization trick
    samples z = mu + eps * exp(logstd) during training. The KL divergence
    term regularizes the latent space.

    This architecture is best suited for link prediction tasks, as the
    probabilistic latent space naturally captures uncertainty.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float,
                 config: "NEURLConfig") -> None:
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_layers, dropout, config)
        latent_dim = config.gnn.vgae_latent_dim

        # Shared encoder for feature extraction
        self.shared_convs = nn.ModuleList()
        self.shared_bns = nn.ModuleList()
        for i in range(num_layers - 1):
            in_dim = in_channels if i == 0 else hidden_channels
            self.shared_convs.append(GCNConv(in_dim, hidden_channels))
            self.shared_bns.append(BatchNorm(hidden_channels))

        # Separate heads for mu and logstd
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logstd = GCNConv(hidden_channels, latent_dim)

        # Project latent to out_channels for uniform interface
        self.project = nn.Linear(latent_dim, out_channels)

        self._mu: Optional[torch.Tensor] = None
        self._logstd: Optional[torch.Tensor] = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """VGAE encoding with reparameterization trick.

        Stores mu and logstd as instance variables for KL loss computation.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge index [2, E].

        Returns:
            Sampled latent embeddings projected to [N, out_channels].
        """
        # Shared feature extraction
        h = x
        for conv, bn in zip(self.shared_convs, self.shared_bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Compute mu and logstd
        self._mu = self.conv_mu(h, edge_index)
        self._logstd = self.conv_logstd(h, edge_index).clamp(max=10)

        # Reparameterization: z = mu + eps * exp(logstd) (only during training)
        if self.training:
            eps = torch.randn_like(self._mu)
            z = self._mu + eps * torch.exp(self._logstd)
        else:
            z = self._mu  # Use mean at inference for deterministic output

        return self.project(z)

    def kl_loss(self) -> torch.Tensor:
        """Compute KL divergence: KL[q(z|x) || p(z)] where p(z)=N(0,I).

        This regularization term encourages the learned latent space to
        resemble a standard Gaussian, preventing posterior collapse.

        Returns:
            Scalar KL divergence loss.

        Raises:
            RuntimeError: If encode() has not been called first.
        """
        if self._mu is None or self._logstd is None:
            raise RuntimeError("encode() must be called before kl_loss()")
        # KL divergence: -0.5 * sum(1 + 2*logstd - mu^2 - exp(2*logstd))
        kl = -0.5 * (1 + 2 * self._logstd - self._mu.pow(2) - (2 * self._logstd).exp())
        return kl.mean()


# =============================================================================
# NEURLMultiTaskHead — multi-task prediction head
# =============================================================================

class NEURLMultiTaskHead(nn.Module):
    """Multi-task prediction head operating on node embeddings.

    Routes embeddings to task-specific MLP heads:
    - node_classification: softmax over node type classes
    - link_prediction: sigmoid score from pair embeddings (handled in trainer)
    - anomaly_score: reconstruction distance (scalar per node)
    - community_assignment: softmax over community classes

    Design rationale: Sharing a single GNN encoder across multiple objectives
    produces more generalizable embeddings than single-task training —
    the encoder learns features useful for multiple downstream tasks.

    Args:
        in_channels: Embedding dimension from the GNN encoder.
        num_node_classes: Number of node classification classes.
        num_communities: Number of community classes for community head.
        hidden_dim: Hidden dimension for MLP heads.
    """

    def __init__(
        self,
        in_channels: int,
        num_node_classes: int = 2,
        num_communities: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Node classification head: MLP → softmax
        self.node_clf_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_node_classes),
        )

        # Anomaly score head: MLP → scalar (reconstruction proxy)
        self.anomaly_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Score in [0, 1]
        )

        # Community assignment head: MLP → softmax
        self.community_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_communities),
        )

        # Node feature reconstruction head (for self-supervised reconstruction loss)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
        )

    def forward(self, z: torch.Tensor, task: str) -> torch.Tensor:
        """Route embeddings to the appropriate task head.

        Args:
            z: Node embeddings, shape [N, in_channels].
            task: One of 'node_classification', 'anomaly_score',
                  'community_assignment', 'reconstruction'.

        Returns:
            Task-specific output tensor.

        Raises:
            ValueError: If task name is unrecognized.
        """
        if task == "node_classification":
            return self.node_clf_head(z)
        elif task == "anomaly_score":
            return self.anomaly_head(z)
        elif task == "community_assignment":
            return self.community_head(z)
        elif task == "reconstruction":
            return self.reconstruction_head(z)
        else:
            raise ValueError(f"Unknown task: '{task}'. Valid: node_classification, "
                             "anomaly_score, community_assignment, reconstruction")

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_weights: "LossWeightsConfig",
    ) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            predictions: Dict of task_name → prediction tensor.
            targets: Dict of task_name → target tensor.
            task_weights: LossWeightsConfig with per-task weights.

        Returns:
            Scalar total weighted loss.
        """
        total_loss = torch.tensor(0.0, requires_grad=True)

        if "node_classification" in predictions and "node_classification" in targets:
            clf_loss = F.cross_entropy(
                predictions["node_classification"],
                targets["node_classification"]
            )
            total_loss = total_loss + clf_loss

        if "reconstruction" in predictions and "reconstruction" in targets:
            recon_loss = F.mse_loss(
                predictions["reconstruction"],
                targets["reconstruction"]
            )
            total_loss = total_loss + task_weights.node_reconstruction * recon_loss

        return total_loss


# =============================================================================
# GNNFactory — factory for architecture instantiation
# =============================================================================

class GNNFactory:
    """Factory class for creating NEURL GNN models by architecture type.

    Centralizes architecture selection so the training pipeline never
    contains architecture-specific conditional logic.

    Example:
        model = GNNFactory.create(
            ArchitectureType.GCN,
            in_channels=13, hidden_channels=64, out_channels=32, config=cfg
        )
    """

    _REGISTRY: Dict[ArchitectureType, type] = {
        ArchitectureType.GCN: NEURLGraphConvNet,
        ArchitectureType.GRAPHSAGE: NEURLGraphSAGE,
        ArchitectureType.GAT: NEURLGraphAttention,
        ArchitectureType.GIN: NEURLGraphIsomorphism,
        ArchitectureType.VGAE: NEURLVariationalGAE,
    }

    @classmethod
    def create(
        cls,
        architecture_type: ArchitectureType,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        config: "NEURLConfig",
    ) -> NEURLBaseGNN:
        """Create a GNN model of the specified architecture type.

        Args:
            architecture_type: ArchitectureType enum value.
            in_channels: Input feature dimension.
            hidden_channels: Hidden layer width.
            out_channels: Output embedding dimension.
            config: NEURLConfig for architecture-specific hyperparameters.

        Returns:
            Initialized NEURLBaseGNN subclass instance.

        Raises:
            NEURLConfigError: If architecture type is unknown.
        """
        if architecture_type not in cls._REGISTRY:
            raise NEURLConfigError(
                f"Unknown architecture: '{architecture_type}'. "
                f"Valid: {list(cls._REGISTRY.keys())}"
            )
        model_class = cls._REGISTRY[architecture_type]
        return model_class(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=config.gnn.num_layers,
            dropout=config.gnn.dropout,
            config=config,
        )

    @classmethod
    def list_architectures(cls) -> List[str]:
        """List all supported architecture type names.

        Returns:
            Sorted list of architecture names.
        """
        return sorted(a.value for a in cls._REGISTRY.keys())


# =============================================================================
# DEMO: Instantiate all architectures and run forward passes
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 7: GNN Architecture Suite")
print("=" * 70)

# Determine input dimension from the PyG data object built in Cell 6
if neurl_graph.pyg_data is not None:
    IN_CHANNELS = neurl_graph.pyg_data.x.shape[1]
    HIDDEN_CHANNELS = neurl_config.gnn.hidden_dim
    OUT_CHANNELS = neurl_config.gnn.out_dim
    N_NODES = neurl_graph.pyg_data.num_nodes

    # Use PyG data for realistic forward pass
    x_demo = neurl_graph.pyg_data.x.to(device_manager.device)
    edge_index_demo = neurl_graph.pyg_data.edge_index.to(device_manager.device)
else:
    # Fallback: synthetic demo data
    IN_CHANNELS = 13
    N_NODES = 50
    HIDDEN_CHANNELS = neurl_config.gnn.hidden_dim
    OUT_CHANNELS = neurl_config.gnn.out_dim
    x_demo = torch.randn(N_NODES, IN_CHANNELS).to(device_manager.device)
    # Random sparse graph
    n_edges = N_NODES * 3
    edge_index_demo = torch.randint(0, N_NODES, (2, n_edges)).to(device_manager.device)

print(f"\n  Input: N={N_NODES} nodes, D={IN_CHANNELS} features")
print(f"  Hidden: {HIDDEN_CHANNELS}   Out: {OUT_CHANNELS}   Layers: {neurl_config.gnn.num_layers}")
print(f"  Device: {device_manager.device}\n")

# Instantiate all 5 architectures
architectures_to_test = [
    ArchitectureType.GCN,
    ArchitectureType.GRAPHSAGE,
    ArchitectureType.GAT,
    ArchitectureType.GIN,
    ArchitectureType.VGAE,
]

print(f"  {'Architecture':<16} {'Params':>10} {'z.shape':<20} {'logits.shape':<20} {'Status'}")
print("  " + "─" * 78)

gnn_models: Dict[str, NEURLBaseGNN] = {}

for arch_type in architectures_to_test:
    try:
        model = GNNFactory.create(
            architecture_type=arch_type,
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            config=neurl_config,
        )
        model = model.to(device_manager.device)
        model.eval()

        with torch.no_grad():
            z, logits = model(x_demo, edge_index_demo)

        params = model.count_parameters()
        status = "✅"
        z_shape = str(tuple(z.shape))
        logits_shape = str(tuple(logits.shape))

        gnn_models[arch_type.value] = model
        model_registry.register(
            f"gnn_{arch_type.value.lower()}",
            model,
            {"architecture": arch_type.value, "params": params}
        )

        print(f"  {arch_type.value:<16} {params:>10,} {z_shape:<20} {logits_shape:<20} {status}")

    except Exception as exc:
        print(f"  {arch_type.value:<16} {'ERROR':>10} {str(exc)[:40]:<40} ❌")

# Demonstrate MultiTaskHead
print("\n  ── MultiTaskHead Demo ─────────────────────────────────────────────")
N_COMMUNITIES = max(graph_stats.get("num_communities", 5), 2)
multi_head = NEURLMultiTaskHead(
    in_channels=OUT_CHANNELS,
    num_node_classes=2,
    num_communities=N_COMMUNITIES,
    hidden_dim=HIDDEN_CHANNELS,
).to(device_manager.device)

# Use embeddings from GCN
if "GCN" in gnn_models:
    gcn_model = gnn_models["GCN"]
    gcn_model.eval()
    with torch.no_grad():
        z_gcn, _ = gcn_model(x_demo, edge_index_demo)
    for task in ["node_classification", "anomaly_score", "community_assignment", "reconstruction"]:
        out = multi_head(z_gcn, task)
        print(f"    Task: {task:<25} → output shape: {tuple(out.shape)}")

# Store primary model for training in Cell 8
primary_model = GNNFactory.create(
    architecture_type=neurl_config.gnn.architecture,
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=OUT_CHANNELS,
    config=neurl_config,
).to(device_manager.device)

print(f"\n  Primary model for training: {neurl_config.gnn.architecture.value}")
print(f"  Parameters: {primary_model.count_parameters():,}")
print(f"  Registered architectures: {model_registry.list_registered()}")
print(f"\n  Available architectures: {GNNFactory.list_architectures()}")

print("\n✅ Cell 7 complete — GNN Architecture Suite ready.")
print(f"   Exports: gnn_models, primary_model, multi_head, GNNFactory")
print(f"   IN_CHANNELS={IN_CHANNELS}, HIDDEN_CHANNELS={HIDDEN_CHANNELS}, OUT_CHANNELS={OUT_CHANNELS}")
