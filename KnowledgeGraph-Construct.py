"""
NEURL Engine — Cell 6: Knowledge Graph Construction

Converts discovered entities and relationships into a NetworkX DiGraph
with two node types (entity-level and instance-level), runs community
detection (Louvain), computes hierarchy levels, and produces a
torch_geometric Data object for GNN training.

Design rationale: The two-level graph (schema + instances) is what makes
NEURL unique — the GNN learns from BOTH the structural schema of the data
and actual instance-level data patterns simultaneously, giving richer
embeddings than schema-only or data-only approaches.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime


# =============================================================================
# DATA CONTAINER — NEURLGraph
# =============================================================================

@dataclass
class NEURLGraph:
    """Wrapper around a NetworkX DiGraph with NEURL metadata.

    Stores entity schemas, relationship schemas, node features, edge features,
    Louvain community assignments, and hierarchy level assignments.

    Attributes:
        graph: The underlying NetworkX DiGraph.
        entity_schemas: Dict mapping entity name → EntitySchema.
        relationship_schemas: List of RelationshipSchema objects.
        node_features: Dict mapping node_id → feature dict.
        edge_features: Dict mapping (u, v) → edge attribute dict.
        communities: Dict mapping node_id → community_id (Louvain).
        hierarchy_levels: Dict mapping node_id → topological level (0=root).
        pyg_data: torch_geometric Data object (set after conversion).
        construction_timestamp: ISO timestamp of graph construction.
    """
    graph: nx.DiGraph
    entity_schemas: Dict[str, "EntitySchema"]
    relationship_schemas: List["RelationshipSchema"]
    node_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_features: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    communities: Dict[str, int] = field(default_factory=dict)
    hierarchy_levels: Dict[str, int] = field(default_factory=dict)
    pyg_data: Optional[Any] = None   # torch_geometric.data.Data
    construction_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# KnowledgeGraphBuilder
# =============================================================================

class KnowledgeGraphBuilder:
    """Builds and enriches the NEURL knowledge graph from entities and relationships.

    Graph structure:
    - Entity nodes (node_type='entity'): one per EntitySchema
    - Instance nodes (node_type='instance'): sampled records from each entity
    - Entity→Entity edges: from RelationshipSchema
    - Instance→Entity edges: membership (instance belongs to entity)
    - Cross-entity Instance edges: FK-linked instance pairs (sampled)

    Example:
        builder = KnowledgeGraphBuilder(config)
        neurl_graph = builder.build(df, entities, relationships, schema)
        stats = builder.graph_statistics(neurl_graph.graph)
    """

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the graph builder.

        Args:
            config: NEURLConfig master configuration object.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("KnowledgeGraphBuilder")

    # ─── MAIN BUILD METHOD ───────────────────────────────────────────────────

    def build(
        self,
        df: pd.DataFrame,
        entities: List["EntitySchema"],
        relationships: List["RelationshipSchema"],
        schema: "SchemaMetadata",
    ) -> NEURLGraph:
        """Build the full NEURL knowledge graph.

        Args:
            df: Cleaned DataFrame.
            entities: List of EntitySchema from Node Intelligence Engine.
            relationships: List of RelationshipSchema from Relationship Engine.
            schema: SchemaMetadata from ingestion engine.

        Returns:
            NEURLGraph wrapping a populated NetworkX DiGraph.
        """
        G = nx.DiGraph()

        entity_schemas_map = {e.entity_name: e for e in entities}
        relationship_schemas_list = list(relationships)

        self._logger.info(
            f"Building graph: {len(entities)} entities, {len(relationships)} relationships"
        )

        # ── Step 1: Add entity-level nodes ───────────────────────────────
        for entity in entities:
            node_id = f"entity:{entity.entity_name}"
            G.add_node(node_id, **{
                "node_type": NodeType.ENTITY.value,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type,
                "column_count": len(entity.columns),
                "row_count": len(df),
                "columns": entity.columns,
                "representative_column": entity.representative_column,
                "confidence": entity.confidence,
                "label": entity.entity_name,
            })

        # ── Step 2: Add instance nodes (sampled records per entity) ──────
        max_instances = self._config.graph.max_instance_nodes_per_entity
        rng = np.random.default_rng(self._config.random_seed)

        # Track instance node IDs for FK edge construction
        instance_registry: Dict[str, Dict[str, str]] = {}  # entity → {key_val → node_id}

        for entity in entities:
            rep_col = entity.representative_column
            if rep_col not in df.columns:
                continue

            # Sample unique key values to cap instance nodes
            unique_keys = df[rep_col].dropna().unique()
            if len(unique_keys) > max_instances:
                unique_keys = rng.choice(unique_keys, size=max_instances, replace=False)

            instance_registry[entity.entity_name] = {}

            for key_val in unique_keys:
                node_id = f"inst:{entity.entity_name}:{key_val}"
                instance_registry[entity.entity_name][str(key_val)] = node_id

                # Get the row for this key value (first occurrence)
                rows = df[df[rep_col].astype(str) == str(key_val)]
                display_label = (
                    str(rows.iloc[0].get(entity.columns[1], key_val))
                    if len(rows) > 0 and len(entity.columns) > 1
                    else str(key_val)
                )

                G.add_node(node_id, **{
                    "node_type": NodeType.INSTANCE.value,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type,
                    "key_value": str(key_val),
                    "display_label": display_label[:50],  # cap label length
                    "label": f"{entity.entity_name}:{key_val}",
                })

                # Instance → Entity membership edge
                entity_node_id = f"entity:{entity.entity_name}"
                G.add_edge(node_id, entity_node_id, **{
                    "edge_type": "MEMBER_OF",
                    "confidence": 1.0,
                    "weight": 1.0,
                })

        # ── Step 3: Add entity-to-entity relationship edges ───────────────
        for rel in relationships:
            src_node = f"entity:{rel.source_entity}"
            tgt_node = f"entity:{rel.target_entity}"
            if G.has_node(src_node) and G.has_node(tgt_node):
                G.add_edge(src_node, tgt_node, **{
                    "edge_type": rel.relationship_type,
                    "relationship_name": rel.relationship_name,
                    "confidence": rel.confidence,
                    "cardinality": rel.cardinality,
                    "weight": rel.confidence,
                })

        # ── Step 4: Cross-entity instance FK edges (sampled) ─────────────
        self._add_fk_instance_edges(G, df, relationships, instance_registry, rng)

        # ── Step 5: Prune isolated nodes ──────────────────────────────────
        if self._config.graph.isolated_node_removal:
            isolated = list(nx.isolates(G))
            G.remove_nodes_from(isolated)
            self._logger.info(f"Pruned {len(isolated)} isolated nodes")

        # Prune low-confidence edges
        low_conf_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("confidence", 1.0) < self._config.graph.edge_confidence_prune_threshold
        ]
        G.remove_edges_from(low_conf_edges)

        self._logger.info(
            f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        neurl_graph = NEURLGraph(
            graph=G,
            entity_schemas=entity_schemas_map,
            relationship_schemas=relationship_schemas_list,
        )

        # ── Step 6: Community detection ───────────────────────────────────
        neurl_graph.communities = self.detect_communities(G)
        for node, community in neurl_graph.communities.items():
            if G.has_node(node):
                G.nodes[node]["community"] = community

        # ── Step 7: Hierarchy assignment ──────────────────────────────────
        neurl_graph.hierarchy_levels = self.compute_hierarchy(G, entities)
        for node, level in neurl_graph.hierarchy_levels.items():
            if G.has_node(node):
                G.nodes[node]["hierarchy_level"] = level

        # ── Step 8: PyG Data object ───────────────────────────────────────
        if TORCH_GEOMETRIC_AVAILABLE:
            neurl_graph.pyg_data = self.compute_node_features_for_gnn(G)

        return neurl_graph

    def _add_fk_instance_edges(
        self,
        G: nx.DiGraph,
        df: pd.DataFrame,
        relationships: List["RelationshipSchema"],
        instance_registry: Dict[str, Dict[str, str]],
        rng: np.random.Generator,
    ) -> None:
        """Add FK-based cross-entity instance edges (sampled for tractability).

        For each FK relationship, sample pairs of instance nodes that are
        connected by a shared linking column value and add an edge between them.

        Args:
            G: The graph being built (mutated in-place).
            df: Cleaned DataFrame.
            relationships: List of RelationshipSchema.
            instance_registry: entity → {key_val → node_id} lookup.
            rng: NumPy random generator.
        """
        MAX_FK_EDGES_PER_REL = 50  # cap to keep graph tractable

        for rel in relationships:
            col_a, col_b = rel.linking_columns
            if not col_a or not col_b or col_a not in df.columns or col_b not in df.columns:
                continue

            src_instances = instance_registry.get(rel.source_entity, {})
            tgt_instances = instance_registry.get(rel.target_entity, {})
            if not src_instances or not tgt_instances:
                continue

            # Build value → instance_node_id lookup for target
            tgt_by_key: Dict[str, str] = {}
            for key_val, node_id in tgt_instances.items():
                tgt_by_key[key_val] = node_id

            # Find linking rows and add edges
            edge_count = 0
            sample_rows = df.sample(
                min(len(df), MAX_FK_EDGES_PER_REL * 3),
                random_state=int(rng.integers(0, 10000))
            )
            for _, row in sample_rows.iterrows():
                if edge_count >= MAX_FK_EDGES_PER_REL:
                    break
                key_src = str(row.get(rel.linking_columns[0], ""))
                key_tgt = str(row.get(rel.linking_columns[1], ""))
                src_node = src_instances.get(key_src)
                tgt_node = tgt_by_key.get(key_tgt)
                if src_node and tgt_node and G.has_node(src_node) and G.has_node(tgt_node):
                    G.add_edge(src_node, tgt_node, **{
                        "edge_type": rel.relationship_name,
                        "confidence": rel.confidence,
                        "weight": rel.confidence,
                    })
                    edge_count += 1

    # ─── COMMUNITY DETECTION (Louvain) ───────────────────────────────────────

    def detect_communities(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Run Louvain community detection on the knowledge graph.

        Converts the directed graph to undirected for Louvain (which requires
        undirected graphs). Communities reveal natural groupings of nodes
        (e.g., all Customer + Order + Invoice nodes in one community).

        Args:
            graph: The NEURL knowledge graph (directed).

        Returns:
            Dict mapping node_id → community_id integer.
        """
        if not LOUVAIN_AVAILABLE:
            self._logger.warning("Louvain not available — using degree-based fallback")
            # Fallback: assign community by entity type
            return {
                node: hash(data.get("entity_type", "unknown")) % 5
                for node, data in graph.nodes(data=True)
            }

        undirected = graph.to_undirected()
        # Use weight attribute for weighted community detection
        partition = community_louvain.best_partition(
            undirected,
            weight="weight",
            resolution=self._config.graph.louvain_resolution,
            random_state=self._config.random_seed,
        )
        n_communities = len(set(partition.values()))
        self._logger.info(f"Louvain detected {n_communities} communities")
        return partition

    # ─── HIERARCHY COMPUTATION ───────────────────────────────────────────────

    def compute_hierarchy(
        self,
        graph: nx.DiGraph,
        entities: List["EntitySchema"],
    ) -> Dict[str, int]:
        """Compute topological hierarchy levels for entity nodes.

        Entity nodes with no incoming entity-to-entity edges are level 0
        (roots/masters). Each step downstream adds 1 to the level.
        Instance nodes inherit their parent entity's level.

        Args:
            graph: The NEURL knowledge graph.
            entities: List of EntitySchema.

        Returns:
            Dict mapping node_id → hierarchy level integer.
        """
        levels: Dict[str, int] = {}
        entity_names = {e.entity_name for e in entities}

        # Build entity-only subgraph for topological sort
        entity_nodes = [n for n in graph.nodes if graph.nodes[n].get("node_type") == "entity"]
        entity_subgraph = graph.subgraph(entity_nodes).copy()

        # Remove cycles (if any) to enable topological processing
        # Using longest path heuristic for cycle-free level assignment
        for node in entity_nodes:
            # In-degree from entity nodes only
            in_entity_edges = [
                u for u in graph.predecessors(node)
                if graph.nodes[u].get("node_type") == "entity"
            ]
            if not in_entity_edges:
                levels[node] = 0

        # BFS to propagate levels
        from collections import deque
        queue = deque([n for n, lvl in levels.items() if lvl == 0])
        while queue:
            current = queue.popleft()
            current_level = levels.get(current, 0)
            for successor in graph.successors(current):
                if graph.nodes[successor].get("node_type") == "entity":
                    new_level = current_level + 1
                    if successor not in levels or levels[successor] < new_level:
                        levels[successor] = new_level
                        queue.append(successor)

        # Assign levels to entity nodes not yet assigned
        for node in entity_nodes:
            if node not in levels:
                levels[node] = 0

        # Instance nodes inherit parent entity level
        for node, data in graph.nodes(data=True):
            if data.get("node_type") == "instance":
                entity_name = data.get("entity_name", "")
                entity_node_id = f"entity:{entity_name}"
                levels[node] = levels.get(entity_node_id, 0)

        return levels

    # ─── PYTORCH GEOMETRIC DATA OBJECT ──────────────────────────────────────

    def compute_node_features_for_gnn(self, graph: nx.DiGraph) -> "Data":
        """Convert the NetworkX graph to a PyTorch Geometric Data object.

        Node features encode: node_type (one-hot), entity_type (label encoded),
        degree (normalized), pagerank (normalized), community_id, hierarchy_level.
        This gives the GNN a rich initial feature set without requiring raw data.

        Args:
            graph: The NEURL knowledge graph.

        Returns:
            torch_geometric.data.Data with x, edge_index, edge_attr, y fields.

        Raises:
            NEURLGraphError: If PyG is not available.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise NEURLGraphError("torch_geometric is required for GNN data conversion")

        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Pre-compute graph centralities for feature encoding
        try:
            pagerank = nx.pagerank(graph, weight="weight", max_iter=100)
        except Exception:
            pagerank = {n: 1.0 / n_nodes for n in nodes}

        max_degree = max(dict(graph.degree()).values(), default=1)

        # Node type one-hot encoding
        node_type_codes = {"entity": 0, "instance": 1}
        entity_type_codes = {
            "business_object": 0, "document": 1, "transaction": 2,
            "person": 3, "product": 4, "reference": 5, "unknown": 6
        }
        n_entity_types = len(entity_type_codes)

        # Feature vector per node: [node_type_0, node_type_1, entity_type_0..6,
        #                           degree_norm, pagerank_norm, community, level]
        # Total: 2 + 7 + 4 = 13 features
        feature_dim = 2 + n_entity_types + 4
        X = np.zeros((n_nodes, feature_dim), dtype=np.float32)

        for node, data in graph.nodes(data=True):
            i = node_to_idx[node]
            # Node type one-hot
            nt = node_type_codes.get(data.get("node_type", "instance"), 1)
            X[i, nt] = 1.0
            # Entity type one-hot
            et = entity_type_codes.get(data.get("entity_type", "unknown"), 6)
            X[i, 2 + et] = 1.0
            # Degree (normalized)
            X[i, 2 + n_entity_types] = graph.degree(node) / max(max_degree, 1)
            # PageRank
            X[i, 2 + n_entity_types + 1] = float(pagerank.get(node, 0.0))
            # Community ID (normalized)
            X[i, 2 + n_entity_types + 2] = float(data.get("community", 0)) / 10.0
            # Hierarchy level (normalized)
            X[i, 2 + n_entity_types + 3] = float(data.get("hierarchy_level", 0)) / 5.0

        # Build edge_index: [2, E] as source/target index pairs
        edge_index = []
        edge_attr = []
        for u, v, edata in graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edge_index.append([node_to_idx[u], node_to_idx[v]])
                edge_attr.append([edata.get("confidence", 1.0), edata.get("weight", 1.0)])

        if not edge_index:
            # Guard against empty graphs
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.zeros((0, 2), dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

        x_tensor = torch.tensor(X, dtype=torch.float)

        # Node label: node_type code (for node classification head)
        y_tensor = torch.tensor(
            [node_type_codes.get(graph.nodes[n].get("node_type", "instance"), 1)
             for n in nodes],
            dtype=torch.long
        )

        pyg_data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=y_tensor,
            num_nodes=n_nodes,
        )
        pyg_data.node_names = nodes  # Store for lookup

        return pyg_data

    # ─── GRAPH STATISTICS ────────────────────────────────────────────────────

    def graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute comprehensive graph topology statistics.

        Args:
            graph: The NEURL knowledge graph.

        Returns:
            Dict with topology statistics for logging and dashboard display.
        """
        stats: Dict[str, Any] = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "avg_degree": round(
                sum(dict(graph.degree()).values()) / max(graph.number_of_nodes(), 1), 2
            ),
            "density": round(nx.density(graph), 4),
            "num_entity_nodes": sum(
                1 for _, d in graph.nodes(data=True) if d.get("node_type") == "entity"
            ),
            "num_instance_nodes": sum(
                1 for _, d in graph.nodes(data=True) if d.get("node_type") == "instance"
            ),
        }

        # Community count
        communities = nx.get_node_attributes(graph, "community")
        stats["num_communities"] = len(set(communities.values())) if communities else 0

        # Connected component analysis (on undirected projection)
        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        stats["num_connected_components"] = len(components)

        # Diameter on largest component (expensive — only for small graphs)
        largest_cc = max(components, key=len) if components else set()
        if len(largest_cc) > 1 and len(largest_cc) < 500:
            try:
                sub = undirected.subgraph(largest_cc)
                stats["diameter"] = nx.diameter(sub)
            except Exception:
                stats["diameter"] = -1
        else:
            stats["diameter"] = -1

        # Average clustering coefficient
        try:
            stats["clustering_coefficient"] = round(
                nx.average_clustering(undirected), 4
            )
        except Exception:
            stats["clustering_coefficient"] = 0.0

        return stats


# =============================================================================
# DEMO: Build the NEURL knowledge graph from the ERP dataset
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 6: Knowledge Graph Construction")
print("=" * 70)

print("\n[1/3] Building knowledge graph...")
graph_builder = KnowledgeGraphBuilder(neurl_config)
neurl_graph = graph_builder.build(
    df=erp_df_clean,
    entities=erp_entities,
    relationships=erp_relationships,
    schema=erp_schema,
)

# 2. Statistics
print("\n[2/3] Computing graph statistics...")
graph_stats = graph_builder.graph_statistics(neurl_graph.graph)
print("\n  ── Graph Statistics ────────────────────────────────────────────────")
for key, val in graph_stats.items():
    print(f"  {key:<30} : {val}")

# 3. Community summary
print("\n[3/3] Community assignments:")
from collections import Counter
community_counts = Counter(neurl_graph.communities.values())
for comm_id, count in sorted(community_counts.items()):
    # List entity nodes in this community
    entity_nodes_in_comm = [
        neurl_graph.graph.nodes[n]["entity_name"]
        for n in neurl_graph.communities
        if neurl_graph.communities[n] == comm_id
        and neurl_graph.graph.has_node(n)
        and neurl_graph.graph.nodes[n].get("node_type") == "entity"
    ]
    print(f"  Community {comm_id}: {count} nodes | entities: {entity_nodes_in_comm}")

# 4. PyG data summary
if neurl_graph.pyg_data is not None:
    pyg = neurl_graph.pyg_data
    print(f"\n  PyG Data object:")
    print(f"    x.shape       : {pyg.x.shape}")
    print(f"    edge_index.shape: {pyg.edge_index.shape}")
    print(f"    edge_attr.shape : {pyg.edge_attr.shape}")
    print(f"    y.shape       : {pyg.y.shape}")
    print(f"    num_nodes     : {pyg.num_nodes}")

print("\n✅ Cell 6 complete — Knowledge Graph ready.")
print(f"   Exports: graph_builder, neurl_graph, graph_stats")
print(f"   Graph  : {neurl_graph.graph.number_of_nodes()} nodes, "
      f"{neurl_graph.graph.number_of_edges()} edges, "
      f"{graph_stats['num_communities']} communities")
