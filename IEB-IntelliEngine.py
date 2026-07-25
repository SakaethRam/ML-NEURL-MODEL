"""
NEURL Engine — Cell 9: Inference, Explainability & Business Intelligence

Runs full model inference and converts raw GNN outputs into actionable
business intelligence insights. Implements node importance scoring,
anomaly detection, link prediction, and per-node explanations.

Design rationale: Raw GNN embeddings are not human-readable. This layer
translates them into business language — which entities are most central,
which connections are missing, which nodes look anomalous — making the
knowledge graph actionable for non-technical stakeholders.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.stats import zscore


# =============================================================================
# DATA CONTAINERS — InferenceResult and NodeExplanation
# =============================================================================

@dataclass
class NodeExplanation:
    """Explanation for a single node's embedding and graph position.

    Attributes:
        node_id: The explained node's identifier.
        most_similar_nodes: Top-K nodes by cosine similarity to this node.
        contributing_features: Top feature dimensions for this node's embedding.
        attention_weights: Attention weights if model is GAT (empty otherwise).
        subgraph_neighbors: Direct neighbors in the knowledge graph.
        business_interpretation: Rule-based natural language interpretation.
    """
    node_id: str
    most_similar_nodes: List[Tuple[str, float]]   # (node_id, cosine_sim)
    contributing_features: List[Tuple[int, float]] # (feature_idx, magnitude)
    attention_weights: Dict[str, float]            # (neighbor_id, weight)
    subgraph_neighbors: List[str]
    business_interpretation: str


@dataclass
class InferenceResult:
    """Complete inference output from the NEURL Engine.

    Attributes:
        node_embeddings: numpy array [N, D] of all node embeddings.
        predicted_links: Top-K predicted missing links as (src, tgt, confidence).
        node_importance_scores: Combined importance score per node.
        anomaly_scores: Reconstruction-error-based anomaly score per node.
        community_assignments: Louvain community ID per node.
        business_insights: Structured business intelligence report dict.
        node_names: List of node identifier strings (index-aligned with embeddings).
    """
    node_embeddings: np.ndarray
    predicted_links: List[Tuple[str, str, float]]
    node_importance_scores: Dict[str, float]
    anomaly_scores: Dict[str, float]
    community_assignments: Dict[str, int]
    business_insights: Dict[str, Any]
    node_names: List[str] = field(default_factory=list)


# =============================================================================
# NEURLInferenceEngine
# =============================================================================

class NEURLInferenceEngine:
    """Runs full inference and generates business intelligence from GNN embeddings.

    Converts trained GNN outputs into:
    1. Node embeddings (semantic vector space)
    2. Predicted missing links (potential new relationships)
    3. Node importance scores (combined centrality)
    4. Anomaly scores (outlier detection)
    5. Per-node explanations (interpretability)
    6. Business intelligence report (actionable insights)

    Example:
        engine = NEURLInferenceEngine(model, graph, entities, relationships, config, device)
        result = engine.run_full_inference(pyg_data)
        bi_report = result.business_insights
    """

    def __init__(
        self,
        model: "NEURLBaseGNN",
        graph: "NEURLGraph",
        entities: List["EntitySchema"],
        relationships: List["RelationshipSchema"],
        config: "NEURLConfig",
        device: torch.device,
    ) -> None:
        """Initialize the inference engine.

        Args:
            model: Trained NEURLBaseGNN model.
            graph: NEURLGraph from Knowledge Graph Builder.
            entities: Discovered EntitySchema list.
            relationships: Discovered RelationshipSchema list.
            config: NEURLConfig master configuration.
            device: Compute device for model inference.
        """
        self._model = model
        self._graph = graph
        self._entities = entities
        self._relationships = relationships
        self._config = config
        self._device = device
        self._logger = NEURLLogger.get_logger("NEURLInferenceEngine")

    # ─── FULL INFERENCE PIPELINE ─────────────────────────────────────────────

    def run_full_inference(self, pyg_data: "Data") -> InferenceResult:
        """Execute the complete inference pipeline.

        Args:
            pyg_data: PyG Data object (output of Cell 6).

        Returns:
            InferenceResult with all inference outputs.
        """
        self._logger.info("Starting full inference pipeline")

        # 1. Extract node embeddings
        embeddings = self._model.get_embeddings(
            pyg_data.x.to(self._device),
            pyg_data.edge_index.to(self._device),
        )
        node_names = list(getattr(pyg_data, "node_names", [f"node_{i}" for i in range(len(embeddings))]))
        node_to_idx = {name: i for i, name in enumerate(node_names)}

        self._logger.info(f"Extracted embeddings: {embeddings.shape}")

        # 2. Predict missing links
        predicted_links = self.predict_missing_links(
            embeddings,
            node_names,
            threshold=self._config.inference.link_prediction_threshold
        )

        # 3. Node importance scores
        importance_scores = self.compute_node_importance(
            embeddings, node_names, self._graph.graph
        )

        # 4. Anomaly scores
        anomaly_scores = self.detect_anomalies(
            embeddings, node_names, self._graph.graph
        )

        # 5. Community assignments (from graph, computed in Cell 6)
        community_assignments = self._graph.communities

        # 6. Business intelligence report
        bi_report = self.generate_business_intelligence_report(
            embeddings=embeddings,
            node_names=node_names,
            importance_scores=importance_scores,
            anomaly_scores=anomaly_scores,
            community_assignments=community_assignments,
            predicted_links=predicted_links,
        )

        return InferenceResult(
            node_embeddings=embeddings,
            predicted_links=predicted_links,
            node_importance_scores=importance_scores,
            anomaly_scores=anomaly_scores,
            community_assignments=community_assignments,
            business_insights=bi_report,
            node_names=node_names,
        )

    # ─── LINK PREDICTION ─────────────────────────────────────────────────────

    def predict_missing_links(
        self,
        embeddings: np.ndarray,
        node_names: List[str],
        threshold: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        """Predict missing entity-level links using embedding inner products.

        Only considers entity-level nodes (not instance nodes) to avoid
        combinatorial explosion. Filters out links already in the graph.

        Args:
            embeddings: Node embedding matrix [N, D].
            node_names: Node identifier strings.
            threshold: Minimum sigmoid(dot_product) to report a link.

        Returns:
            List of (source_node, target_node, confidence) tuples, sorted
            by confidence descending, limited to top_k_links from config.
        """
        # Filter to entity-level nodes only
        entity_indices = [
            i for i, name in enumerate(node_names)
            if name.startswith("entity:")
        ]

        if len(entity_indices) < 2:
            return []

        entity_embs = embeddings[entity_indices]
        entity_names_filtered = [node_names[i] for i in entity_indices]

        # Compute pairwise dot products and apply sigmoid
        dot_products = entity_embs @ entity_embs.T  # [N_e, N_e]
        # Normalize by embedding norms for cosine similarity
        norms = np.linalg.norm(entity_embs, axis=1, keepdims=True) + 1e-9
        cosine_sim = entity_embs @ entity_embs.T / (norms @ norms.T)

        existing_edges = set(self._graph.graph.edges())
        predicted: List[Tuple[str, str, float]] = []

        n_e = len(entity_indices)
        for i in range(n_e):
            for j in range(i + 1, n_e):
                src = entity_names_filtered[i]
                tgt = entity_names_filtered[j]
                conf = float(1.0 / (1.0 + np.exp(-cosine_sim[i, j])))  # sigmoid

                # Only report links not already in the graph
                if (src, tgt) not in existing_edges and conf >= threshold:
                    predicted.append((src, tgt, conf))

        predicted.sort(key=lambda x: x[2], reverse=True)
        return predicted[:self._config.inference.top_k_links]

    # ─── NODE IMPORTANCE ──────────────────────────────────────────────────────

    def compute_node_importance(
        self,
        embeddings: np.ndarray,
        node_names: List[str],
        graph: nx.DiGraph,
    ) -> Dict[str, float]:
        """Compute a composite node importance score.

        Combines four signals:
        1. Embedding L2 magnitude: high magnitude = node encodes rich information
        2. PageRank: structural importance in the graph
        3. Approximated betweenness centrality: bridging nodes
        4. Degree centrality: connectivity

        Each signal is min-max normalized before weighted combination.

        Args:
            embeddings: Node embeddings [N, D].
            node_names: Node identifier strings.
            graph: The NEURL knowledge graph.

        Returns:
            Dict mapping node_id → normalized importance score [0, 1].
        """
        weights = self._config.inference.importance_weights
        n = len(node_names)
        node_set = set(graph.nodes())

        # Signal 1: Embedding magnitude
        magnitudes = np.linalg.norm(embeddings, axis=1)  # [N]

        # Signal 2: PageRank
        try:
            pr = nx.pagerank(graph, weight="weight", max_iter=100)
        except Exception:
            pr = {node: 1.0 / max(n, 1) for node in graph.nodes()}
        pr_vals = np.array([pr.get(name, 0.0) for name in node_names])

        # Signal 3: Betweenness centrality (k-approximation for scalability)
        try:
            k_approx = min(len(node_set), 50)  # sample-based approximation
            bc = nx.betweenness_centrality(graph, k=k_approx, normalized=True, weight="weight")
        except Exception:
            bc = {node: 0.0 for node in graph.nodes()}
        bc_vals = np.array([bc.get(name, 0.0) for name in node_names])

        # Signal 4: Degree centrality
        try:
            dc = nx.degree_centrality(graph)
        except Exception:
            dc = {node: 0.0 for node in graph.nodes()}
        dc_vals = np.array([dc.get(name, 0.0) for name in node_names])

        def _normalize(arr: np.ndarray) -> np.ndarray:
            """Min-max normalize to [0, 1]."""
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / (rng + 1e-9)

        composite = (
            weights.get("embedding_magnitude", 0.2) * _normalize(magnitudes)
            + weights.get("pagerank", 0.4) * _normalize(pr_vals)
            + weights.get("betweenness", 0.2) * _normalize(bc_vals)
            + weights.get("degree", 0.2) * _normalize(dc_vals)
        )

        # Normalize final composite to [0, 1]
        composite = _normalize(composite)
        return {name: float(composite[i]) for i, name in enumerate(node_names)}

    # ─── ANOMALY DETECTION ───────────────────────────────────────────────────

    def detect_anomalies(
        self,
        embeddings: np.ndarray,
        node_names: List[str],
        graph: nx.DiGraph,
    ) -> Dict[str, float]:
        """Detect anomalous nodes by reconstruction error from neighborhood mean.

        For each node, computes the Euclidean distance between its embedding
        and the mean embedding of its immediate neighbors. Nodes whose
        embeddings are far from their neighborhood average (z-score > threshold)
        are flagged as anomalous — they're structurally unexpected given
        the surrounding graph context.

        Args:
            embeddings: Node embeddings [N, D].
            node_names: Node identifier strings.
            graph: The NEURL knowledge graph.

        Returns:
            Dict mapping node_id → anomaly score (z-score of reconstruction error).
        """
        node_to_idx = {name: i for i, name in enumerate(node_names)}
        reconstruction_errors = np.zeros(len(node_names))

        for i, node in enumerate(node_names):
            if not graph.has_node(node):
                continue
            neighbors = list(graph.predecessors(node)) + list(graph.successors(node))
            neighbor_indices = [
                node_to_idx[n] for n in neighbors
                if n in node_to_idx
            ]
            if not neighbor_indices:
                reconstruction_errors[i] = 0.0
                continue
            # Mean neighborhood embedding
            neighbor_mean = embeddings[neighbor_indices].mean(axis=0)
            # Distance from node's own embedding to neighborhood mean
            reconstruction_errors[i] = float(np.linalg.norm(
                embeddings[i] - neighbor_mean
            ))

        # Convert to z-scores so the threshold is data-adaptive
        if reconstruction_errors.std() > 0:
            z_scores = zscore(reconstruction_errors)
        else:
            z_scores = reconstruction_errors

        return {name: float(z_scores[i]) for i, name in enumerate(node_names)}

    # ─── NODE EXPLANATION ─────────────────────────────────────────────────────

    def explain_node(
        self,
        node_id: str,
        embeddings: np.ndarray,
        node_names: List[str],
        graph: nx.DiGraph,
        top_k: int = 5,
    ) -> NodeExplanation:
        """Generate a human-readable explanation for a specific node.

        Args:
            node_id: Identifier of the node to explain.
            embeddings: Node embeddings [N, D].
            node_names: Node identifier strings.
            graph: The NEURL knowledge graph.
            top_k: Number of similar nodes and features to include.

        Returns:
            NodeExplanation dataclass.
        """
        node_to_idx = {name: i for i, name in enumerate(node_names)}

        if node_id not in node_to_idx:
            return NodeExplanation(
                node_id=node_id,
                most_similar_nodes=[],
                contributing_features=[],
                attention_weights={},
                subgraph_neighbors=[],
                business_interpretation=f"Node '{node_id}' not found in embedding space.",
            )

        idx = node_to_idx[node_id]
        node_emb = embeddings[idx]

        # Most similar nodes: cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        cosine_sims = embeddings @ node_emb / (norms.squeeze() * (np.linalg.norm(node_emb) + 1e-9))
        top_k_idx = np.argsort(cosine_sims)[::-1][1:top_k + 1]
        similar_nodes = [(node_names[k], float(cosine_sims[k])) for k in top_k_idx]

        # Contributing features: top-K dimensions by absolute magnitude
        top_feat_idx = np.argsort(np.abs(node_emb))[::-1][:top_k]
        contrib_features = [(int(k), float(node_emb[k])) for k in top_feat_idx]

        # Subgraph neighbors
        neighbors = []
        if graph.has_node(node_id):
            neighbors = (
                list(graph.predecessors(node_id)) +
                list(graph.successors(node_id))
            )[:10]  # cap at 10

        # Business interpretation: rule-based from node attributes
        node_data = graph.nodes.get(node_id, {})
        entity_name = node_data.get("entity_name", "Unknown")
        entity_type = node_data.get("entity_type", "unknown")
        node_type = node_data.get("node_type", "unknown")
        community = node_data.get("community", "N/A")
        degree = graph.degree(node_id) if graph.has_node(node_id) else 0

        interpretation = (
            f"Node '{node_id}' is a {node_type} of type '{entity_type}' "
            f"belonging to entity '{entity_name}'. "
            f"It is in community {community} with {degree} connections. "
            f"Its top-{top_k} most similar nodes suggest it shares structural "
            f"properties with {[n for n, _ in similar_nodes[:2]]}."
        )

        return NodeExplanation(
            node_id=node_id,
            most_similar_nodes=similar_nodes,
            contributing_features=contrib_features,
            attention_weights={},  # Would be populated for GAT models
            subgraph_neighbors=neighbors,
            business_interpretation=interpretation,
        )

    # ─── BUSINESS INTELLIGENCE REPORT ────────────────────────────────────────

    def generate_business_intelligence_report(
        self,
        embeddings: np.ndarray,
        node_names: List[str],
        importance_scores: Dict[str, float],
        anomaly_scores: Dict[str, float],
        community_assignments: Dict[str, int],
        predicted_links: List[Tuple[str, str, float]],
    ) -> Dict[str, Any]:
        """Generate a structured business intelligence report.

        Synthesizes inference results into six business-ready insight categories.
        Each insight is derived solely from the graph structure and GNN embeddings
        — no hard-coded domain assumptions.

        Args:
            embeddings: Node embeddings [N, D].
            node_names: Node identifier strings.
            importance_scores: Per-node importance scores.
            anomaly_scores: Per-node z-score anomaly scores.
            community_assignments: Per-node community IDs.
            predicted_links: Predicted missing links.

        Returns:
            Dict with six structured insight sections.
        """
        graph = self._graph.graph
        threshold = self._config.inference.anomaly_zscore_threshold
        top_k = 5

        # 1. Most influential nodes: top-5 by importance score (entity nodes only)
        entity_importance = {
            name: score for name, score in importance_scores.items()
            if graph.has_node(name) and graph.nodes[name].get("node_type") == "entity"
        }
        top_influential = sorted(entity_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        most_influential = [
            {
                "node": name,
                "entity_name": graph.nodes[name].get("entity_name", name),
                "entity_type": graph.nodes[name].get("entity_type", "unknown"),
                "importance_score": round(score, 4),
            }
            for name, score in top_influential
        ]

        # 2. Missing connections: top predicted links not in the current graph
        missing_connections = [
            {"source": src, "target": tgt, "confidence": round(conf, 4)}
            for src, tgt, conf in predicted_links[:top_k]
        ]

        # 3. Anomalous entities: entity nodes with anomaly z-score > threshold
        anomalous = [
            {
                "node": name,
                "entity_name": graph.nodes[name].get("entity_name", name) if graph.has_node(name) else name,
                "anomaly_score": round(score, 4),
            }
            for name, score in anomaly_scores.items()
            if abs(score) > threshold
            and graph.has_node(name)
            and graph.nodes[name].get("node_type") == "entity"
        ]
        anomalous.sort(key=lambda x: abs(x["anomaly_score"]), reverse=True)

        # 4. Community summary: entity distribution per community
        community_entity_map: Dict[int, List[str]] = {}
        for node, comm_id in community_assignments.items():
            if graph.has_node(node) and graph.nodes[node].get("node_type") == "entity":
                entity_name = graph.nodes[node].get("entity_name", node)
                community_entity_map.setdefault(comm_id, []).append(entity_name)

        community_summary = [
            {
                "community_id": comm_id,
                "entity_count": len(entities),
                "entities": entities,
            }
            for comm_id, entities in sorted(community_entity_map.items())
        ]

        # 5. Workflow bottlenecks: high betweenness + low out-degree entity nodes
        try:
            bc = nx.betweenness_centrality(graph, normalized=True, weight="weight")
        except Exception:
            bc = {}

        bottlenecks = []
        for node in graph.nodes():
            if graph.nodes[node].get("node_type") != "entity":
                continue
            out_deg = graph.out_degree(node)
            between = bc.get(node, 0.0)
            # Bottleneck: high betweenness but limited outgoing connections
            if between > 0.1 and out_deg <= 1:
                bottlenecks.append({
                    "node": node,
                    "entity_name": graph.nodes[node].get("entity_name", node),
                    "betweenness": round(between, 4),
                    "out_degree": out_deg,
                })
        bottlenecks.sort(key=lambda x: x["betweenness"], reverse=True)

        # 6. Graph health score: composite 0-1 metric
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        anomaly_rate = len(anomalous) / max(len(self._entities), 1)
        n_communities = len(set(community_assignments.values()))
        community_balance = min(n_communities / max(len(self._entities), 1), 1.0)
        connectivity_score = min(n_edges / max(n_nodes, 1), 1.0)

        graph_health_score = (
            0.4 * connectivity_score
            + 0.4 * community_balance
            + 0.2 * max(0.0, 1.0 - anomaly_rate)
        )

        return {
            "most_influential_nodes": most_influential,
            "missing_connections": missing_connections,
            "anomalous_entities": anomalous,
            "community_summary": community_summary,
            "workflow_bottlenecks": bottlenecks,
            "graph_health_score": round(float(graph_health_score), 4),
        }


# =============================================================================
# DEMO: Run full inference on the trained GNN
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 9: Inference, Explainability & Business Intelligence")
print("=" * 70)

print("\n[1/4] Running full inference pipeline...")
inference_engine = NEURLInferenceEngine(
    model=primary_model,
    graph=neurl_graph,
    entities=erp_entities,
    relationships=erp_relationships,
    config=neurl_config,
    device=device_manager.device,
)

inference_result = inference_engine.run_full_inference(pyg_data)

print(f"\n  Embeddings shape  : {inference_result.node_embeddings.shape}")
print(f"  Predicted links   : {len(inference_result.predicted_links)}")
print(f"  Anomaly nodes     : {sum(1 for s in inference_result.anomaly_scores.values() if abs(s) > neurl_config.inference.anomaly_zscore_threshold)}")
print(f"  Communities       : {len(set(inference_result.community_assignments.values()))}")

print("\n[2/4] Business Intelligence Report:")
bi = inference_result.business_insights
print(f"\n  📊 Graph Health Score: {bi['graph_health_score']:.4f}")

print("\n  🏆 Most Influential Entities:")
for item in bi["most_influential_nodes"]:
    print(f"     {item['entity_name']:<16} | type: {item['entity_type']:<18} | importance: {item['importance_score']:.4f}")

print("\n  🔗 Predicted Missing Connections:")
for item in bi["missing_connections"][:5]:
    src_label = item['source'].replace('entity:', '')
    tgt_label = item['target'].replace('entity:', '')
    print(f"     {src_label:<16} → {tgt_label:<16} | confidence: {item['confidence']:.4f}")

print("\n  ⚠️  Anomalous Entities:")
if bi["anomalous_entities"]:
    for item in bi["anomalous_entities"][:3]:
        print(f"     {item['entity_name']:<16} | z-score: {item['anomaly_score']:.4f}")
else:
    print("     None detected (all entities within normal range)")

print("\n  🏘️  Community Distribution:")
for comm in bi["community_summary"]:
    print(f"     Community {comm['community_id']}: {comm['entities']}")

print("\n  🚧 Workflow Bottlenecks:")
if bi["workflow_bottlenecks"]:
    for b in bi["workflow_bottlenecks"][:3]:
        print(f"     {b['entity_name']:<16} | betweenness: {b['betweenness']:.4f} | out_degree: {b['out_degree']}")
else:
    print("     None identified")

# Node explanation for the first entity node
print("\n[3/4] Node explanation example:")
entity_nodes = [n for n in inference_result.node_names if n.startswith("entity:")]
if entity_nodes:
    explain_node_id = entity_nodes[0]
    explanation = inference_engine.explain_node(
        node_id=explain_node_id,
        embeddings=inference_result.node_embeddings,
        node_names=inference_result.node_names,
        graph=neurl_graph.graph,
        top_k=neurl_config.inference.top_k_explanations,
    )
    print(f"\n  Node: {explanation.node_id}")
    print(f"  Interpretation: {explanation.business_interpretation}")
    print(f"  Most similar nodes: {explanation.most_similar_nodes[:3]}")
    print(f"  Direct neighbors: {explanation.subgraph_neighbors[:5]}")

print("\n[4/4] Summary statistics:")
importance_arr = np.array(list(inference_result.node_importance_scores.values()))
anomaly_arr = np.array(list(inference_result.anomaly_scores.values()))
print(f"  Importance: mean={importance_arr.mean():.4f}, std={importance_arr.std():.4f}")
print(f"  Anomaly z-scores: mean={anomaly_arr.mean():.4f}, std={anomaly_arr.std():.4f}")

print("\n✅ Cell 9 complete — Inference & Business Intelligence ready.")
print(f"   Exports: inference_engine, inference_result, bi (business insights dict)")
