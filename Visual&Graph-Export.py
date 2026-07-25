"""
NEURL Engine — Cell 10: Visualization & Graph Export

Implements NEURLVisualizer (5 interactive Plotly figures) and NEURLExporter
(JSON, GraphML, Cytoscape.js, Markdown model card).

The final cell also runs the complete end-to-end pipeline demo, confirming
all 10 cells work together as an integrated system.

Design rationale: Publication-grade interactive visualizations make the
knowledge graph accessible to non-technical stakeholders. Export formats
enable downstream consumption by graph databases, web apps, and reports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import networkx as nx


# =============================================================================
# NEURLVisualizer — interactive Plotly knowledge graph visualizations
# =============================================================================

class NEURLVisualizer:
    """Generates interactive Plotly visualizations for the NEURL Engine.

    Produces five visualizations:
    1. Knowledge graph: entity + instance nodes, colored by type/community
    2. Embedding space: UMAP 2D projection of node embeddings
    3. Training dashboard: loss, AUC, silhouette over epochs
    4. Entity-relationship matrix: heatmap of inter-entity link strengths
    5. Business intelligence dashboard: importance, anomalies, communities

    Example:
        viz = NEURLVisualizer(config)
        fig = viz.render_knowledge_graph(neurl_graph, inference_result)
    """

    # Color palette: entity type → hex color
    _ENTITY_TYPE_COLORS: Dict[str, str] = {
        "business_object": "#3D5A80",
        "document": "#6B8F71",
        "transaction": "#B07D62",
        "person": "#8C6A9E",
        "product": "#C9A227",
        "reference": "#4C8C8C",
        "unknown": "#5A5A5A",
    }

    # Community palette (cycling for more than N_COLORS communities)
    _COMMUNITY_COLORS = [
        "#3D5A80", "#B07D62", "#6B8F71", "#8C6A9E", "#C9A227",
        "#4C8C8C", "#9E4A4A", "#5A5A5A", "#7B9E87", "#A07855",
    ]

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the visualizer.

        Args:
            config: NEURLConfig master configuration.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("NEURLVisualizer")

    # ─── 1. KNOWLEDGE GRAPH ──────────────────────────────────────────────────

    def render_knowledge_graph(
        self,
        neurl_graph: "NEURLGraph",
        inference_result: "InferenceResult",
        layout: str = "spring",
    ) -> "go.Figure":
        """Render an interactive knowledge graph visualization.

        Uses a NetworkX layout algorithm to compute 2D node positions, then
        renders entity nodes (large circles) and instance nodes (small dots)
        with hover tooltips and annotations for top-5 important nodes.

        Args:
            neurl_graph: NEURLGraph from Cell 6.
            inference_result: InferenceResult from Cell 9.
            layout: Layout algorithm: 'spring', 'kamada_kawai', or 'hierarchical'.

        Returns:
            Plotly Figure with the interactive knowledge graph.
        """
        G = neurl_graph.graph
        self._logger.info(f"Rendering knowledge graph with {layout} layout")

        # Compute node positions
        pos = self._compute_layout(G, layout)

        # Separate entity and instance nodes
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
        instance_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "instance"]

        # Truncate instance nodes for visual clarity
        MAX_INSTANCE_VIZ = 100
        if len(instance_nodes) > MAX_INSTANCE_VIZ:
            rng = np.random.default_rng(self._config.random_seed)
            instance_nodes = list(rng.choice(instance_nodes, MAX_INSTANCE_VIZ, replace=False))

        traces = []

        # Entity node traces (one per entity type for legend grouping)
        entity_type_groups: Dict[str, List[str]] = {}
        for node in entity_nodes:
            etype = G.nodes[node].get("entity_type", "unknown")
            entity_type_groups.setdefault(etype, []).append(node)

        for etype, nodes in entity_type_groups.items():
            color = self._ENTITY_TYPE_COLORS.get(etype, "#5A5A5A")
            xs = [pos[n][0] for n in nodes if n in pos]
            ys = [pos[n][1] for n in nodes if n in pos]
            hover_texts = [
                f"<b>{G.nodes[n].get('entity_name', n)}</b><br>"
                f"Type: {G.nodes[n].get('entity_type', 'N/A')}<br>"
                f"Columns: {G.nodes[n].get('column_count', 'N/A')}<br>"
                f"Community: {G.nodes[n].get('community', 'N/A')}<br>"
                f"Importance: {inference_result.node_importance_scores.get(n, 0):.3f}<br>"
                f"Anomaly z: {inference_result.anomaly_scores.get(n, 0):.3f}"
                for n in nodes if n in pos
            ]
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                marker=dict(size=28, color=color, line=dict(width=2, color="white")),
                text=[G.nodes[n].get("entity_name", n) for n in nodes if n in pos],
                textposition="top center",
                textfont=dict(size=10, color="#222"),
                hovertext=hover_texts,
                hoverinfo="text",
                name=etype.replace("_", " ").title(),
                legendgroup=etype,
            ))

        # Instance node trace (colored by community)
        if instance_nodes:
            community_colors_inst = [
                self._COMMUNITY_COLORS[
                    G.nodes[n].get("community", 0) % len(self._COMMUNITY_COLORS)
                ]
                for n in instance_nodes if n in pos
            ]
            xs_inst = [pos[n][0] for n in instance_nodes if n in pos]
            ys_inst = [pos[n][1] for n in instance_nodes if n in pos]
            hover_inst = [
                f"<b>{G.nodes[n].get('display_label', n)}</b><br>"
                f"Entity: {G.nodes[n].get('entity_name', 'N/A')}<br>"
                f"Community: {G.nodes[n].get('community', 'N/A')}"
                for n in instance_nodes if n in pos
            ]
            traces.append(go.Scatter(
                x=xs_inst, y=ys_inst,
                mode="markers",
                marker=dict(size=6, color=community_colors_inst,
                            line=dict(width=0.5, color="white"), opacity=0.7),
                hovertext=hover_inst,
                hoverinfo="text",
                name="Instances",
            ))

        # Edge traces: entity-to-entity edges
        edge_x, edge_y = [], []
        for u, v, edata in G.edges(data=True):
            if u in pos and v in pos:
                if G.nodes[u].get("node_type") == "entity" and G.nodes[v].get("node_type") == "entity":
                    edge_x += [pos[u][0], pos[v][0], None]
                    edge_y += [pos[u][1], pos[v][1], None]

        if edge_x:
            traces.insert(0, go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=1.5, color="#AAAAAA"),
                hoverinfo="none",
                showlegend=False,
                name="Edges",
            ))

        # Annotations for top-5 most important entity nodes
        annotations = []
        top_nodes = sorted(
            [(n, inference_result.node_importance_scores.get(n, 0)) for n in entity_nodes if n in pos],
            key=lambda x: x[1], reverse=True
        )[:5]
        for node, score in top_nodes:
            entity_name = G.nodes[node].get("entity_name", node)
            annotations.append(dict(
                x=pos[node][0], y=pos[node][1],
                text=f"★ {entity_name}",
                showarrow=False,
                font=dict(size=9, color="#9E4A4A"),
                xanchor="left", yanchor="bottom",
            ))

        fig_kg = go.Figure(data=traces)
        fig_kg.update_layout(
            title=dict(text="NEURL Engine — Knowledge Graph", font=dict(size=16, color="#222")),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(x=1.01, y=0.5, bgcolor="white", bordercolor="#DDD", borderwidth=1),
            width=1000, height=700,
            annotations=annotations,
        )
        fig_kg.show()
        return fig_kg

    def _compute_layout(self, G: nx.DiGraph, layout: str) -> Dict[str, Tuple[float, float]]:
        """Compute 2D node positions using the specified layout algorithm.

        Args:
            G: NetworkX DiGraph.
            layout: 'spring', 'kamada_kawai', or 'hierarchical'.

        Returns:
            Dict mapping node → (x, y) position.
        """
        undirected = G.to_undirected()
        if layout == "spring":
            return nx.spring_layout(undirected, seed=self._config.random_seed, k=2.0)
        elif layout == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(undirected, weight="weight")
            except Exception:
                return nx.spring_layout(undirected, seed=self._config.random_seed)
        elif layout == "hierarchical":
            # Assign x by hierarchy level, y by community
            pos = {}
            for node, data in G.nodes(data=True):
                level = data.get("hierarchy_level", 0)
                comm = data.get("community", 0)
                pos[node] = (float(level), float(comm) + np.random.uniform(-0.4, 0.4))
            return pos
        else:
            return nx.spring_layout(undirected, seed=self._config.random_seed)

    # ─── 2. EMBEDDING SPACE (UMAP) ───────────────────────────────────────────

    def render_embedding_space(
        self,
        embeddings: np.ndarray,
        node_names: List[str],
        node_types: List[str],
        graph: nx.DiGraph,
    ) -> "go.Figure":
        """Render a UMAP 2D projection of node embeddings.

        Reveals the learned semantic structure — nodes of the same entity
        type should cluster together if training was successful.

        Args:
            embeddings: Node embeddings [N, D].
            node_names: Node identifier strings.
            node_types: Node type label per node.
            graph: Knowledge graph for hover label enrichment.

        Returns:
            Plotly Figure with the UMAP scatter plot.
        """
        # UMAP projection
        if UMAP_AVAILABLE and len(embeddings) > 5:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(embeddings) - 1),
                random_state=self._config.random_seed,
            )
            try:
                coords_2d = reducer.fit_transform(embeddings)
            except Exception:
                coords_2d = embeddings[:, :2]
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self._config.random_seed)
            coords_2d = pca.fit_transform(embeddings)

        # Color by entity type
        type_set = list(set(node_types))
        color_map = {t: self._ENTITY_TYPE_COLORS.get(t, "#5A5A5A") for t in type_set}
        colors = [color_map.get(t, "#5A5A5A") for t in node_types]

        hover_labels = [
            f"{name}<br>Type: {ntype}"
            for name, ntype in zip(node_names, node_types)
        ]

        fig_emb = go.Figure(data=go.Scatter(
            x=coords_2d[:, 0], y=coords_2d[:, 1],
            mode="markers",
            marker=dict(
                size=[14 if "entity:" in n else 5 for n in node_names],
                color=colors,
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=hover_labels,
            hoverinfo="text",
        ))
        fig_emb.update_layout(
            title=dict(text="NEURL Engine — Node Embedding Space (UMAP/PCA 2D)", font=dict(size=16)),
            xaxis=dict(title="UMAP Dim 1", showgrid=True, gridcolor="#EEE"),
            yaxis=dict(title="UMAP Dim 2", showgrid=True, gridcolor="#EEE"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            width=800, height=600,
        )
        fig_emb.show()
        return fig_emb

    # ─── 3. TRAINING DASHBOARD ───────────────────────────────────────────────

    def render_training_dashboard(
        self,
        tracker: "ExperimentTracker",
    ) -> "go.Figure":
        """Render a multi-panel training metrics dashboard.

        Args:
            tracker: ExperimentTracker with logged metrics.

        Returns:
            Plotly Figure with loss, AUC, and silhouette panels.
        """
        metrics = tracker.all_metrics
        available = [m for m in ["train_loss", "auc", "silhouette"] if m in metrics]

        n_panels = max(len(available), 1)
        fig_dash = make_subplots(rows=1, cols=n_panels,
                                  subplot_titles=[m.replace("_", " ").title() for m in available])

        metric_colors = {"train_loss": "#3D5A80", "auc": "#6B8F71", "silhouette": "#B07D62"}

        for col_idx, metric_name in enumerate(available, start=1):
            epoch_data = sorted(metrics[metric_name].items())
            xs = [e for e, _ in epoch_data]
            ys = [v for _, v in epoch_data]
            fig_dash.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines+markers",
                           name=metric_name.replace("_", " ").title(),
                           line=dict(color=metric_colors.get(metric_name, "#5A5A5A"), width=2),
                           marker=dict(size=4)),
                row=1, col=col_idx
            )
            fig_dash.update_xaxes(title_text="Epoch", row=1, col=col_idx)

        fig_dash.update_layout(
            title=dict(text="NEURL Engine — Training Dashboard", font=dict(size=16)),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            height=350, width=900,
        )
        fig_dash.show()
        return fig_dash

    # ─── 4. ENTITY-RELATIONSHIP MATRIX ──────────────────────────────────────

    def render_entity_relationship_matrix(
        self,
        entities: List["EntitySchema"],
        relationships: List["RelationshipSchema"],
    ) -> "go.Figure":
        """Render a heatmap of entity-to-entity relationship strengths.

        Args:
            entities: Discovered EntitySchema list.
            relationships: Discovered RelationshipSchema list.

        Returns:
            Plotly heatmap Figure.
        """
        names = [e.entity_name for e in entities]
        n = len(names)
        name_to_idx = {name: i for i, name in enumerate(names)}
        matrix = np.zeros((n, n), dtype=float)

        for rel in relationships:
            i = name_to_idx.get(rel.source_entity, -1)
            j = name_to_idx.get(rel.target_entity, -1)
            if i >= 0 and j >= 0:
                matrix[i, j] = max(matrix[i, j], rel.confidence)

        fig_rel = go.Figure(data=go.Heatmap(
            z=matrix,
            x=names,
            y=names,
            colorscale="Blues",
            zmin=0, zmax=1,
            colorbar=dict(title="Confidence"),
            text=np.round(matrix, 2),
            hovertemplate="Source: %{y}<br>Target: %{x}<br>Confidence: %{z:.3f}<extra></extra>",
        ))
        fig_rel.update_layout(
            title=dict(text="NEURL Engine — Entity Relationship Strength Matrix", font=dict(size=16)),
            xaxis=dict(tickangle=-45, title="Target Entity"),
            yaxis=dict(title="Source Entity"),
            paper_bgcolor="white",
            width=700, height=600,
        )
        fig_rel.show()
        return fig_rel

    # ─── 5. BUSINESS INTELLIGENCE DASHBOARD ──────────────────────────────────

    def render_business_intelligence_dashboard(
        self,
        bi_report: Dict[str, Any],
        entities: List["EntitySchema"],
    ) -> "go.Figure":
        """Render a multi-panel business intelligence dashboard.

        Args:
            bi_report: Business intelligence report dict from Cell 9.
            entities: Discovered EntitySchema list.

        Returns:
            Plotly Figure with importance, anomaly, and community panels.
        """
        fig_bi = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Node Importance", "Anomaly Scores", "Community Sizes"),
        )

        # Panel 1: Node importance bar chart
        influential = bi_report.get("most_influential_nodes", [])
        if influential:
            names_imp = [item["entity_name"] for item in influential]
            scores_imp = [item["importance_score"] for item in influential]
            fig_bi.add_trace(
                go.Bar(x=names_imp, y=scores_imp,
                       marker_color="#3D5A80", name="Importance"),
                row=1, col=1
            )

        # Panel 2: Anomaly scores bar chart (absolute value)
        anomalous = bi_report.get("anomalous_entities", [])
        if anomalous:
            names_anom = [item["entity_name"] for item in anomalous[:8]]
            scores_anom = [abs(item["anomaly_score"]) for item in anomalous[:8]]
            fig_bi.add_trace(
                go.Bar(x=names_anom, y=scores_anom,
                       marker_color="#9E4A4A", name="Anomaly |z|"),
                row=1, col=2
            )
        else:
            fig_bi.add_trace(
                go.Bar(x=["None"], y=[0], marker_color="#9E4A4A", name="Anomaly"),
                row=1, col=2
            )

        # Panel 3: Community size distribution
        comm_summary = bi_report.get("community_summary", [])
        if comm_summary:
            comm_ids = [f"C{item['community_id']}" for item in comm_summary]
            comm_sizes = [item["entity_count"] for item in comm_summary]
            fig_bi.add_trace(
                go.Bar(x=comm_ids, y=comm_sizes,
                       marker_color="#6B8F71", name="Community Size"),
                row=1, col=3
            )

        fig_bi.update_layout(
            title=dict(text=f"NEURL Engine — Business Intelligence Dashboard "
                            f"(Health: {bi_report.get('graph_health_score', 0):.2f})",
                        font=dict(size=14)),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            height=400, width=1000,
        )
        fig_bi.show()
        return fig_bi


# =============================================================================
# NEURLExporter — multi-format graph export
# =============================================================================

class NEURLExporter:
    """Exports the NEURL knowledge graph in multiple formats.

    Supports: JSON (universal), GraphML (graph databases), Cytoscape.js
    (web visualization), and Markdown model card (documentation).

    Example:
        exporter = NEURLExporter(config)
        graph_json = exporter.to_json(neurl_graph, inference_result)
    """

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the exporter.

        Args:
            config: NEURLConfig master configuration.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("NEURLExporter")

    def to_json(
        self,
        neurl_graph: "NEURLGraph",
        inference_result: "InferenceResult",
    ) -> Dict[str, Any]:
        """Serialize the knowledge graph and inference results as JSON.

        Produces a complete, self-contained JSON representation with nodes
        (attributes + embedding coordinates), edges (type, confidence), and
        metadata. Suitable for REST APIs and downstream data consumers.

        Args:
            neurl_graph: NEURLGraph from Cell 6.
            inference_result: InferenceResult from Cell 9.

        Returns:
            JSON-serializable dict.
        """
        G = neurl_graph.graph

        # Get 2D layout for export coordinates
        pos_2d = nx.spring_layout(G.to_undirected(), seed=self._config.random_seed)
        node_name_idx = {n: i for i, n in enumerate(inference_result.node_names)}

        nodes_list = []
        for node, data in G.nodes(data=True):
            idx = node_name_idx.get(node, -1)
            emb_2d = pos_2d.get(node, (0.0, 0.0))
            nodes_list.append({
                "id": node,
                "label": data.get("label", data.get("entity_name", node)),
                "type": data.get("node_type", "unknown"),
                "entity_type": data.get("entity_type", "unknown"),
                "entity_name": data.get("entity_name", ""),
                "community": int(data.get("community", 0)),
                "hierarchy_level": int(data.get("hierarchy_level", 0)),
                "importance": round(inference_result.node_importance_scores.get(node, 0.0), 4),
                "anomaly_score": round(inference_result.anomaly_scores.get(node, 0.0), 4),
                "x": round(float(emb_2d[0]), 6),
                "y": round(float(emb_2d[1]), 6),
                "attributes": {k: str(v) for k, v in data.items()
                               if k not in {"label", "community", "hierarchy_level"}},
            })

        edges_list = []
        for u, v, edata in G.edges(data=True):
            edges_list.append({
                "source": u,
                "target": v,
                "type": edata.get("edge_type", "RELATED_TO"),
                "confidence": round(float(edata.get("confidence", 1.0)), 4),
                "name": edata.get("relationship_name", ""),
                "weight": round(float(edata.get("weight", 1.0)), 4),
            })

        out = {
            "neurl_version": self._config.version,
            "construction_timestamp": neurl_graph.construction_timestamp,
            "graph_metadata": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "num_entities": len(neurl_graph.entity_schemas),
                "num_communities": len(set(neurl_graph.communities.values())),
                "architecture": self._config.gnn.architecture.value,
            },
            "entities": [e.to_dict() for e in neurl_graph.entity_schemas.values()],
            "relationships": [r.to_dict() for r in neurl_graph.relationship_schemas],
            "nodes": nodes_list,
            "edges": edges_list,
        }
        return out

    def to_graphml(self, neurl_graph: "NEURLGraph", path: Path) -> None:
        """Export the graph in GraphML format for graph databases.

        GraphML is supported by Gephi, Cytoscape, and most graph databases.
        Node attributes are stringified for compatibility.

        Args:
            neurl_graph: NEURLGraph from Cell 6.
            path: Output file path.
        """
        # GraphML requires scalar string/int/float attributes
        G_export = neurl_graph.graph.copy()
        for node, data in G_export.nodes(data=True):
            for key, val in list(data.items()):
                if isinstance(val, list):
                    data[key] = ",".join(str(v) for v in val)
                elif not isinstance(val, (str, int, float, bool)):
                    data[key] = str(val)

        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G_export, str(path))
        self._logger.info(f"GraphML exported: {path}")

    def to_cytoscape_json(
        self,
        neurl_graph: "NEURLGraph",
        inference_result: "InferenceResult",
    ) -> Dict[str, Any]:
        """Export in Cytoscape.js format for web visualization.

        Cytoscape.js uses an elements array with nodes and edges dicts.
        Each node carries its full attribute set in the 'data' field.

        Args:
            neurl_graph: NEURLGraph from Cell 6.
            inference_result: InferenceResult from Cell 9.

        Returns:
            Cytoscape.js-compatible dict with 'elements' key.
        """
        G = neurl_graph.graph
        pos_2d = nx.spring_layout(G.to_undirected(), seed=self._config.random_seed)

        elements = []

        for node, data in G.nodes(data=True):
            position = pos_2d.get(node, (0, 0))
            elements.append({
                "data": {
                    "id": node,
                    "label": data.get("entity_name", node),
                    "node_type": data.get("node_type", "unknown"),
                    "entity_type": data.get("entity_type", "unknown"),
                    "community": int(data.get("community", 0)),
                    "importance": round(
                        inference_result.node_importance_scores.get(node, 0.0), 4
                    ),
                    "anomaly_score": round(
                        inference_result.anomaly_scores.get(node, 0.0), 4
                    ),
                },
                "position": {
                    "x": round(float(position[0]) * 1000, 2),
                    "y": round(float(position[1]) * 1000, 2),
                },
            })

        for u, v, edata in G.edges(data=True):
            elements.append({
                "data": {
                    "id": f"{u}__{v}",
                    "source": u,
                    "target": v,
                    "edge_type": edata.get("edge_type", "RELATED_TO"),
                    "confidence": round(float(edata.get("confidence", 1.0)), 4),
                    "label": edata.get("relationship_name", ""),
                },
            })

        return {"elements": elements}

    def to_model_card(
        self,
        entities: List["EntitySchema"],
        relationships: List["RelationshipSchema"],
        bi_report: Dict[str, Any],
        tracker: "ExperimentTracker",
    ) -> str:
        """Generate a Markdown model card documenting the NEURL Engine run.

        Args:
            entities: Discovered EntitySchema list.
            relationships: Discovered RelationshipSchema list.
            bi_report: Business intelligence report from Cell 9.
            tracker: ExperimentTracker with training metrics.

        Returns:
            Markdown model card string.
        """
        metrics = tracker.all_metrics
        best_auc = 0.0
        best_epoch = 0
        try:
            best_epoch = tracker.get_best_epoch("auc")
            best_auc = metrics.get("auc", {}).get(best_epoch, 0.0)
        except ValueError:
            pass

        entity_rows = "\n".join(
            f"| {e.entity_name} | {e.entity_type} | {len(e.columns)} | {e.confidence:.3f} | {e.representative_column} |"
            for e in entities
        )
        rel_rows = "\n".join(
            f"| {r.source_entity} | {r.target_entity} | {r.relationship_name} | {r.cardinality} | {r.confidence:.3f} |"
            for r in relationships
        )

        model_card = f"""# NEURL Engine — Model Card

**Version**: {neurl_config.version}
**Timestamp**: {datetime.utcnow().isoformat()}
**Architecture**: {neurl_config.gnn.architecture.value}

---

## Dataset Summary

- **Rows**: {neurl_config.data.sample_size}
- **Columns**: See entity table below
- **Entities Discovered**: {len(entities)}
- **Relationships Discovered**: {len(relationships)}

---

## Discovered Entities

| Entity | Type | Columns | Confidence | Representative Col |
|--------|------|---------|------------|-------------------|
{entity_rows}

---

## Discovered Relationships

| Source | Target | Name | Cardinality | Confidence |
|--------|--------|------|-------------|------------|
{rel_rows}

---

## Training Metrics

| Metric | Value |
|--------|-------|
| Best AUC | {best_auc:.4f} |
| Best Epoch | {best_epoch} |
| Epochs Run | {len(metrics.get('train_loss', {}))} |
| Architecture | {neurl_config.gnn.architecture.value} |
| Hidden Dim | {neurl_config.gnn.hidden_dim} |
| Output Dim | {neurl_config.gnn.out_dim} |
| Layers | {neurl_config.gnn.num_layers} |
| LR | {neurl_config.training.lr} |

---

## Graph Statistics

| Statistic | Value |
|-----------|-------|
| Graph Health Score | {bi_report.get('graph_health_score', 0):.4f} |
| Communities | {len(bi_report.get('community_summary', []))} |
| Missing Links Predicted | {len(bi_report.get('missing_connections', []))} |
| Anomalous Entities | {len(bi_report.get('anomalous_entities', []))} |

---

## Business Insights

**Most Influential Entities**: {[item['entity_name'] for item in bi_report.get('most_influential_nodes', [])]}

**Anomalous Entities**: {[item['entity_name'] for item in bi_report.get('anomalous_entities', [])]}

**Predicted Missing Connections**:
{chr(10).join(f"- {item['source']} → {item['target']} (confidence: {item['confidence']:.3f})" for item in bi_report.get('missing_connections', [])[:5])}

---

## Limitations & Assumptions

- Entity discovery uses statistical and linguistic heuristics — results may vary
  on datasets with non-standard column naming conventions.
- Relationship confidence scores are based on data patterns, not ground-truth FK constraints.
- GNN training uses self-supervised objectives (no labeled graph data required).
- VGAE KL loss requires sufficient graph connectivity to learn a meaningful prior.

---

*Generated by NEURL Engine v{neurl_config.version} — LLM-Free, Pure ML Knowledge Graph System*
"""
        return model_card


# =============================================================================
# FINAL END-TO-END PIPELINE DEMO
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 10: Visualization & Graph Export")
print("=" * 70)

visualizer = NEURLVisualizer(neurl_config)
exporter = NEURLExporter(neurl_config)

print("\n╔══════════════════════════════════════════════════════════════════════╗")
print("║         NEURL Engine — Full End-to-End Pipeline Demo                 ║")
print("╚══════════════════════════════════════════════════════════════════════╝\n")

print("  Step 1/9 ✅  Ingest → Validate → Clean")
print(f"             {raw_erp_df.shape[0]} rows × {raw_erp_df.shape[1]} columns → "
      f"cleaned to {erp_df_clean.shape[0]} rows × {erp_df_clean.shape[1]} columns")

print("\n  Step 2/9 ✅  Feature Engineering")
print(f"             Column feature matrix: {erp_feature_matrix.shape}")

print("\n  Step 3/9 ✅  Node Intelligence → Discover Entities")
print(f"             {len(erp_entities)} entities: {[e.entity_name for e in erp_entities]}")

print("\n  Step 4/9 ✅  Relationship Intelligence → Discover Relationships")
print(f"             {len(erp_relationships)} relationships discovered")

print("\n  Step 5/9 ✅  Build Knowledge Graph")
print(f"             {neurl_graph.graph.number_of_nodes()} nodes, "
      f"{neurl_graph.graph.number_of_edges()} edges")

print("\n  Step 6/9 ✅  Train GNN (GCN, 30 epochs)")
print(f"             Best AUC: {experiment_tracker.get_metric_history('auc').get(experiment_tracker.get_best_epoch('auc') if experiment_tracker.get_metric_history('auc') else 1, 0.0):.4f}")

print("\n  Step 7/9 ✅  Run Inference")
print(f"             Embeddings: {inference_result.node_embeddings.shape}")
print(f"             Predicted links: {len(inference_result.predicted_links)}")

print("\n  Step 8/9 ✅  Business Intelligence Report")
print(f"             Graph health score: {bi['graph_health_score']:.4f}")

# Step 9: Render all 5 visualizations
print("\n  Step 9/9 — Rendering 5 visualizations...")

# 1. Knowledge graph
print("\n    [1/5] Knowledge Graph...")
if PLOTLY_AVAILABLE:
    fig_kg = visualizer.render_knowledge_graph(
        neurl_graph, inference_result,
        layout=neurl_config.graph.layout_algorithm
    )

# 2. Embedding space
print("    [2/5] Embedding Space (UMAP/PCA)...")
if PLOTLY_AVAILABLE and neurl_graph.pyg_data is not None:
    node_type_labels = [
        neurl_graph.graph.nodes[n].get("entity_type", "unknown")
        if neurl_graph.graph.has_node(n) else "unknown"
        for n in inference_result.node_names
    ]
    fig_emb = visualizer.render_embedding_space(
        inference_result.node_embeddings,
        inference_result.node_names,
        node_type_labels,
        neurl_graph.graph,
    )

# 3. Training dashboard
print("    [3/5] Training Dashboard...")
if PLOTLY_AVAILABLE:
    fig_train_dash = visualizer.render_training_dashboard(experiment_tracker)

# 4. Entity-relationship matrix
print("    [4/5] Entity-Relationship Matrix...")
if PLOTLY_AVAILABLE:
    fig_rel_matrix = visualizer.render_entity_relationship_matrix(erp_entities, erp_relationships)

# 5. BI dashboard
print("    [5/5] Business Intelligence Dashboard...")
if PLOTLY_AVAILABLE:
    fig_bi_dash = visualizer.render_business_intelligence_dashboard(bi, erp_entities)

# Step 10: Export
print("\n  Step 10 — Exporting...")
graph_json = exporter.to_json(neurl_graph, inference_result)
export_path = neurl_config.output.export_dir / "neurl_erp_knowledge_graph.json"
with open(export_path, "w") as f:
    json.dump(graph_json, f, indent=2)
print(f"    JSON exported: {export_path} ({export_path.stat().st_size // 1024} KB)")

# GraphML export
graphml_path = neurl_config.output.export_dir / "neurl_erp_graph.graphml"
exporter.to_graphml(neurl_graph, graphml_path)
print(f"    GraphML exported: {graphml_path}")

# Cytoscape.js export
cyto_json = exporter.to_cytoscape_json(neurl_graph, inference_result)
cyto_path = neurl_config.output.export_dir / "neurl_erp_cytoscape.json"
with open(cyto_path, "w") as f:
    json.dump(cyto_json, f, indent=2)
print(f"    Cytoscape.js exported: {cyto_path}")

# Model card
model_card_md = exporter.to_model_card(erp_entities, erp_relationships, bi, experiment_tracker)
card_path = neurl_config.output.export_dir / "neurl_model_card.md"
with open(card_path, "w") as f:
    f.write(model_card_md)
print(f"    Model card exported: {card_path}")

# Print condensed model card
print("\n  ── Model Card (excerpt) ────────────────────────────────────────────")
for line in model_card_md.split("\n")[:40]:
    print("  " + line)

print("\n" + "═" * 70)
print("✅ NEURL Engine pipeline complete — Knowledge Graph ready for downstream analytics")
print("═" * 70)
print(f"\n  Entities    : {[e.entity_name for e in erp_entities]}")
print(f"  Nodes       : {neurl_graph.graph.number_of_nodes()}")
print(f"  Edges       : {neurl_graph.graph.number_of_edges()}")
print(f"  Communities : {len(set(neurl_graph.communities.values()))}")
print(f"  Health      : {bi['graph_health_score']:.4f}")
print(f"\n  Exports:")
print(f"    JSON     → {export_path}")
print(f"    GraphML  → {graphml_path}")
print(f"    Cytoscape→ {cyto_path}")
print(f"    ModelCard→ {card_path}")
print(f"\n  Exports: visualizer, exporter, graph_json, model_card_md")
