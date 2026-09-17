# Inference, Business Intelligence, and Export

## Inference and explainability (Cell 9)

Given a trained model and the constructed knowledge graph, this stage produces:

- **Node embeddings** — the learned vector representation for every node, usable for downstream similarity search or clustering outside NEURL Engine itself.
- **Link predictions** — candidate edges the model believes should exist but weren't observed in the source data.
- **Anomaly scores** — per-node deviation from learned neighborhood distributions, i.e. "how unusual is this node relative to nodes like it."
- **Community assignments** — which cluster/community each node belongs to, from the graph-construction stage's Louvain detection (or its NumPy fallback).
- **Node importance rankings** — centrality-based ranking of which entities matter most in the graph's structure.
- **Attention maps** — for attention-based architectures (GAT specifically), which neighbors the model weighted most heavily for a given node's representation.

## Business Intelligence report

The structured BI output assembled from the above:

- Entity inventory, with confidence scores and semantic type labels
- Directed relationship map with cardinality (1:1, 1:N, N:N)
- Central node ranking (most influential entities by graph centrality)
- Anomaly scores for outlier detection — explicitly framed for fraud detection and data-quality issue discovery
- Community structure — read as natural business-process groupings, not just abstract clusters
- Link prediction candidates — missing relationships the model expects to exist
- Embedding quality metrics (silhouette score, explained variance)

### Reading anomaly scores correctly

An anomaly score is relative to a node's *learned neighborhood distribution*, not an absolute measure of "wrongness." A node can score as anomalous either because it represents a genuinely unusual real-world case (a fraud candidate, a data-entry error) or because the model's neighborhood representation for that region of the graph is under-trained (e.g. a sparse or under-represented entity type). Before treating a high anomaly score as an actionable fraud/quality flag, check whether the flagged node's entity type has enough training examples to have a well-formed neighborhood distribution in the first place; an anomaly score on a rare entity type is weaker evidence than the same score on a well-represented one.

## Visualization and export (Cell 10)

**Interactive Plotly charts:**

- The knowledge graph itself (schema and/or instance layer)
- UMAP embedding-space projection (subject to UMAP being installed; see the graceful-degradation note in `ARCHITECTURE.md`)
- Training metrics
- Entity heatmap
- The full BI dashboard

**Export formats:**

- JSON
- GraphML — for opening in dedicated graph-analysis tools (Gephi, Cytoscape desktop)
- Cytoscape.js — for embedding an interactive graph directly in a web page
- Markdown report — the BI report rendered as a readable document, suitable for sharing with a non-technical stakeholder without requiring them to open the interactive visualization

## Which export format for which audience

- **GraphML** — a data analyst who wants to explore the graph further in Gephi or a similar desktop tool.
- **Cytoscape.js** — embedding the graph in an internal dashboard or web app.
- **JSON** — programmatic downstream consumption (feeding the graph or BI results into another system).
- **Markdown report** — a business stakeholder who wants the findings, not the graph structure itself.
