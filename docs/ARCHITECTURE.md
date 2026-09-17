# Architecture

NEURL Engine converts a raw structured dataset into an interactive knowledge graph and a business intelligence report, entirely through learned models rather than LLM prompting. This document covers the full pipeline and the reasoning behind its central design choice.

## Design philosophy: learned weights, not prompts

The README frames this directly: where a comparable system would ask an LLM "find the entities in this dataset" and "find the relationships between these columns," NEURL Engine trains a model to answer both questions from structural and statistical patterns in the data itself. The intelligence lives in learned weights, not in a prompt template.

This trade gives up an LLM's ability to use broad world knowledge or handle wildly novel schemas zero-shot, in exchange for:

- **Determinism** — the same dataset produces the same graph, run to run, which an LLM-prompted approach cannot guarantee.
- **Zero marginal inference cost** — once trained, running the model on a new dataset costs a forward pass, not an API call.
- **No vendor lock-in or rate limits** — nothing here depends on Gemini, GPT, or Claude being available or affordable at query time.
- **No hallucination risk on schema/relationship claims** — a trained classifier can be wrong, but it can't invent a column that doesn't exist the way a language model can.
- **Continuously improving representations** — the model can be retrained/fine-tuned as more labeled or corrected graphs become available, which a prompt template can't meaningfully do.

## The core innovation: Node Intelligence

Node Intelligence is the specific capability of learning which columns in a dataset deserve to become nodes in a knowledge graph, and how those nodes relate, purely from structural and statistical patterns (column name similarity, value overlap, correlation, naming conventions) rather than semantic understanding of what the columns "mean" in a human sense. A worked example from the README: `Customer Name + Email + Phone` columns cluster into a `Customer` node; `Invoice Number + Date + Amount` cluster into an `Invoice` node. Neither grouping requires the model to "know" what a customer or invoice is: it only needs to recognize the statistical signature of columns that co-vary and share naming patterns consistent with describing one entity.

## Full pipeline

```
Raw Dataset (CSV / JSON / Parquet / Excel / XML / JSONL)
       │
       ▼
Data Ingestion and Validation      Cell 2 — universal loader, type inference, schema detection
       │
       ▼
Feature Engineering                Cell 3 — per-column statistical/linguistic/structural features,
       │                                     TF-IDF similarity, value overlap, temporal/frequency features
       ▼
Node Intelligence Engine           Cell 4 — hierarchical clustering over column feature vectors →
       │                                     EntitySchema list with confidence scores
       ▼
Relationship Intelligence          Cell 5 — foreign-key patterns, naming conventions, correlation,
       │                                     temporal ordering → RelationshipSchema list with cardinality
       ▼
Knowledge Graph Construction       Cell 6 — NetworkX dual-level graph (schema layer + instance layer),
       │                                     deduplication, Louvain community detection, hierarchy
       ▼
GNN Architecture Suite             Cell 7 — GCN / GraphSAGE / GAT / GIN / VGAE, shared interface
       │
       ▼
Multi-Task Training                Cell 8 — link prediction + reconstruction + KL + InfoNCE losses,
       │                                     AdamW, cosine LR, AMP, early stopping, checkpointing
       ▼
Inference, Explainability, BI      Cell 9 — embeddings, link predictions, anomaly scores, community
       │                                     assignments, BI report, node importance, attention maps
       ▼
Visualization and Export           Cell 10 — interactive Plotly charts, UMAP projection,
       │                                      JSON / GraphML / Cytoscape.js / Markdown export
       ▼
Visualization-Ready Knowledge Graph + Business Intelligence Report
```

## Dual-level graph: schema layer vs. instance layer

Knowledge Graph Construction (Cell 6) builds two layers, not one:

- **Schema layer** — entity nodes (e.g. `Customer`, `Invoice`) and relationship edges between entity *types*. This is the graph of "what kinds of things exist and how kinds relate."
- **Instance layer** — sampled actual data rows as instance nodes. This is the graph of "what specific things exist," at a sampled scale rather than the full dataset, presumably for tractability on large datasets.

Keeping these separate means the GNN suite and the BI report can operate at whichever level of abstraction the task calls for: entity-type relationship discovery at the schema layer, anomaly detection on specific records at the instance layer.

## Graceful dependency degradation

The README states scipy, sklearn, umap, and louvain are all optional, with NumPy fallbacks. This matters for deployment flexibility: a minimal install can still run the core pipeline, with reduced functionality (e.g. no UMAP projection, a simpler community-detection fallback instead of Louvain) rather than failing outright on a missing optional dependency. See `docs/SETUP_AND_USAGE.md` for what degrades and what doesn't.
