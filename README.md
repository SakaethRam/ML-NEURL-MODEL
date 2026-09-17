# NEURL Engine
## AI-Native Node Intelligence Foundation Model

**Version:** 1.0.0  
**Architecture:** Multi-Task Graph Neural Network with Self-Supervised Node Intelligence  
**Compute:** CPU / CUDA (auto-detected)  
**Dependencies:** PyTorch, PyTorch Geometric, NetworkX, scikit-learn, Plotly  
**License:** Proprietary

<img width="1800" height="1000" alt="ZREX NEURL" src="https://github.com/user-attachments/assets/580503a1-1985-4e40-b08a-c708f42ee1e2" />

---

## Overview

NEURL Engine is a production-grade machine learning system that automatically converts structured datasets into clean, meaningful, interactive knowledge graphs using learned machine intelligence rather than large language model prompting.

The system eliminates every dependency on external AI services (Gemini, GPT, Claude) by replacing prompt-based graph generation with trained neural representations. Every capability the existing GraphAPI, TableAPI, and document understanding API delegate to Gemini is instead performed by NEURL Engine's learned models.

The core innovation is Node Intelligence: the ability to learn which columns in a dataset deserve to become nodes in a knowledge graph, and how those nodes relate to each other, purely from structural and statistical patterns in the data.

---

## Design Philosophy

Traditional graph generation systems ask a language model:

    "Find the entities in this dataset."
    "Find the relationships between these columns."

NEURL Engine instead trains a model to recognize the answer directly from data patterns. The intelligence lives in learned weights, not in prompts.

This produces:

- Deterministic, reproducible outputs
- Zero inference cost per query after training
- No vendor lock-in or API rate limits
- No hallucinations
- Scalable batch processing
- Continuously improving representations

---

## Architecture

```
Raw Dataset (CSV / JSON / Parquet / Excel / XML / JSONL)
       |
       v
+------------------+
| Data Ingestion   |  Cell 2 -- Universal loader, type inference, schema detection
| and Validation   |  Formats: CSV, JSON, Parquet, Excel, JSONL, XML, SQL exports
+------------------+
       |
       v
+------------------+
| Feature          |  Cell 3 -- Per-column statistical, linguistic, structural features
| Engineering      |  Outputs: column metadata vectors, TF-IDF similarity matrix,
|                  |           value overlap matrix, temporal and frequency features
+------------------+
       |
       v
+------------------+
| Node Intelligence|  Cell 4 -- CORE INNOVATION
| Engine           |  Hierarchical clustering over column feature vectors
|                  |  Groups: Customer Name + Email + Phone -> Customer Node
|                  |          Invoice Number + Date + Amount -> Invoice Node
|                  |  Output: EntitySchema list with confidence scores
+------------------+
       |
       v
+------------------+
| Relationship     |  Cell 5 -- Directed relationship discovery
| Intelligence     |  Evidence: foreign key patterns, naming conventions,
|                  |            numeric correlation, temporal ordering
|                  |  Output: RelationshipSchema list with cardinality labels
+------------------+
       |
       v
+------------------+
| Knowledge Graph  |  Cell 6 -- NetworkX dual-level graph construction
| Construction     |  Schema layer: entity nodes + relationship edges
|                  |  Instance layer: sampled data rows as instance nodes
|                  |  Operations: deduplication, community detection (Louvain),
|                  |              hierarchy generation, topology validation
+------------------+
       |
       v
+------------------+
| GNN Architecture |  Cell 7 -- Five interchangeable architectures
| Suite            |  GCN: spectral convolution, fast, homogeneous graphs
|                  |  GraphSAGE: inductive, scales to unseen nodes
|                  |  GAT: attention-weighted message passing
|                  |  GIN: maximally expressive (WL-equivalent)
|                  |  VGAE: variational autoencoder for link prediction
+------------------+
       |
       v
+------------------+
| Multi-Task       |  Cell 8 -- End-to-end training framework
| Training         |  Loss: link prediction + reconstruction + KL + InfoNCE
|                  |  Features: AdamW, cosine LR schedule, gradient clipping,
|                  |            mixed precision (AMP), early stopping,
|                  |            checkpointing, experiment tracking
+------------------+
       |
       v
+------------------+
| Inference and    |  Cell 9 -- Full inference pipeline
| Explainability   |  Outputs: node embeddings, link predictions, anomaly scores,
|                  |           community assignments, business intelligence report,
|                  |           node importance rankings, attention maps
+------------------+
       |
       v
+------------------+
| Visualization    |  Cell 10 -- Interactive Plotly charts
| and Export       |  Charts: knowledge graph, UMAP embedding space,
|                  |          training metrics, entity heatmap, BI dashboard
|                  |  Exports: JSON, GraphML, Cytoscape.js, Markdown report
+------------------+
       |
       v
Visualization-Ready Knowledge Graph + Business Intelligence Report
```

---

## Cell Reference

| Cell | Name | Responsibility |
|------|------|----------------|
| 1 | Configuration and System Bootstrap | Central config dataclasses, logging, device management, enums, experiment tracker, model registry |
| 2 | Universal Data Ingestion and Validation | Multi-format loader, type inference, data quality checks, schema metadata extraction |
| 3 | Feature Engineering | Per-column feature vectors, TF-IDF name embeddings, similarity matrix, value overlap, entropy computation |
| 4 | Node Intelligence Engine | Semantic column clustering, entity schema discovery, confidence scoring, entity type labeling |
| 5 | Relationship Intelligence | Foreign key detection, naming convention analysis, correlation scoring, directed relationship schema |
| 6 | Knowledge Graph Construction | NetworkX graph assembly, community detection, hierarchy computation, PyG data conversion |
| 7 | GNN Architecture Suite | GCN, GraphSAGE, GAT, GIN, VGAE implementations with shared interface and model factory |
| 8 | Multi-Task Training Pipeline | Mini-batch GNN training, multi-component loss, checkpointing, early stopping, metrics tracking |
| 9 | Inference, Explainability and Business Intelligence | Embedding extraction, link prediction, anomaly detection, BI report generation |
| 10 | Visualization and Graph Export | Interactive Plotly charts, UMAP projection, multi-format graph export |

---

## Data Processing Pipeline

```
Dataset Upload
    |
    v
Data Validation        -- schema checks, format detection, null audit
    |
    v
Cleaning               -- drop constant columns, handle missing values, deduplicate
    |
    v
Type Inference         -- numeric, categorical, datetime, text, identifier detection
    |
    v
Feature Engineering    -- statistical, linguistic, structural features per column
    |
    v
Semantic Grouping      -- TF-IDF column name similarity + hierarchical clustering
    |
    v
Entity Discovery       -- column groups become entity schemas with confidence scores
    |
    v
Relationship Learning  -- FK patterns, correlation, naming conventions -> edges
    |
    v
Graph Construction     -- NetworkX schema + instance graph, community detection
    |
    v
GNN Encoding           -- message passing over graph, node embeddings
    |
    v
Multi-Task Training    -- link prediction, reconstruction, contrastive learning
    |
    v
Inference              -- embeddings, predictions, anomaly scores, BI report
    |
    v
Visualization          -- interactive Plotly graph + UMAP + export
```

---

## Supported Input Formats

- CSV (comma-separated values)
- Excel (.xlsx, .xls via openpyxl)
- JSON (records, list, or dict orientations)
- JSONL (newline-delimited JSON)
- Parquet (Apache Arrow columnar)
- XML (auto-parsed to tabular form)
- SQL exports (treated as CSV)
- ERP exports, CRM exports, business logs

---

## Machine Learning Architectures

### Graph Convolutional Network (GCN)
Spectral-domain convolution over the normalized graph Laplacian. Fast training,
strong performance on homogeneous graphs with consistent node degree distributions.

### GraphSAGE
Inductive representation learning via neighborhood sampling and aggregation.
Generalizes to nodes unseen during training, making it suitable for streaming
datasets where new entities arrive continuously.

### Graph Attention Network (GAT)
Multi-head attention mechanism assigns learned importance weights to each
neighbor during message passing. Preferred when relationship strength is
heterogeneous across the graph.

### Graph Isomorphism Network (GIN)
Provably as expressive as the Weisfeiler-Lehman graph isomorphism test.
Theoretically the most powerful architecture in this suite for distinguishing
structurally distinct subgraphs.

### Variational Graph Autoencoder (VGAE)
Generative model that learns a latent Gaussian distribution over node
representations. Optimal for link prediction tasks including knowledge graph
completion and missing relationship discovery.

---

## Multi-Task Learning Objectives

The training pipeline simultaneously optimizes:

- **Entity Discovery**: learn which columns form coherent semantic groups
- **Relationship Discovery**: learn which entity pairs share meaningful connections
- **Node Classification**: classify nodes into business entity type categories
- **Link Prediction**: predict missing edges in the knowledge graph
- **Graph Reconstruction**: reconstruct adjacency from embeddings (autoencoder objective)
- **Anomaly Detection**: score nodes by deviation from learned neighborhood distributions
- **Community Detection**: unsupervised clustering of semantically related nodes

---

## Loss Function Components

| Loss | Symbol | Purpose |
|------|--------|---------|
| Link Prediction | L_lp | Binary cross-entropy over positive and negative edges |
| Graph Reconstruction | L_recon | MSE between predicted and actual adjacency |
| KL Divergence | L_kl | Variational regularization for VGAE latent space |
| InfoNCE Contrastive | L_nce | Self-supervised alignment of semantically similar nodes |
| Total | L_total | w1 * L_lp + w2 * L_recon + w3 * L_kl + w4 * L_nce |

---

## Evaluation Metrics

**Link Prediction:** ROC-AUC, Average Precision, Hits@K, MRR  
**Node Classification:** Accuracy, F1 (macro), Precision, Recall  
**Clustering Quality:** Silhouette Score, Modularity  
**Graph Quality:** Graph Reconstruction Accuracy, Connected Component Ratio  
**System:** Training Throughput (samples/sec), Inference Latency (ms), Memory (MB)

---

## Business Intelligence Outputs

After inference the engine produces a structured business intelligence report containing:

- Entity inventory with confidence scores and semantic type labels
- Directed relationship map with cardinality (1:1, 1:N, N:N)
- Central node ranking (most influential entities by graph centrality)
- Anomaly scores for outlier detection (fraud, data quality issues)
- Community structure (natural business process groupings)
- Link prediction candidates (missing relationships the model expects to exist)
- Embedding quality metrics (silhouette, explained variance)

---

## Engineering Standards

- Object-oriented design with abstract base classes and factory pattern
- Full type hints on all public interfaces
- Google-style docstrings on all classes and methods
- Configuration-driven design via NEURLConfig dataclass hierarchy
- Structured exception hierarchy (NEURLBaseError subclasses)
- Structured logging with per-module loggers and file handlers
- Experiment tracking with JSON artifact storage
- Model registry with checkpoint management and version tagging
- Graceful dependency degradation: scipy, sklearn, umap, louvain all optional with numpy fallbacks

---

## Quick Start

```python
# Bootstrap the engine
neurl_config = NEURLConfig()
logger, device, tracker, registry = bootstrap_neurl_engine(neurl_config)

# Ingest a dataset
ingestion_engine = DataIngestionEngine(neurl_config.data_ingestion)
dataset = ingestion_engine.ingest_from_synthetic(n_rows=500)

# Discover entities
feature_engine = ColumnFeatureEngineer(neurl_config.feature_engineering)
feature_matrix = feature_engine.build_feature_matrix(dataset.dataframe)

node_engine = NodeIntelligenceEngine(neurl_config.feature_engineering)
entity_schemas = node_engine.discover_entities(dataset.dataframe)

# Discover relationships
rel_engine = RelationshipIntelligenceEngine(neurl_config.graph_construction)
relationship_schemas = rel_engine.discover_relationships(dataset.dataframe, entity_schemas)

# Build and train
graph_engine = KnowledgeGraphEngine(neurl_config.graph_construction)
neurl_graph = graph_engine.build_graph(dataset.dataframe, entity_schemas, relationship_schemas)

trainer = NEURLTrainer(neurl_config.training, neurl_config.loss_weights)
trained_model, metrics = trainer.train(neurl_graph.pyg_data)

# Run inference
inference_engine = NEURLInferenceEngine(neurl_config.inference)
results = inference_engine.run(trained_model, neurl_graph)

# Visualize
viz_engine = NEURLVisualizationEngine(neurl_config.output)
viz_engine.render_all(neurl_graph, results)
```

---

## Repository Structure (Notebook Cells)

```
NEURL Engine Notebook
|
|-- [MARKDOWN]  README                          (this block)
|-- [PYTHON]    Cell 1: Configuration           NEURLConfig, enums, exceptions, bootstrap
|-- [PYTHON]    Cell 2: Data Ingestion          DataIngestionEngine, DatasetMetadata
|-- [PYTHON]    Cell 3: Feature Engineering     ColumnFeatureEngineer, feature vectors
|-- [PYTHON]    Cell 4: Node Intelligence       NodeIntelligenceEngine, EntitySchema
|-- [PYTHON]    Cell 5: Relationship Intel      RelationshipIntelligenceEngine, RelationshipSchema
|-- [PYTHON]    Cell 6: Graph Construction      KnowledgeGraphEngine, NEURLGraph
|-- [PYTHON]    Cell 7: GNN Architecture Suite  BaseGNN, GCNEncoder, SAGEEncoder, GATEncoder, GINEncoder, VGAEEncoder
|-- [PYTHON]    Cell 8: Training Pipeline       NEURLTrainer, TrainingResult, loss components
|-- [PYTHON]    Cell 9: Inference               NEURLInferenceEngine, InferenceResult, BI report
|-- [PYTHON]    Cell 10: Visualization          NEURLVisualizationEngine, graph export
```

---

*NEURL Engine is a proprietary AI-native system. All graph intelligence is derived from learned representations, not language model prompting.*

---

# License

ML-NEURL is distributed under the terms defined in `LICENSE`.
