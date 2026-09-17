# GNN Architectures and Training

## The five architectures, and when to reach for each

| Architecture | Core mechanism | Best suited for |
|---------------|-----------------|-------------------|
| **GCN** | Spectral convolution over the normalized graph Laplacian | Fast training, homogeneous graphs with consistent node degree distributions |
| **GraphSAGE** | Inductive learning via neighborhood sampling/aggregation | Streaming datasets where new entities arrive continuously; generalizes to unseen nodes |
| **GAT** | Multi-head attention, learned per-neighbor importance | Graphs with heterogeneous relationship strength, where not all edges should count equally |
| **GIN** | Provably as expressive as the Weisfeiler-Lehman test | Distinguishing structurally distinct subgraphs; the most theoretically powerful option in the suite |
| **VGAE** | Variational autoencoder over a latent Gaussian node distribution | Link prediction specifically: knowledge graph completion, missing relationship discovery |

All five share a common interface (per the README's "shared interface and model factory" design), so switching architectures is a configuration change, not a rewrite of the training or inference code around it.

### Picking one in practice

- Start with **GCN** for a first pass on a new dataset: it's the fastest to train and gives a reasonable baseline before investing in a more expensive architecture.
- Move to **GraphSAGE** if the target use case involves a dataset that grows over time (new customers, new invoices) and the model needs to produce embeddings for nodes it never saw during training.
- Move to **GAT** if some relationships in the data are clearly more load-bearing than others (e.g. a primary foreign key vs. a loose naming-convention match) and you want the model to learn that distinction rather than treating every edge uniformly.
- Reach for **GIN** specifically when structural distinctions matter more than feature values, e.g. telling apart two entities that look similar in isolation but sit in structurally different positions in the graph.
- Use **VGAE** when the actual task is "what relationships are we missing," not just "represent the graph well." Its latent-Gaussian design is purpose-built for link prediction, not a general-purpose encoder repurposed for it.

## Multi-task training

The training pipeline (Cell 8) optimizes seven objectives simultaneously, not one:

- Entity Discovery
- Relationship Discovery
- Node Classification
- Link Prediction
- Graph Reconstruction
- Anomaly Detection
- Community Detection

### Loss composition

| Loss | Symbol | Purpose |
|------|--------|---------|
| Link Prediction | L_lp | Binary cross-entropy over positive/negative edges |
| Graph Reconstruction | L_recon | MSE between predicted and actual adjacency |
| KL Divergence | L_kl | Variational regularization for the VGAE latent space |
| InfoNCE Contrastive | L_nce | Self-supervised alignment of semantically similar nodes |
| **Total** | L_total | `w1*L_lp + w2*L_recon + w3*L_kl + w4*L_nce` |

The four weights (`w1`–`w4`) are configuration, not fixed constants; per the "Engineering Standards" section of the README, they're driven through the `NEURLConfig` dataclass hierarchy rather than hardcoded in the training loop. Confirm the actual default weights in `MultiTask-Train-Pipeline.py` before assuming a specific balance between the four objectives.

### Training infrastructure

- AdamW optimizer
- Cosine learning-rate schedule
- Gradient clipping
- Mixed precision (AMP) — meaningful for training throughput specifically on the GPU path
- Early stopping and checkpointing
- Experiment tracking (JSON artifact storage, per "Engineering Standards")

## Evaluation metrics, by task

| Task | Metrics |
|------|---------|
| Link Prediction | ROC-AUC, Average Precision, Hits@K, MRR |
| Node Classification | Accuracy, macro F1, Precision, Recall |
| Clustering Quality | Silhouette Score, Modularity |
| Graph Quality | Graph Reconstruction Accuracy, Connected Component Ratio |
| System | Training throughput (samples/sec), inference latency (ms), memory (MB) |

Note the system-level metrics (throughput, latency, memory) sit alongside the model-quality metrics as first-class outputs, not an afterthought. This is consistent with the README's "production-grade" framing: a model that scores well on ROC-AUC but is too slow or memory-hungry for the target deployment isn't actually production-ready by this project's own stated bar.

## Choosing loss weights and architecture together

VGAE's loss profile (dominated by L_kl and L_recon) is a different optimization landscape than GAT's (dominated by L_lp and L_nce, with no KL term). If you swap architectures, revisit the loss weights (`w1`–`w4`) rather than assuming the same weighting that worked for one architecture transfers cleanly to another.
