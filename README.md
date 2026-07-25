# ML-NEURL

**Node Enhanced Universal Reasoning Layer**

ML-NEURL is an AI-native machine learning framework for transforming structured enterprise datasets into clean, semantically meaningful knowledge graphs without relying on Large Language Models or prompt engineering.

Instead of using external AI services to infer entities and relationships, ML-NEURL learns graph structures directly from data through representation learning, graph neural networks, and self-supervised graph intelligence. The framework is designed as a reusable foundation model for structured business intelligence, enabling scalable graph construction, node understanding, relationship prediction, and knowledge discovery.

## Vision

Structured datasets already contain hidden knowledge.

ML-NEURL learns to discover that knowledge by identifying entities, inferring relationships, constructing optimized knowledge graphs, and learning semantic representations that improve over time.

The project aims to become a foundation model for enterprise graph intelligence, replacing rule-based systems and LLM-assisted graph generation with a fully trainable machine learning architecture.

## Core Objectives

- Automatically discover business entities from structured datasets
- Learn semantic relationships between entities
- Construct clean and optimized knowledge graphs
- Generate high-quality node and graph embeddings
- Predict missing relationships and incomplete workflows
- Detect anomalies and disconnected business processes
- Learn reusable graph representations across multiple domains
- Continuously improve as additional datasets are processed

## Design Philosophy

ML-NEURL is built around a single principle:

> Every structured dataset is a graph waiting to be discovered.

Rather than treating datasets as independent tables, ML-NEURL interprets them as interconnected systems of entities, relationships, and workflows.

The framework focuses on learning graph structure directly from data instead of relying on manually engineered rules or prompt-based reasoning.

## Architecture

```
                 Raw Dataset
                      │
                      ▼
              Data Validation
                      │
                      ▼
              Feature Engineering
                      │
                      ▼
              Schema Understanding
                      │
                      ▼
              Node Intelligence
                      │
                      ▼
         Relationship Intelligence
                      │
                      ▼
          Knowledge Graph Builder
                      │
                      ▼
         Graph Representation Learning
                      │
                      ▼
          Graph Neural Networks
                      │
                      ▼
           Embedding Generation
                      │
                      ▼
          Prediction and Inference
                      │
                      ▼
       Interactive Knowledge Graph
```

## Supported Data Sources

ML-NEURL is designed to learn from a wide range of structured and semi-structured datasets, including:

- CSV
- Excel
- JSON
- JSONL
- Parquet
- SQL exports
- ERP datasets
- CRM datasets
- Business logs
- Knowledge graphs
- Relational databases

## Machine Learning Pipeline

The framework implements a complete graph learning workflow.

1. Dataset ingestion
2. Data preprocessing
3. Schema detection
4. Semantic feature extraction
5. Entity discovery
6. Relationship learning
7. Knowledge graph construction
8. Graph optimization
9. Representation learning
10. Model training
11. Graph inference
12. Business intelligence

## Graph Intelligence

ML-NEURL focuses on understanding graph semantics rather than simply converting columns into nodes.

The framework learns to:

- Discover meaningful entities
- Merge semantically related attributes
- Infer hidden relationships
- Remove graph noise
- Simplify graph topology
- Generate hierarchical graph structures
- Rank important nodes
- Detect graph communities
- Predict missing graph connections

## Learning Tasks

ML-NEURL supports multi-task graph learning, including:

- Entity discovery
- Relationship prediction
- Node classification
- Edge classification
- Link prediction
- Graph classification
- Community detection
- Knowledge graph completion
- Business workflow prediction
- Anomaly detection

## Neural Architectures

The framework supports interchangeable graph learning models, including:

- Graph Convolutional Networks (GCN)
- GraphSAGE
- Graph Attention Networks (GAT)
- Graph Isomorphism Networks (GIN)
- Relational Graph Convolutional Networks (R-GCN)
- Graph Transformers
- Graph Autoencoders
- Variational Graph Autoencoders
- Node2Vec
- DeepWalk
- Contrastive Graph Learning
- Self-Supervised Graph Learning

## Node Intelligence

Node Intelligence is the defining capability of ML-NEURL.

Instead of representing every dataset column as an individual node, the framework learns semantic abstractions.

For example:

```
Customer Name
Customer Email
Customer Phone
Customer Address
```

becomes

```
Customer
```

Similarly,

```
Invoice Number
Invoice Date
Invoice Amount
```

becomes

```
Invoice
```

This produces significantly cleaner and more meaningful knowledge graphs.

## Relationship Intelligence

After learning nodes, ML-NEURL predicts semantic relationships such as:

```
Customer
    │
    ▼
Order
    │
    ▼
Payment
    │
    ▼
Invoice
    │
    ▼
Journal Entry
```

Relationships are learned from structural and statistical patterns instead of predefined business rules.

## Explainability

ML-NEURL includes explainable AI techniques for graph learning.

Capabilities include:

- Node importance
- Edge importance
- Attention visualization
- Feature importance
- Confidence estimation
- Subgraph explanations
- Business reasoning reports

## Production Features

- Modular architecture
- Configuration-driven training
- Mixed precision support
- Distributed training
- Automatic checkpointing
- Experiment tracking
- Model versioning
- Incremental learning
- Continual learning
- ONNX export
- TorchScript export
- FastAPI deployment
- Enterprise logging

## Repository Structure

```
ML-NEURL/

├── notebooks/
├── datasets/
├── models/
├── training/
├── inference/
├── graph/
├── embeddings/
├── explainability/
├── visualization/
├── api/
├── configs/
├── checkpoints/
├── exports/
├── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Technology Stack

- Python
- PyTorch
- PyTorch Geometric
- DGL
- NetworkX
- NumPy
- Pandas
- Scikit-learn
- Optuna
- MLflow
- FastAPI
- Plotly
- ONNX

## Roadmap

### Phase 1

- Data ingestion
- Graph construction
- Entity learning
- Relationship learning

### Phase 2

- Graph neural networks
- Representation learning
- Link prediction
- Node embeddings

### Phase 3

- Continual learning
- Self-supervised graph learning
- Graph transformers
- Large-scale graph optimization

### Phase 4

- Enterprise deployment
- Distributed inference
- Multi-domain foundation model
- Production graph intelligence platform

## Long-Term Vision

ML-NEURL is designed to become a reusable Graph Foundation Model for enterprise intelligence.

Its long-term objective is to autonomously understand structured datasets, discover semantic entities, construct optimized knowledge graphs, and learn reusable representations that continuously improve as additional data becomes available.

Rather than acting as a conversational AI system, ML-NEURL serves as the intelligence layer for graph-native analytics, enabling scalable knowledge discovery, business process understanding, and explainable graph reasoning across enterprise domains.

## License

This project is released under the MIT License.
