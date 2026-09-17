# Setup and Usage

## Important: same notebook-cell design as ML-N2FC

Like this author's other ML repositories, NEURL Engine is organized as 10 numbered "Cells," and several filenames contain characters that are not valid in a Python `import` statement: `&` (`Config&SysBootstrap.py`, `DataIngestion&Validate.py`, `Visual&Graph-Export.py`) and `-` (`Feature-Engineer.py`, `GNN-Architect-Suite.py`, `IEB-IntelliEngine.py`, `KnowledgeGraph-Construct.py`, `MultiTask-Train-Pipeline.py`, `NodeIntelli-Engine.py`, `RelationIntelli-Engine.py`).

The README's own Quick Start snippet confirms this design directly: it calls `NEURLConfig()`, `bootstrap_neurl_engine(...)`, `DataIngestionEngine(...)`, and so on as if they're already defined in scope, with no `import` statement anywhere in the snippet. That's only possible if all 10 cells have already executed into one shared namespace, exactly like running every cell in a notebook top to bottom.

## Running via the provided runner script

```bash
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` executes the 10 cell files in the README's documented order, in one shared namespace, using `exec()`. It does not rename or modify anything in the original repository.

## Running the Quick Start example end to end

This delivery also includes `example_end_to_end.py`, which runs `run_pipeline.py`'s loader and then executes the README's own Quick Start flow against synthetic data, so you can confirm the whole system works before pointing it at a real dataset:

```bash
python example_end_to_end.py
```

This mirrors the README's Quick Start exactly:

```python
neurl_config = NEURLConfig()
logger, device, tracker, registry = bootstrap_neurl_engine(neurl_config)

ingestion_engine = DataIngestionEngine(neurl_config.data_ingestion)
dataset = ingestion_engine.ingest_from_synthetic(n_rows=500)

feature_engine = ColumnFeatureEngineer(neurl_config.feature_engineering)
feature_matrix = feature_engine.build_feature_matrix(dataset.dataframe)

node_engine = NodeIntelligenceEngine(neurl_config.feature_engineering)
entity_schemas = node_engine.discover_entities(dataset.dataframe)

rel_engine = RelationshipIntelligenceEngine(neurl_config.graph_construction)
relationship_schemas = rel_engine.discover_relationships(dataset.dataframe, entity_schemas)

graph_engine = KnowledgeGraphEngine(neurl_config.graph_construction)
neurl_graph = graph_engine.build_graph(dataset.dataframe, entity_schemas, relationship_schemas)

trainer = NEURLTrainer(neurl_config.training, neurl_config.loss_weights)
trained_model, metrics = trainer.train(neurl_graph.pyg_data)

inference_engine = NEURLInferenceEngine(neurl_config.inference)
results = inference_engine.run(trained_model, neurl_graph)

viz_engine = NEURLVisualizationEngine(neurl_config.output)
viz_engine.render_all(neurl_graph, results)
```

Using `ingest_from_synthetic(n_rows=500)` means this runs without needing a real dataset on hand, which makes it a genuine smoke test rather than something you need production data to exercise.

## Running against your own dataset

Once the Quick Start example runs cleanly, swap `ingestion_engine.ingest_from_synthetic(n_rows=500)` for the equivalent real-data loading call (confirm the exact method name and signature in `DataIngestion&Validate.py`; the README documents the synthetic path explicitly but doesn't show the real-file-loading call signature).

## Installing PyTorch Geometric

PyTorch Geometric's binary extensions (`torch-scatter`, `torch-sparse`, etc., where used by the installed PyG version) must match your exact PyTorch version and CUDA version (or CPU-only build). Installing `torch` and `torch_geometric` from generic PyPI without matching them can produce a install that imports fine but fails at first GNN forward pass. See `requirements.txt` and `Dockerfile` / `Dockerfile.gpu` in this delivery for a version-matched installation; if you're installing manually outside Docker, use the PyG installation matrix at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html to pick wheels matching your exact torch/CUDA combination rather than a bare `pip install torch_geometric`.

## Dependencies

Per the README: PyTorch, PyTorch Geometric, NetworkX, scikit-learn, Plotly, required; scipy, sklearn (beyond what's required), umap-learn, and python-louvain are optional with NumPy fallbacks per the "graceful dependency degradation" design (see `ARCHITECTURE.md`). `requirements.txt` in this delivery installs the full, non-degraded dependency set — see the file for exact versions.

## Full setup checklist

- [ ] Python 3.9+ available
- [ ] `pip install -r requirements.txt`, or the Docker path below
- [ ] `python run_pipeline.py` completes without a missing-block error
- [ ] `python example_end_to_end.py` completes and writes visualization output, confirming the full pipeline (including GNN training) works before real data is involved
- [ ] Swap in real-dataset ingestion once the synthetic run is confirmed working
