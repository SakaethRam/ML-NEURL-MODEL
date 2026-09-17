# Contributing

## Before you start

- Same shared-namespace caveat as `docs/SETUP_AND_USAGE.md` describes: a change to an earlier cell can silently break a later one at runtime rather than at import time. Always test with `python example_end_to_end.py`, not just the cell you directly edited.
- Changes to loss weights (`w1`–`w4` in `MultiTask-Train-Pipeline.py`) or to which architecture is used should include before/after values for the evaluation metrics in `docs/GNN_ARCHITECTURES_AND_TRAINING.md`'s table, since a change here affects seven training objectives simultaneously.
- Changes to Node Intelligence or Relationship Intelligence (Cells 4-5) should include a worked example (a small synthetic or anonymized dataset) showing what entities/relationships were discovered before and after, since these are the hardest parts of the pipeline to reason about from a code diff alone.

## Workflow

1. Fork the repository.
2. Create a feature branch.
3. Implement your change, scoped to as few cells as possible.
4. Run `python example_end_to_end.py` and confirm the full pipeline (ingestion through visualization) still completes.
5. Submit a pull request describing which cell(s) changed and any metric shifts observed.

## Where to make changes

| Area | Cell / File |
|------|-------------|
| Config, bootstrap, logging | Cell 1, `Config&SysBootstrap.py` |
| Data ingestion, validation | Cell 2, `DataIngestion&Validate.py` |
| Feature engineering | Cell 3, `Feature-Engineer.py` |
| Node Intelligence | Cell 4, `NodeIntelli-Engine.py` |
| Relationship Intelligence | Cell 5, `RelationIntelli-Engine.py` |
| Knowledge graph construction | Cell 6, `KnowledgeGraph-Construct.py` |
| GNN architectures | Cell 7, `GNN-Architect-Suite.py` |
| Training pipeline | Cell 8, `MultiTask-Train-Pipeline.py` |
| Inference, explainability, BI | Cell 9, `IEB-IntelliEngine.py` |
| Visualization, export | Cell 10, `Visual&Graph-Export.py` |
| Runner / example scripts | `run_pipeline.py`, `example_end_to_end.py` (this delivery) |
| CI | `.github/workflows/` |

## A note on the "Proprietary" license

The README states a Proprietary license. Confirm the actual terms with the repository owner before treating this as open to external contributions at all; this contributing guide assumes contributions are welcome in the same spirit as the author's other repositories, but that assumption should be verified against the real license terms rather than taken for granted here.

## Reporting issues

Use the repository's Issues tab for bugs, training instability, or documentation gaps. Given the multi-task loss design, a training-stability report is most useful if it includes which architecture was in use and the loss curve shape (all four components, not just the total), since a stability issue in one component (e.g. KL divergence exploding) can look like a generic "training diverged" report without that detail.
