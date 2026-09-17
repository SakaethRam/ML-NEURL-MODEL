"""
run_pipeline.py

Runs NEURL Engine's 10 cell files in the order documented in the
README, in one shared namespace. See docs/SETUP_AND_USAGE.md for why:
several filenames contain "&" or "-" and are not valid Python module
names, so they cannot be `import`-ed directly. The README's own Quick
Start snippet uses classes (NEURLConfig, DataIngestionEngine, ...)
with no import statements, which only works if every cell has already
executed into a shared namespace, exactly like a notebook run top to
bottom.

Usage:
    python run_pipeline.py
    python -c "from run_pipeline import run_pipeline; ns = run_pipeline(); print(ns['NEURLConfig'])"
"""

from __future__ import annotations

import pathlib
import sys

# Order matches the README's "Repository Structure (Notebook Cells)" section.
CELLS = [
    "Config&SysBootstrap.py",
    "DataIngestion&Validate.py",
    "Feature-Engineer.py",
    "NodeIntelli-Engine.py",
    "RelationIntelli-Engine.py",
    "KnowledgeGraph-Construct.py",
    "GNN-Architect-Suite.py",
    "MultiTask-Train-Pipeline.py",
    "IEB-IntelliEngine.py",
    "Visual&Graph-Export.py",
]


def run_pipeline(repo_root: str | pathlib.Path = ".") -> dict:
    """
    Execute all 10 cells in order within one shared namespace and
    return that namespace, so callers (or example_end_to_end.py) can
    use the classes/functions each cell defines.
    """
    root = pathlib.Path(repo_root)
    namespace: dict = {"__name__": "__main__"}

    for cell_name in CELLS:
        cell_path = root / cell_name
        if not cell_path.exists():
            print(f"!! Missing cell file: {cell_path}", file=sys.stderr)
            print(
                "   Place all 10 cell files at the repository root "
                "(or pass the correct --root) before running.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"--- Running {cell_name} ---")
        source = cell_path.read_text(encoding="utf-8")
        code = compile(source, filename=str(cell_path), mode="exec")
        exec(code, namespace)  # noqa: S102 - intentional, see module docstring

    return namespace


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="Directory containing the 10 cell .py files (default: current directory).",
    )
    args = parser.parse_args()

    run_pipeline(args.root)
    print("--- All cells loaded ---")
