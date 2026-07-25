"""
NEURL Engine — Cell 1: Configuration & System Bootstrap

This module establishes the entire configuration space, logging infrastructure,
device management, experiment tracking, and model registry for the NEURL Engine.
All subsequent cells import from the session scope created here.

Design rationale: Centralizing config in a single dataclass eliminates magic numbers
throughout the codebase and makes experiments reproducible and auditable.
"""

# =============================================================================
# UNIVERSAL IMPORTS — all dependencies needed across the entire NEURL Engine
# =============================================================================
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Type, Iterator
)

import numpy as np
import pandas as pd

# ─── Optional ML dependencies with graceful fallbacks ─────────────────────────
try:
    import scipy.sparse as sp_sparse
    import scipy.stats as sp_stats
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import cdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available — numpy fallbacks will be used")

try:
    from sklearn.preprocessing import (
        StandardScaler, LabelEncoder, MinMaxScaler, normalize
    )
    from sklearn.metrics import (
        roc_auc_score, silhouette_score, average_precision_score
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available — numpy fallbacks will be used")

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Mixed precision — use torch.amp instead of torch.cuda.amp for compatibility
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False
        warnings.warn("torch.amp not available — mixed precision disabled")

try:
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.loader import NeighborLoader, DataLoader as PyGDataLoader
    try:
        from torch_geometric.nn import (
            GCNConv, SAGEConv, GATv2Conv, GINConv, VGAE,
            BatchNorm, global_mean_pool, global_max_pool
        )
    except ImportError:
        # GATv2Conv may not be available — fall back to GATConv
        from torch_geometric.nn import (
            GCNConv, SAGEConv, GATConv as GATv2Conv, GINConv, VGAE,
            BatchNorm, global_mean_pool, global_max_pool
        )
    from torch_geometric.transforms import RandomLinkSplit
    from torch_geometric.utils import (
        to_networkx, from_networkx, negative_sampling,
        add_self_loops, remove_self_loops, to_undirected
    )
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not available — GNN cells will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available — visualization cells will be limited.")

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap as _umap_module
        UMAP = _umap_module.UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        warnings.warn("umap-learn not available — embedding visualization limited.")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    try:
        import community
        community_louvain = community
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False
        warnings.warn("python-louvain not available — community detection limited.")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# ENUMERATIONS — all categorical choices as enums (no magic strings)
# =============================================================================

class ArchitectureType(str, Enum):
    """Supported GNN architecture types for the NEURL Engine.

    Selecting the right architecture depends on graph structure:
    - GCN: fast, good for homogeneous graphs
    - GRAPHSAGE: inductive, scales to large graphs
    - GAT: attention-based, best when edge weights matter
    - GIN: most expressive (Weisfeiler-Lehman equivalent)
    - VGAE: generative model, best for link prediction tasks
    """
    GCN = "GCN"
    GRAPHSAGE = "GraphSAGE"
    GAT = "GAT"
    GIN = "GIN"
    VGAE = "VGAE"


class TaskType(str, Enum):
    """Task types supported by the NEURL Engine multi-task head.

    Each task type corresponds to a distinct downstream objective and
    influences which loss functions are activated during training.
    """
    ENTITY_DISCOVERY = "entity_discovery"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"
    GRAPH_CLASSIFICATION = "graph_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    COMMUNITY_DETECTION = "community_detection"


class NodeType(str, Enum):
    """Distinct node categories in the NEURL knowledge graph."""
    ENTITY = "entity"
    INSTANCE = "instance"


class EntityTypeLabel(str, Enum):
    """Semantic entity categories inferred from column patterns."""
    BUSINESS_OBJECT = "business_object"
    DOCUMENT = "document"
    TRANSACTION = "transaction"
    PERSON = "person"
    PRODUCT = "product"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class RelationshipTypeLabel(str, Enum):
    """Semantic relationship categories between entities."""
    HAS = "HAS"
    BELONGS_TO = "BELONGS_TO"
    GENERATES = "GENERATES"
    PROCESSES = "PROCESSES"
    CONTAINS = "CONTAINS"
    REFERENCES = "REFERENCES"
    RELATED_TO = "RELATED_TO"


class CardinalityType(str, Enum):
    """Cardinality labels for entity relationships."""
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_MANY = "N:N"


# =============================================================================
# CUSTOM EXCEPTIONS — structured error hierarchy for NEURL Engine
# =============================================================================

class NEURLBaseError(Exception):
    """Base exception for all NEURL Engine errors."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()


class NEURLDataError(NEURLBaseError):
    """Raised when data ingestion, validation, or feature engineering fails."""
    pass


class NEURLGraphError(NEURLBaseError):
    """Raised when graph construction or topology operations fail."""
    pass


class NEURLTrainingError(NEURLBaseError):
    """Raised when GNN training encounters an unrecoverable error."""
    pass


class NEURLInferenceError(NEURLBaseError):
    """Raised when inference or explainability computation fails."""
    pass


class NEURLConfigError(NEURLBaseError):
    """Raised when configuration validation fails."""
    pass


# =============================================================================
# NEURLConfig — single source of truth for all hyperparameters and paths
# =============================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for the data ingestion subsystem."""
    max_rows: int = 50_000          # cap to prevent OOM on large files
    sample_size: int = 500          # synthetic demo dataset size
    date_formats: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y%m%d"
    ])
    text_min_avg_tokens: int = 3    # min avg tokens to treat col as 'text'
    id_uniqueness_threshold: float = 0.95   # cols above this are potential IDs
    missing_drop_threshold: float = 0.80    # drop cols with >80% missing
    constant_col_n_unique: int = 1          # cols with ≤1 unique → constant


@dataclass
class FeatureEngineeringConfig:
    """Configuration for the feature engineering subsystem."""
    tfidf_max_features: int = 512       # TF-IDF vocabulary cap for col names
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    similarity_threshold: float = 0.3   # min cosine sim to call cols "similar"
    value_overlap_sample: int = 1000    # sample size for Jaccard computation
    feature_vector_dim: int = 15        # per-column metadata feature count
    top_k_frequency: int = 3            # top-K value frequencies to include
    entropy_bins: int = 20              # histogram bins for entropy estimate


@dataclass
class GraphConstructionConfig:
    """Configuration for knowledge graph construction."""
    max_instance_nodes_per_entity: int = 200    # cap to keep graph tractable
    min_fk_overlap_ratio: float = 0.80          # FK detection threshold
    edge_confidence_prune_threshold: float = 0.1
    louvain_resolution: float = 1.0             # community detection resolution
    isolated_node_removal: bool = True
    layout_algorithm: str = "spring"            # spring | kamada_kawai | hierarchical


@dataclass
class GNNArchitectureConfig:
    """GNN architecture selection and structural hyperparameters."""
    architecture: ArchitectureType = ArchitectureType.GCN
    hidden_dim: int = 64
    out_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.3
    gat_heads: int = 8                  # GAT-specific: number of attention heads
    sage_aggregation: str = "mean"      # GraphSAGE: mean | max | lstm
    gin_eps: float = 0.0                # GIN learnable epsilon init
    vgae_latent_dim: int = 16           # VGAE latent space dimension


@dataclass
class TrainingConfig:
    """Training hyperparameters for the GNN training pipeline."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 64
    grad_clip: float = 1.0              # gradient clipping max norm
    mixed_precision: bool = False       # use AMP (requires CUDA)
    grad_accumulation_steps: int = 1    # gradient accumulation for large batches
    warmup_epochs: int = 5              # linear LR warmup before cosine decay
    checkpoint_every_n_epochs: int = 10
    early_stopping_patience: int = 10
    early_stopping_metric: str = "auc"
    val_ratio: float = 0.1              # fraction of edges for validation
    test_ratio: float = 0.1             # fraction of edges for test
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])  # NeighborLoader


@dataclass
class LossWeightsConfig:
    """Weights for the multi-task learning loss combination."""
    link_prediction: float = 1.0
    node_reconstruction: float = 0.5
    kl_divergence: float = 0.001        # β-VAE style KL annealing weight
    contrastive: float = 0.3            # InfoNCE contrastive loss
    temperature: float = 0.07           # InfoNCE temperature


@dataclass
class InferenceConfig:
    """Configuration for the inference and explainability pipeline."""
    link_prediction_threshold: float = 0.5
    top_k_links: int = 20
    anomaly_z_threshold: float = 2.5    # z-score threshold for anomaly detection
    top_k_explanations: int = 10        # nodes to explain in detail
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "embedding_magnitude": 0.2,
        "pagerank": 0.4,
        "betweenness": 0.2,
        "degree": 0.2,
    })


@dataclass
class OutputConfig:
    """Output paths for model artifacts and visualizations."""
    output_dir: str = "./neurl_output"
    checkpoint_dir: str = "./neurl_output/checkpoints"
    export_dir: str = "./neurl_output/exports"
    visualization_dir: str = "./neurl_output/viz"


@dataclass
class NEURLConfig:
    """Master configuration dataclass for the entire NEURL Engine.

    Single source of truth: all subsystem configs are nested here.
    Pass this object to every subsystem constructor.

    Example:
        config = NEURLConfig()
        config.gnn.architecture = ArchitectureType.GAT
        config.training.epochs = 50
    """
    # Subsystem configs
    data: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    graph: GraphConstructionConfig = field(default_factory=GraphConstructionConfig)
    gnn: GNNArchitectureConfig = field(default_factory=GNNArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Global hyperparameters
    random_seed: int = 42
    experiment_name: str = "neurl_engine_v1"
    verbose: bool = True

    # Entity discovery hyperparameters
    entity_distance_threshold: float = 1.5     # Ward linkage cut height
    entity_name_similarity_merge: float = 0.5  # merge threshold (name sim)
    entity_min_cluster_size: int = 1            # min columns per entity

    # Relationship discovery hyperparameters
    relationship_confidence_threshold: float = 0.15
    fk_subset_threshold: float = 0.6           # FK value-subset overlap threshold

    def __post_init__(self) -> None:
        """Validate configuration consistency after initialization."""
        if self.training.val_ratio + self.training.test_ratio >= 1.0:
            raise NEURLConfigError(
                "val_ratio + test_ratio must be < 1.0",
                {"val": self.training.val_ratio, "test": self.training.test_ratio}
            )
        if self.entity_distance_threshold <= 0:
            raise NEURLConfigError("entity_distance_threshold must be > 0")
        # Create output directories
        for dir_path in [
            self.output.output_dir,
            self.output.checkpoint_dir,
            self.output.export_dir,
            self.output.visualization_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-compatible dict."""
        return asdict(self)


# =============================================================================
# NEURLLogger — structured, leveled logging with context
# =============================================================================

class NEURLLogger:
    """Centralized logging factory for the NEURL Engine.

    All subsystems should use NEURLLogger.get_logger(__name__) to obtain
    a named logger. This enables per-module log filtering and structured
    output in notebook environments.

    Example:
        logger = NEURLLogger.get_logger("MySubsystem")
        logger.info("Starting", extra_context={"param": value})
    """

    _FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    _DATE_FORMAT = "%H:%M:%S"
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Configure the root logger once on first use."""
        if not cls._initialized:
            logging.basicConfig(
                level=logging.INFO,
                format=cls._FORMAT,
                datefmt=cls._DATE_FORMAT,
            )
            # Suppress overly verbose third-party loggers
            for noisy_logger in ["urllib3", "matplotlib", "PIL", "numba"]:
                logging.getLogger(noisy_logger).setLevel(logging.WARNING)
            cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> "NEURLLoggerAdapter":
        """Get a named logger adapter.

        Args:
            name: Logger name (typically __name__ or class name).

        Returns:
            NEURLLoggerAdapter wrapping a standard Python logger.
        """
        cls._initialize()
        base_logger = logging.getLogger(f"NEURL.{name}")
        return NEURLLoggerAdapter(base_logger)


class NEURLLoggerAdapter:
    """Adapter adding extra_context support to standard loggers.

    Formats extra context dict as a JSON string appended to the message.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _format(self, message: str, extra_context: Optional[Dict] = None) -> str:
        if extra_context:
            try:
                ctx_str = json.dumps(extra_context, default=str)
                return f"{message} | {ctx_str}"
            except Exception:
                return f"{message} | (context serialization failed)"
        return message

    def info(self, message: str, extra_context: Optional[Dict] = None) -> None:
        self._logger.info(self._format(message, extra_context))

    def warning(self, message: str, extra_context: Optional[Dict] = None) -> None:
        self._logger.warning(self._format(message, extra_context))

    def error(self, message: str, extra_context: Optional[Dict] = None) -> None:
        self._logger.error(self._format(message, extra_context))

    def debug(self, message: str, extra_context: Optional[Dict] = None) -> None:
        self._logger.debug(self._format(message, extra_context))


# =============================================================================
# DeviceManager — GPU/CPU detection and device selection
# =============================================================================

class DeviceManager:
    """Selects and manages the compute device for PyTorch operations.

    Checks for CUDA availability, MPS (Apple Silicon), and falls back to CPU.
    Also sets global random seeds for reproducibility across all libraries.

    Example:
        device = DeviceManager.get_device()
        DeviceManager.set_seeds(42)
    """

    @staticmethod
    def get_device() -> torch.device:
        """Get the best available compute device.

        Priority: CUDA > MPS > CPU.

        Returns:
            torch.device object for the selected device.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA device: {device_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Apple MPS device")
        else:
            device = torch.device("cpu")
            print("✓ CPU device (no GPU available)")
        return device

    @staticmethod
    def set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility across all libraries.

        Args:
            seed: Integer seed value.
        """
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# ExperimentTracker — lightweight metric logging without MLflow dependency
# =============================================================================

class ExperimentTracker:
    """Lightweight experiment tracking with metric history and JSON export.

    Tracks training metrics (loss, AUC, etc.) per epoch and logs them to
    the console and a JSON file. No external MLflow/W&B dependency required.

    Example:
        tracker = ExperimentTracker("my_experiment", config)
        tracker.log_metrics({"loss": 0.5, "auc": 0.8}, step=1)
        tracker.save()
    """

    def __init__(self, experiment_name: str, config: "NEURLConfig") -> None:
        """Initialize the tracker.

        Args:
            experiment_name: Name for this experiment run.
            config: NEURLConfig for output directory configuration.
        """
        self._name = experiment_name
        self._config = config
        self._history: List[Dict[str, Any]] = []
        self._start_time = time.time()
        self._logger = NEURLLogger.get_logger("ExperimentTracker")
        self._logger.info(f"Experiment started: {experiment_name}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log a dict of metrics for a given step.

        Args:
            metrics: Dict of metric name → float value.
            step: Training step or epoch number.
            prefix: Optional prefix for metric names (e.g., "val/").
        """
        record = {
            "step": step,
            "elapsed_seconds": time.time() - self._start_time,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for k, v in metrics.items():
            key = f"{prefix}{k}" if prefix else k
            record[key] = float(v) if isinstance(v, (int, float)) else v
        self._history.append(record)

        # Console output
        metric_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        print(f"  [{prefix}step {step:3d}] {metric_str}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full metric history.

        Returns:
            List of step metric dicts.
        """
        return self._history

    def get_best(self, metric: str, mode: str = "max") -> Tuple[int, float]:
        """Get the step and value of the best metric.

        Args:
            metric: Metric name to search for.
            mode: 'max' or 'min'.

        Returns:
            Tuple of (best_step, best_value).
        """
        valid = [(r["step"], r[metric]) for r in self._history if metric in r]
        if not valid:
            return 0, 0.0
        return max(valid, key=lambda x: x[1]) if mode == "max" else min(valid, key=lambda x: x[1])

    def save(self) -> Path:
        """Save experiment history to a JSON file.

        Returns:
            Path to the saved JSON file.
        """
        output_path = Path(self._config.output.output_dir) / f"{self._name}_history.json"
        try:
            with open(output_path, "w") as f:
                json.dump(self._history, f, indent=2, default=str)
            self._logger.info(f"Experiment saved: {output_path}")
        except Exception as e:
            self._logger.warning(f"Could not save experiment: {e}")
        return output_path


# =============================================================================
# ModelRegistry — model versioning and checkpoint management
# =============================================================================

class ModelRegistry:
    """Manages model checkpoints and versioned artifact storage.

    Provides save/load operations for PyTorch model state dicts,
    with metadata tracking (epoch, metrics, config hash).

    Example:
        registry = ModelRegistry(config)
        path = registry.save(model, epoch=10, metrics={"auc": 0.92})
        model = registry.load(model, path)
    """

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the registry.

        Args:
            config: NEURLConfig for checkpoint directory configuration.
        """
        self._config = config
        self._checkpoint_dir = Path(config.output.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._logger = NEURLLogger.get_logger("ModelRegistry")

    def save(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save model checkpoint with metadata.

        Args:
            model: PyTorch model to save.
            epoch: Current epoch number (used in filename).
            metrics: Optional dict of eval metrics to store in checkpoint.

        Returns:
            Path to the saved checkpoint file.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._config.experiment_name}_epoch{epoch:04d}_{timestamp}.pt"
        checkpoint_path = self._checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "config": self._config.to_dict(),
            "timestamp": timestamp,
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            self._logger.info(
                f"Checkpoint saved: {filename}",
                extra_context={"epoch": epoch, "metrics": metrics}
            )
        except Exception as e:
            self._logger.error(f"Checkpoint save failed: {e}")

        return checkpoint_path

    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: Union[str, Path],
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load model from checkpoint.

        Args:
            model: Uninitialized or randomly initialized model (same architecture).
            checkpoint_path: Path to checkpoint file.

        Returns:
            Tuple of (model_with_loaded_weights, checkpoint_metadata_dict).
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise NEURLTrainingError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        self._logger.info(
            f"Checkpoint loaded: {path.name}",
            extra_context={"epoch": checkpoint.get("epoch"), "metrics": checkpoint.get("metrics")}
        )
        return model, checkpoint


# =============================================================================
# BOOTSTRAP FUNCTION — run this to initialize the NEURL Engine
# =============================================================================

def bootstrap_neurl_engine(
    experiment_name: str = "neurl_engine_v1",
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple["NEURLConfig", torch.device, "ExperimentTracker", "ModelRegistry"]:
    """Bootstrap the NEURL Engine: config, device, tracker, registry.

    This is the single entry point for Cell 1. Call it to get all the
    infrastructure objects needed by the rest of the pipeline.

    Args:
        experiment_name: Name for this experiment run.
        random_seed: Global random seed for reproducibility.
        verbose: Whether to print initialization summary.

    Returns:
        Tuple of (config, device, experiment_tracker, model_registry).
    """
    config = NEURLConfig(experiment_name=experiment_name, random_seed=random_seed)
    device = DeviceManager.get_device()
    DeviceManager.set_seeds(random_seed)

    tracker = ExperimentTracker(experiment_name, config)
    registry = ModelRegistry(config)

    if verbose:
        print("\n" + "=" * 60)
        print("   NEURL ENGINE — Neural Knowledge Graph Discovery")
        print("=" * 60)
        print(f"  Experiment : {experiment_name}")
        print(f"  Device     : {device}")
        print(f"  Seed       : {random_seed}")
        print(f"  scipy      : {'✓' if SCIPY_AVAILABLE else '✗ (numpy fallbacks)'}")
        print(f"  sklearn    : {'✓' if SKLEARN_AVAILABLE else '✗ (numpy fallbacks)'}")
        print(f"  PyG        : {'✓' if TORCH_GEOMETRIC_AVAILABLE else '✗'}")
        print(f"  Plotly     : {'✓' if PLOTLY_AVAILABLE else '✗'}")
        print(f"  UMAP       : {'✓' if UMAP_AVAILABLE else '✗'}")
        print(f"  Louvain    : {'✓' if LOUVAIN_AVAILABLE else '✗'}")
        print("=" * 60 + "\n")

    return config, device, tracker, registry


# =============================================================================
# CELL 1 ENTRY POINT — run bootstrap and expose to session scope
# =============================================================================
config, device, tracker, registry = bootstrap_neurl_engine()
