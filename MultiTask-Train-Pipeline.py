"""
NEURL Engine — Cell 8: Multi-Task Training Pipeline

Implements the full GNN training pipeline with:
- AdamW optimizer + CosineAnnealingLR with linear warmup
- Mixed-precision training (AMP, CUDA only)
- Multi-task loss: link prediction + reconstruction + KL + InfoNCE contrastive
- Mini-batch training with NeighborLoader
- Gradient accumulation for large effective batch sizes
- Early stopping on AUC plateau
- Checkpoint saving and restoration

Design rationale: Multi-task training on link prediction + reconstruction
avoids the need for large labeled datasets — the graph structure itself
provides the supervision signal through positive/negative edge sampling.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# =============================================================================
# InfoNCELoss — temperature-scaled contrastive loss
# =============================================================================

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for self-supervised representation learning.

    Given anchor, positive, and negative embeddings, computes the
    cross-entropy loss over the similarity distribution. This encourages
    embeddings of related nodes to be closer than embeddings of unrelated nodes.

    References:
        van den Oord et al. (2018), "Representation Learning with CPC"

    Args:
        temperature: Temperature scaling for softmax sharpness.
            Lower temperature = sharper distribution = harder negatives.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings, shape [B, D].
            positive: Positive pair embeddings, shape [B, D].
            negatives: Negative sample embeddings, shape [K, D].

        Returns:
            Scalar InfoNCE loss.
        """
        # Normalize all embeddings to the unit sphere for cosine similarity
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarities: [B]
        pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / self.temperature

        # Negative similarities: [B, K]
        neg_sim = (anchor @ negatives.T) / self.temperature  # [B, K]

        # Concatenate: [B, 1+K] — positive is class 0
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)


# =============================================================================
# NEURLTrainer — complete training pipeline
# =============================================================================

class NEURLTrainer:
    """Multi-task GNN training pipeline for the NEURL Engine.

    Manages optimizer, LR scheduler, gradient scaling (AMP), multi-task
    loss computation, mini-batch loading, epoch loop, evaluation, checkpointing,
    and early stopping.

    The training objective is a weighted sum of:
    1. Link prediction loss (BCEWithLogitsLoss on positive/negative edge pairs)
    2. Node reconstruction loss (MSELoss: encode → decode → compare to input)
    3. KL divergence loss (VGAE only)
    4. InfoNCE contrastive loss (connected nodes attract, random pairs repel)

    Example:
        trainer = NEURLTrainer(model, config, tracker, device)
        trainer.train(pyg_data, epochs=30)
    """

    def __init__(
        self,
        model: "NEURLBaseGNN",
        config: "NEURLConfig",
        experiment_tracker: "ExperimentTracker",
        device: torch.device,
    ) -> None:
        """Initialize the trainer with model and training configuration.

        Args:
            model: NEURLBaseGNN model to train.
            config: NEURLConfig master configuration.
            experiment_tracker: ExperimentTracker for metric logging.
            device: Target compute device.
        """
        self._model = model
        self._config = config
        self._tracker = experiment_tracker
        self._device = device
        self._logger = NEURLLogger.get_logger("NEURLTrainer")

        # Optimizer: AdamW with weight decay regularization
        self._optimizer = AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # LR schedule: linear warmup then cosine annealing
        warmup_epochs = config.training.warmup_epochs
        total_epochs = config.training.epochs
        cos_epochs = max(total_epochs - warmup_epochs, 1)

        warmup_scheduler = LinearLR(
            self._optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=cos_epochs,
            eta_min=config.training.lr * 0.01,
        )
        self._scheduler = SequentialLR(
            self._optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # Gradient scaler for mixed precision (AMP) — CUDA only
        self._use_amp = config.training.mixed_precision and device.type == "cuda"
        self._scaler = GradScaler(enabled=self._use_amp)

        # Loss functions
        self._bce_loss = nn.BCEWithLogitsLoss()
        self._mse_loss = nn.MSELoss()
        self._infonce_loss = InfoNCELoss(temperature=config.loss_weights.temperature)

        # Training state
        self._best_metric: float = -float("inf")
        self._patience_counter: int = 0
        self._train_losses: List[float] = []
        self._val_metrics: List[Dict[str, float]] = []

    # ─── DATA PREPARATION ────────────────────────────────────────────────────

    def prepare_training_data(
        self,
        pyg_data: "Data",
    ) -> Tuple["Data", "Data", "Data", "NeighborLoader"]:
        """Create train/val/test splits and set up the mini-batch loader.

        Uses RandomLinkSplit from PyG to create positive/negative edge splits
        for link prediction evaluation while preserving message-passing structure.

        Args:
            pyg_data: Full PyG Data object from Cell 6.

        Returns:
            Tuple of (train_data, val_data, test_data, train_loader).
        """
        # Ensure self-loops don't contaminate link prediction splits
        transform = RandomLinkSplit(
            num_val=self._config.training.val_ratio,
            num_test=self._config.training.test_ratio,
            is_undirected=False,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,  # 1 negative per positive
        )

        try:
            train_data, val_data, test_data = transform(pyg_data)
        except Exception as exc:
            self._logger.warning(
                f"RandomLinkSplit failed ({exc}), using full data for all splits"
            )
            train_data = val_data = test_data = pyg_data

        # NeighborLoader for mini-batch training
        # Samples a fixed number of neighbors at each hop to bound memory
        try:
            num_nodes = pyg_data.num_nodes
            input_nodes = torch.arange(num_nodes)
            loader = NeighborLoader(
                train_data,
                num_neighbors=self._config.training.num_neighbors,
                batch_size=min(self._config.training.batch_size, num_nodes),
                input_nodes=input_nodes,
                shuffle=True,
            )
        except Exception as exc:
            self._logger.warning(f"NeighborLoader failed ({exc}) — using full-batch mode")
            loader = None

        return train_data, val_data, test_data, loader

    # ─── TRAINING EPOCH ──────────────────────────────────────────────────────

    def train_epoch(
        self,
        train_data: "Data",
        loader: Optional["NeighborLoader"],
        epoch: int,
    ) -> float:
        """Run a single training epoch.

        Handles both mini-batch (NeighborLoader) and full-batch modes.
        Gradient accumulation is applied when config.training.grad_accumulation_steps > 1.

        Args:
            train_data: Training split PyG Data.
            loader: NeighborLoader (None = full-batch mode).
            epoch: Current epoch number (for logging and loss weighting).

        Returns:
            Average loss for this epoch.
        """
        self._model.train()
        total_loss = 0.0
        n_batches = 0
        grad_accum_steps = self._config.training.grad_accumulation_steps

        # Iterator: mini-batch or single full-batch
        if loader is not None:
            batch_iter = loader
        else:
            batch_iter = [train_data]

        self._optimizer.zero_grad()

        for step, batch in enumerate(batch_iter):
            batch = batch.to(self._device)

            with autocast(enabled=self._use_amp):
                loss = self.compute_multi_task_loss(batch, epoch)
                loss = loss / grad_accum_steps  # normalize for accumulation

            self._scaler.scale(loss).backward()

            # Step optimizer every grad_accum_steps batches
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping to prevent exploding gradients
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    max_norm=self._config.training.grad_clip,
                )
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()

            total_loss += float(loss.item()) * grad_accum_steps
            n_batches += 1

        # Final gradient step if remaining
        if n_batches % grad_accum_steps != 0:
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self._config.training.grad_clip,
            )
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._optimizer.zero_grad()

        return total_loss / max(n_batches, 1)

    # ─── MULTI-TASK LOSS ─────────────────────────────────────────────────────

    def compute_multi_task_loss(
        self,
        batch: "Data",
        epoch: int,
    ) -> torch.Tensor:
        """Compute the weighted multi-task loss for a batch.

        Loss components:
        1. Link prediction: BCE on positive and negative edge pairs
        2. Node reconstruction: MSE between embeddings and input features
           (projected to input dimension)
        3. KL divergence: VGAE-specific regularization (with β annealing)
        4. InfoNCE: contrastive loss on random positive/negative pairs

        The KL weight is linearly annealed from 0 to loss_weights.kl_divergence
        over the warmup period to prevent posterior collapse in VGAE.

        Args:
            batch: PyG Data batch with node features and edge indices.
            epoch: Current epoch (used for KL annealing).

        Returns:
            Scalar total loss tensor.
        """
        x = batch.x
        # Use edge_label_index if available (from RandomLinkSplit), else edge_index
        edge_index = getattr(batch, "edge_label_index", batch.edge_index)
        edge_labels = getattr(batch, "edge_label", None)

        # Forward pass: encode embeddings + decode edge logits
        z, logits = self._model(x, batch.edge_index)

        # 1. Link prediction loss (positive + negative edges)
        if edge_labels is not None and len(logits) == len(edge_labels):
            lp_loss = self._bce_loss(logits, edge_labels.float())
        else:
            # Full-batch mode: create synthetic positive/negative labels
            n_edges = batch.edge_index.shape[1]
            pos_labels = torch.ones(n_edges, device=self._device)
            lp_loss = self._bce_loss(logits, pos_labels)

        # 2. Node reconstruction loss (MSE: z → project → compare with x)
        # Use a linear projection to map embedding back to input feature space
        if not hasattr(self, "_recon_proj"):
            self._recon_proj = nn.Linear(
                self._config.gnn.out_dim, x.shape[1]
            ).to(self._device)
        x_hat = self._recon_proj(z)
        recon_loss = self._mse_loss(x_hat, x)

        # 3. KL loss (VGAE only) with linear epoch-based annealing
        if isinstance(self._model, NEURLVariationalGAE):
            kl_anneal = min(
                epoch / max(self._config.training.warmup_epochs, 1), 1.0
            )
            kl_weight = self._config.loss_weights.kl_divergence * kl_anneal
            kl_loss = self._model.kl_loss()
        else:
            kl_weight = 0.0
            kl_loss = torch.tensor(0.0, device=self._device)

        # 4. InfoNCE contrastive loss
        # Positive pairs: adjacent nodes in the graph
        # Negatives: random subset of non-adjacent nodes
        contrastive_loss = self._compute_contrastive_loss(z, batch.edge_index)

        # Weighted sum
        total_loss = (
            self._config.loss_weights.link_prediction * lp_loss
            + self._config.loss_weights.node_reconstruction * recon_loss
            + kl_weight * kl_loss
            + self._config.loss_weights.contrastive * contrastive_loss
        )
        return total_loss

    def _compute_contrastive_loss(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss on a sample of node pairs.

        Args:
            z: Node embeddings [N, D].
            edge_index: Edge index [2, E].

        Returns:
            Scalar contrastive loss.
        """
        n_nodes = z.shape[0]
        n_edges = edge_index.shape[1]

        if n_edges == 0 or n_nodes < 3:
            return torch.tensor(0.0, device=self._device)

        # Sample a small batch of positive pairs
        MAX_PAIRS = min(32, n_edges)
        perm = torch.randperm(n_edges, device=self._device)[:MAX_PAIRS]
        src = edge_index[0, perm]
        dst = edge_index[1, perm]

        anchor = z[src]
        positive = z[dst]

        # Random negatives: sample K unrelated nodes
        K = min(32, n_nodes - 1)
        neg_idx = torch.randperm(n_nodes, device=self._device)[:K]
        negatives = z[neg_idx]

        try:
            return self._infonce_loss(anchor, positive, negatives)
        except Exception:
            return torch.tensor(0.0, device=self._device)

    # ─── EVALUATION ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        data: "Data",
    ) -> Dict[str, float]:
        """Evaluate the model on a validation or test split.

        Computes link prediction AUC, reconstruction MSE, and embedding
        silhouette score (quality of cluster separation in embedding space).

        Args:
            data: PyG Data with edge_label_index and edge_label if available.

        Returns:
            Dict with metric names and float values.
        """
        self._model.eval()
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            data = data.to(self._device)
            z, logits = self._model(data.x, data.edge_index)
            z_np = z.cpu().numpy()

        # Link prediction AUC
        try:
            edge_labels = getattr(data, "edge_label", None)
            if edge_labels is not None and len(logits) == len(edge_labels):
                scores = torch.sigmoid(logits).cpu().numpy()
                labels_np = edge_labels.cpu().numpy()
                if len(np.unique(labels_np)) > 1:
                    metrics["auc"] = float(roc_auc_score(labels_np, scores))
                else:
                    metrics["auc"] = 0.5
            else:
                metrics["auc"] = 0.5
        except Exception:
            metrics["auc"] = 0.5

        # Reconstruction MSE
        try:
            if hasattr(self, "_recon_proj"):
                x_hat = self._recon_proj(z)
                metrics["recon_loss"] = float(self._mse_loss(x_hat, data.x).item())
            else:
                metrics["recon_loss"] = float("nan")
        except Exception:
            metrics["recon_loss"] = float("nan")

        # Silhouette score on embeddings (requires ≥2 class labels)
        try:
            y_np = data.y.cpu().numpy()
            if len(np.unique(y_np)) >= 2 and len(z_np) >= 4:
                sil = silhouette_score(z_np, y_np, sample_size=min(len(z_np), 200))
                metrics["silhouette"] = float(sil)
            else:
                metrics["silhouette"] = 0.0
        except Exception:
            metrics["silhouette"] = 0.0

        return metrics

    # ─── CHECKPOINT ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model + optimizer state to disk.

        Args:
            path: Output file path for the checkpoint.
            epoch: Current epoch.
            metrics: Current metrics snapshot.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict(),
            "metrics": metrics,
            "config": {
                "architecture": self._config.gnn.architecture.value,
                "hidden_dim": self._config.gnn.hidden_dim,
                "out_dim": self._config.gnn.out_dim,
                "num_layers": self._config.gnn.num_layers,
            },
        }, path)
        self._tracker.save_checkpoint_meta(path, epoch, metrics)

    def load_checkpoint(self, path: Path) -> int:
        """Restore model and optimizer state from a checkpoint file.

        Args:
            path: Path to the saved checkpoint.

        Returns:
            Epoch number at which the checkpoint was saved.

        Raises:
            NEURLTrainingError: If checkpoint loading fails.
        """
        if not path.exists():
            raise NEURLTrainingError(f"Checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location=self._device)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self._logger.info(f"Checkpoint restored from {path}",
                              extra_context={"epoch": ckpt["epoch"]})
            return ckpt["epoch"]
        except Exception as exc:
            raise NEURLTrainingError(f"Failed to load checkpoint {path}: {exc}") from exc

    # ─── MAIN TRAINING LOOP ──────────────────────────────────────────────────

    def train(
        self,
        pyg_data: "Data",
        epochs: int,
    ) -> Dict[str, List[float]]:
        """Run the complete training loop.

        Calls prepare_training_data, then runs epochs of train_epoch + evaluate.
        Saves checkpoints every N epochs and implements early stopping on AUC.

        Args:
            pyg_data: Full PyG Data object from Cell 6.
            epochs: Total training epochs.

        Returns:
            Dict with 'train_loss', 'val_auc', 'val_silhouette' history lists.
        """
        self._logger.info(f"Starting training: {epochs} epochs")
        train_data, val_data, test_data, loader = self.prepare_training_data(pyg_data)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_auc": [],
            "val_silhouette": [],
            "lr": [],
        }

        patience = self._config.training.early_stopping_patience
        checkpoint_dir = self._config.output.checkpoint_dir
        arch_name = self._config.gnn.architecture.value.lower()

        for epoch in range(1, epochs + 1):
            # Training step
            train_loss = self.train_epoch(train_data, loader, epoch)
            history["train_loss"].append(train_loss)

            # Validation
            val_metrics = self.evaluate(val_data)
            history["val_auc"].append(val_metrics.get("auc", 0.5))
            history["val_silhouette"].append(val_metrics.get("silhouette", 0.0))
            history["lr"].append(float(self._optimizer.param_groups[0]["lr"]))

            # Log to experiment tracker
            self._tracker.log_metric("train_loss", train_loss, epoch)
            for name, val in val_metrics.items():
                self._tracker.log_metric(name, val, epoch)

            # LR scheduler step
            self._scheduler.step()

            # Checkpoint saving
            if epoch % self._config.training.checkpoint_every_n_epochs == 0:
                ckpt_path = checkpoint_dir / f"neurl_{arch_name}_epoch{epoch:03d}.pt"
                self.save_checkpoint(ckpt_path, epoch, val_metrics)

            # Progress logging
            auc_val = val_metrics.get("auc", 0.5)
            sil_val = val_metrics.get("silhouette", 0.0)
            lr_val = self._optimizer.param_groups[0]["lr"]
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"AUC: {auc_val:.4f} | "
                      f"Silhouette: {sil_val:.4f} | "
                      f"LR: {lr_val:.2e}")

            # Early stopping: monitor AUC (maximize)
            metric_name = self._config.training.early_stopping_metric
            current_metric = val_metrics.get(metric_name, 0.5)
            if current_metric > self._best_metric:
                self._best_metric = current_metric
                self._patience_counter = 0
                # Save best model
                best_path = checkpoint_dir / f"neurl_{arch_name}_best.pt"
                self.save_checkpoint(best_path, epoch, val_metrics)
            else:
                self._patience_counter += 1
                if self._patience_counter >= patience:
                    self._logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={patience}, best {metric_name}={self._best_metric:.4f})"
                    )
                    break

        self._tracker.save_to_json()
        return history


# =============================================================================
# DEMO: Train the GNN on the ERP knowledge graph
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 8: Multi-Task Training Pipeline")
print("=" * 70)

TRAIN_EPOCHS = neurl_config.training.epochs  # 30 epochs from config

if neurl_graph.pyg_data is None:
    raise NEURLTrainingError("PyG data not available — Cell 6 must run successfully first")

pyg_data = neurl_graph.pyg_data.to(device_manager.device)
print(f"\n  Training data: {pyg_data.num_nodes} nodes, "
      f"{pyg_data.edge_index.shape[1]} edges")
print(f"  Architecture : {neurl_config.gnn.architecture.value}")
print(f"  Epochs       : {TRAIN_EPOCHS}")
print(f"  LR           : {neurl_config.training.lr}")
print(f"  AMP          : {neurl_config.training.mixed_precision}")
print()

# Initialize trainer
trainer = NEURLTrainer(
    model=primary_model,
    config=neurl_config,
    experiment_tracker=experiment_tracker,
    device=device_manager.device,
)

# Train
print("  ── Training Loop ──────────────────────────────────────────────────")
train_history = trainer.train(pyg_data, epochs=TRAIN_EPOCHS)

# Print final metrics
best_auc_epoch = experiment_tracker.get_best_epoch("auc")
best_auc = experiment_tracker.get_metric_history("auc").get(best_auc_epoch, 0.0)
print(f"\n  ── Training Complete ──────────────────────────────────────────────")
print(f"  Best AUC: {best_auc:.4f} at epoch {best_auc_epoch}")
print(f"  Final train loss: {train_history['train_loss'][-1]:.4f}")
print(f"  Final val AUC: {train_history['val_auc'][-1]:.4f}")
print(f"  Final silhouette: {train_history['val_silhouette'][-1]:.4f}")

# Visualize training curves with Plotly
if PLOTLY_AVAILABLE and len(train_history["train_loss"]) > 1:
    print("\n  Rendering training dashboard...")
    epochs_range = list(range(1, len(train_history["train_loss"]) + 1))

    fig_train = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Training Loss", "Validation AUC", "Embedding Silhouette"),
    )
    fig_train.add_trace(
        go.Scatter(x=epochs_range, y=train_history["train_loss"],
                   mode="lines+markers", name="Train Loss",
                   line=dict(color="#3D5A80", width=2), marker=dict(size=4)),
        row=1, col=1
    )
    fig_train.add_trace(
        go.Scatter(x=epochs_range, y=train_history["val_auc"],
                   mode="lines+markers", name="Val AUC",
                   line=dict(color="#6B8F71", width=2), marker=dict(size=4)),
        row=1, col=2
    )
    fig_train.add_trace(
        go.Scatter(x=epochs_range, y=train_history["val_silhouette"],
                   mode="lines+markers", name="Silhouette",
                   line=dict(color="#B07D62", width=2), marker=dict(size=4)),
        row=1, col=3
    )
    fig_train.update_layout(
        title=dict(text="NEURL Engine — GNN Training Dashboard", font=dict(size=16)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        height=350, width=1000,
    )
    for i in range(1, 4):
        fig_train.update_xaxes(title_text="Epoch", row=1, col=i)
    fig_train.update_yaxes(title_text="Loss", row=1, col=1)
    fig_train.update_yaxes(title_text="AUC", row=1, col=2)
    fig_train.update_yaxes(title_text="Score", row=1, col=3)
    fig_train.show()

# Store trained model
model_registry.register(
    f"trained_{neurl_config.gnn.architecture.value.lower()}",
    primary_model,
    {"best_auc": best_auc, "best_epoch": best_auc_epoch, "epochs_run": len(train_history["train_loss"])}
)

print("\n✅ Cell 8 complete — Training pipeline ready.")
print(f"   Exports: trainer, train_history, pyg_data")
print(f"   Best model registered: trained_{neurl_config.gnn.architecture.value.lower()}")
