"""Training harness — port of transformer_forecaster.ipynb cells 22/27.

Same optimizer/scheduler/AMP/early-stopping/metrics as the transformer zoo so the
LSTM is judged on an identical footing. Two hardening changes vs the notebook:
  1. ``torch.amp.GradScaler(device_type, ...)`` (the modern API) instead of the
     deprecated ``torch.cuda.amp.GradScaler`` that warns/breaks on recent torch.
  2. ``cleanup_gpu()`` runs in a ``finally`` so an OOM mid-epoch still frees memory
     (the user hit CUDA OOM previously).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sentisense.config import LR, MAX_EPOCHS, PATIENCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_gpu() -> None:
    """Free cached GPU memory + run gc (mitigates the prior CUDA OOM)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(model: nn.Module, dl_tr: DataLoader, dl_va: DataLoader, *,
                lr: float = LR, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                weight_decay: float = 1e-4, model_name: str = "model",
                save_dir: Path | None = None) -> dict[str, Any]:
    """Train with class-weighted BCE, AMP, cosine LR, early stop on val_loss."""
    from sentisense.models.sequence import compute_class_weights

    model = model.to(DEVICE)
    class_weights = compute_class_weights(dl_tr).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balacc": []}

    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for X, y in dl_tr:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    logits = model(X)
                    loss = F.binary_cross_entropy_with_logits(logits, y, weight=class_weights[y.long()])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            preds, labels = [], []
            with torch.no_grad():
                for X, y in dl_va:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
                        logits = model(X)
                        loss = F.binary_cross_entropy_with_logits(logits, y, weight=class_weights[y.long()])
                    val_loss += loss.item()
                    preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                    labels.extend(y.cpu().numpy())

            train_loss /= max(len(dl_tr), 1)
            val_loss /= max(len(dl_va), 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(accuracy_score(labels, preds) if labels else 0.0)
            history["val_balacc"].append(balanced_accuracy_score(labels, preds) if labels else 0.0)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch {} ({}).", epoch, model_name)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_dir / f"{model_name}.pt")
    finally:
        cleanup_gpu()

    return {"model": model.cpu(), "history": history, "best_val_loss": best_val_loss}


def evaluate_on_test(model: nn.Module, dl_te: DataLoader) -> dict[str, float]:
    """Test metrics: accuracy, balanced accuracy, macro-F1, ROC-AUC, MCC."""
    model = model.to(DEVICE)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in dl_te:
            X, y = X.to(DEVICE), y.to(DEVICE)
            probs.extend(torch.sigmoid(model(X)).cpu().numpy())
            labels.extend(y.cpu().numpy())
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        "mcc": float(matthews_corrcoef(labels, preds)),
    }
