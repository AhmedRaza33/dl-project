"""Shared utilities for the MLP experimental study.

Centralised here so the training script, the ablation script, the comparison
script, and the Jupyter notebook all use the **exact same data splits and
preprocessing**. This makes every reported number directly comparable.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & reproducibility
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "processed_creditcard.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MLP_DIR = ARTIFACTS_DIR / "mlp_study"
COMPARE_DIR = ARTIFACTS_DIR / "comparison"
MLP_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def describe(self) -> str:
        def _row(name, y):
            n = len(y)
            pos = int(y.sum())
            return f"  {name:<7}: n={n:>6}  fraud={pos:>4} ({100*pos/n:.3f}%)"
        return "\n".join([
            f"Features (n={self.n_features}): {self.feature_names}",
            _row("train", self.y_train),
            _row("val",   self.y_val),
            _row("test",  self.y_test),
        ])


def load_splits(test_size: float = 0.15,
                val_size: float = 0.15,
                seed: int = SEED) -> Splits:
    """Stratified train/val/test split of the **full** preprocessed dataset.

    Unlike the autoencoder pipeline (which trains only on normals), the MLP is
    a *supervised* classifier — both classes must be present in every split.
    """
    from sklearn.model_selection import train_test_split

    assert DATA_PATH.exists(), (
        f"Could not find {DATA_PATH}. Run fraud_detection_preprocessing.ipynb "
        "to generate processed_creditcard.csv first."
    )
    df = pd.read_csv(DATA_PATH)
    feature_names = [c for c in df.columns if c != "Class"]
    X = df[feature_names].values.astype(np.float32)
    y = df["Class"].values.astype(np.int32)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    # val_size is given as fraction of the whole dataset, so adjust within trainval
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative,
        random_state=seed, stratify=y_trainval,
    )
    return Splits(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        feature_names=feature_names,
    )


def class_weight_dict(y: np.ndarray) -> dict:
    """Inverse-frequency class weights for the binary target."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {0: 1.0, 1: float(n_neg) / max(n_pos, 1)}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    """Compute the headline binary-classification metrics from raw scores."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        precision_recall_curve,
    )

    if threshold is None:
        # Choose threshold that maximises F1 on the supplied scores (used for
        # validation-set tuning; for the test set we always pass a fixed one).
        prec, rec, thr = precision_recall_curve(y_true, scores)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
        threshold = float(thr[best_idx]) if len(thr) else 0.5

    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, scores)),
        "pr_auc":    float(average_precision_score(y_true, scores)),
        "confusion_matrix": cm.tolist(),
    }


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))
