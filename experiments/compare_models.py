"""Head-to-head comparison: best MLP vs the trained Autoencoder.

Both models are evaluated **on the same supervised test split** that the MLP
study uses, so the numbers are directly comparable. The autoencoder produces
an anomaly score (reconstruction MSE) which we threshold using the value
chosen on the autoencoder's own validation set; the MLP produces a fraud
probability thresholded at the value found by maximising F1 on the MLP
validation set.

Outputs (under ``artifacts/comparison/``):
    - comparison_metrics.csv     side-by-side table
    - comparison_metrics.json    same data, machine-readable
    - cm_mlp.png, cm_ae.png      confusion matrices
    - roc_pr.png                 ROC and PR curves overlaid
    - efficiency.png             bar chart: params, train time, inference time
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    ARTIFACTS_DIR, COMPARE_DIR, MLP_DIR, SEED,
    evaluate_scores, load_splits, save_json, set_global_seed,
)


AE_MODEL_PATH  = ARTIFACTS_DIR / "autoencoder_fraud.keras"   # canonical AE (trained by the notebook on the shared split)
MLP_MODEL_PATH = MLP_DIR / "mlp_best.keras"
MLP_BEST_JSON  = MLP_DIR / "7_best.json"


def _plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Fraud"])
    ax.set_yticks([0, 1], ["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_curves(y_true: np.ndarray,
                 scores_mlp: np.ndarray,
                 scores_ae: np.ndarray,
                 path: Path) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for name, scores, color in [("MLP (best)", scores_mlp, "C0"),
                                 ("Autoencoder",  scores_ae,  "C3")]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        axes[0].plot(fpr, tpr, color=color, label=f"{name}  AUC={auc(fpr,tpr):.4f}")
        prec, rec, _ = precision_recall_curve(y_true, scores)
        axes[1].plot(rec, prec, color=color,
                     label=f"{name}  AP={average_precision_score(y_true, scores):.4f}")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC"); axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall"); axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_efficiency(records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    names = [r["name"] for r in records]
    colors = ["C0", "C3"]

    axes[0].bar(names, [r["n_params"] for r in records], color=colors)
    axes[0].set_title("Parameters"); axes[0].set_ylabel("# params")

    axes[1].bar(names, [r["train_time_sec"] for r in records], color=colors)
    axes[1].set_title("Training time"); axes[1].set_ylabel("seconds")

    axes[2].bar(names, [r["inference_ms_per_1k"] for r in records], color=colors)
    axes[2].set_title("Inference latency"); axes[2].set_ylabel("ms / 1k samples")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    set_global_seed(SEED)
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    splits = load_splits()
    X_val,  y_val  = splits.X_val,  splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    # ---------- MLP ----------
    mlp_rec = json.loads(MLP_BEST_JSON.read_text())
    mlp = load_model(MLP_MODEL_PATH)
    mlp_val_scores  = mlp.predict(X_val,  batch_size=8192, verbose=0).ravel()
    t0 = time.time()
    mlp_scores = mlp.predict(X_test, batch_size=8192, verbose=0).ravel()
    mlp_infer = (time.time() - t0) * 1000 * 1000 / len(X_test)  # ms per 1k samples

    # ---------- Autoencoder (trained by fraud_detection_autoencoder.ipynb on the SAME shared split) ----------
    ae = load_model(AE_MODEL_PATH, compile=False)
    ae_params = int(ae.count_params())
    X_val_hat  = ae.predict(X_val,  batch_size=8192, verbose=0)
    ae_val_scores  = np.mean((X_val  - X_val_hat ) ** 2, axis=1)
    t0 = time.time()
    X_hat = ae.predict(X_test, batch_size=8192, verbose=0)
    ae_scores = np.mean((X_test - X_hat) ** 2, axis=1)
    ae_infer = (time.time() - t0) * 1000 * 1000 / len(X_test)

    # Identical fair-comparison protocol: pick threshold that maximises F1 on
    # the (shared) validation set, then apply it on the held-out test set.
    mlp_val_metrics = evaluate_scores(y_val, mlp_val_scores, threshold=None)
    ae_val_metrics  = evaluate_scores(y_val, ae_val_scores,  threshold=None)
    mlp_metrics = evaluate_scores(y_test, mlp_scores, threshold=mlp_val_metrics["threshold"])
    ae_metrics  = evaluate_scores(y_test, ae_scores,  threshold=ae_val_metrics["threshold"])

    # The AE notebook doesn't persist a structured training-time JSON like the
    # MLP study, so we capture what we need on the fly. Training time is taken
    # from the notebook's recorded history if available.
    ae_train_time = float("nan")
    history_path = ARTIFACTS_DIR / "history.json"
    if history_path.exists():
        try:
            hist = json.loads(history_path.read_text())
            ae_train_time = float(hist.get("train_time_sec", float("nan")))
        except Exception:
            pass
    ae_rec = {"n_params": ae_params, "train_time_sec": ae_train_time}

    # ---------- Save raw scores so the notebook can reproduce charts ----------
    np.save(COMPARE_DIR / "mlp_scores.npy", mlp_scores)
    np.save(COMPARE_DIR / "ae_scores.npy",  ae_scores)
    np.save(COMPARE_DIR / "y_test.npy",     y_test)

    # ---------- Tabulate ----------
    rows = []
    for name, m, params, ttime, infer in [
        ("MLP (best)",  mlp_metrics, mlp_rec["n_params"], mlp_rec["train_time_sec"], mlp_infer),
        ("Autoencoder", ae_metrics,  ae_rec["n_params"],  ae_rec["train_time_sec"],  ae_infer),
    ]:
        rows.append({
            "model": name,
            "params": params,
            "train_time_sec": ttime,
            "inference_ms_per_1k": infer,
            "threshold": m["threshold"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMPARE_DIR / "comparison_metrics.csv", index=False)
    print(df.to_string(index=False))

    save_json(COMPARE_DIR / "comparison_metrics.json", {
        "mlp": mlp_metrics, "autoencoder": ae_metrics,
        "mlp_params": mlp_rec["n_params"], "ae_params": ae_rec["n_params"],
        "mlp_train_time_sec": mlp_rec["train_time_sec"],
        "ae_train_time_sec":  ae_rec["train_time_sec"],
        "mlp_inference_ms_per_1k": mlp_infer,
        "ae_inference_ms_per_1k":  ae_infer,
        "test_set_size": int(len(y_test)),
        "test_fraud_count": int(y_test.sum()),
    })

    # ---------- Plots ----------
    _plot_confusion(np.array(mlp_metrics["confusion_matrix"]),
                    "Confusion matrix — MLP (best)", COMPARE_DIR / "cm_mlp.png")
    _plot_confusion(np.array(ae_metrics["confusion_matrix"]),
                    "Confusion matrix — Autoencoder", COMPARE_DIR / "cm_ae.png")
    _plot_curves(y_test, mlp_scores, ae_scores, COMPARE_DIR / "roc_pr.png")
    _plot_efficiency([
        {"name": "MLP (best)",  "n_params": mlp_rec["n_params"],
         "train_time_sec": mlp_rec["train_time_sec"], "inference_ms_per_1k": mlp_infer},
        {"name": "Autoencoder", "n_params": ae_rec["n_params"],
         "train_time_sec": ae_rec["train_time_sec"], "inference_ms_per_1k": ae_infer},
    ], COMPARE_DIR / "efficiency.png")
    print(f"Saved comparison artifacts under {COMPARE_DIR}")


if __name__ == "__main__":
    main()
