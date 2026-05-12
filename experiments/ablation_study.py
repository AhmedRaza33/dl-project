"""Ablation study: remove ONE technique at a time from the full-stack variant.

The 7-variant sweep in ``train_mlp_study.py`` is *additive* — it shows what
each technique adds. This script is *subtractive* — it starts from the full
stack and removes one component to measure how much it actually contributes
on its own. The result is a clean table that answers:

    "Of {class weights, BatchNorm, Dropout, L2, LeakyReLU, He init},
     which design choice mattered most?"

Outputs:
    artifacts/mlp_study/ablation_results.csv
    artifacts/mlp_study/ablation_<name>.json
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from common import MLP_DIR, SEED, load_splits, save_json, set_global_seed
from train_mlp_study import Variant, train_variant


FULL = Variant(
    name="full_stack",
    optimizer="adam", learning_rate=1e-3,
    activation="leaky_relu",
    initializer="he_normal",
    use_class_weight=True, use_batchnorm=True,
    dropout=0.3, l2=1e-4,
    use_reduce_lr=True,
    description="Full stack (all techniques on)",
)

ABLATIONS = [
    FULL,
    replace(FULL, name="ablate_classweights", use_class_weight=False,
            description="Full stack minus class weights"),
    replace(FULL, name="ablate_batchnorm",    use_batchnorm=False,
            description="Full stack minus BatchNorm"),
    replace(FULL, name="ablate_dropout",      dropout=0.0,
            description="Full stack minus Dropout"),
    replace(FULL, name="ablate_l2",           l2=0.0,
            description="Full stack minus L2"),
    replace(FULL, name="ablate_leakyrelu",    activation="relu",
            description="Full stack minus LeakyReLU (use ReLU)"),
    replace(FULL, name="ablate_he_init",      initializer="glorot_uniform",
            description="Full stack minus He init (use Glorot)"),
]


def main():
    set_global_seed(SEED)
    splits = load_splits()

    rows = []
    for cfg in ABLATIONS:
        print(f"\n=== {cfg.name}: {cfg.description} ===")
        rec = train_variant(cfg, splits)
        t = rec["test"]
        v = rec["val"]
        print(
            f"  params={rec['n_params']:>6}  epochs={rec['epochs_run']:>2}"
            f"  time={rec['train_time_sec']:>5.1f}s"
            f"  | test P={t['precision']:.3f} R={t['recall']:.3f}"
            f" F1={t['f1']:.3f} ROC-AUC={t['roc_auc']:.4f} PR-AUC={t['pr_auc']:.4f}"
        )
        rows.append({
            "variant": cfg.name,
            "description": cfg.description,
            "epochs": rec["epochs_run"],
            "train_time_sec": round(rec["train_time_sec"], 2),
            "test_precision": round(t["precision"], 4),
            "test_recall":    round(t["recall"],    4),
            "test_f1":        round(t["f1"],        4),
            "test_roc_auc":   round(t["roc_auc"],   4),
            "test_pr_auc":    round(t["pr_auc"],    4),
            "val_pr_auc":     round(v["pr_auc"],    4),
        })

    df = pd.DataFrame(rows)

    # delta-vs-full columns for easy reading
    full_row = df.iloc[0]
    for col in ["test_f1", "test_roc_auc", "test_pr_auc"]:
        df[f"delta_{col}"] = (df[col] - full_row[col]).round(4)

    df.to_csv(MLP_DIR / "ablation_results.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved to {MLP_DIR / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
