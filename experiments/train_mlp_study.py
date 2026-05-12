"""Train every MLP variant in the experimental study.

Each variant *adds one technique* on top of the previous one so the marginal
contribution of every design choice is visible:

    1. baseline      — plain MLP, SGD, ReLU, no regularisation, no class weights
    2. +adam         — swap SGD for Adam
    3. +classweights — handle the 0.17% fraud imbalance with inverse-freq weights
    4. +batchnorm    — BatchNormalization after every hidden Dense
    5. +dropout_l2   — add Dropout(0.3) + L2(1e-4)
    6. +leakyrelu    — swap ReLU for LeakyReLU(0.1)
    7. best          — all of the above + He initialisation + ReduceLROnPlateau

The script prints a metrics table to stdout and persists per-variant artifacts
(history, threshold, confusion matrix, metrics JSON) plus a combined
``metrics_summary.csv`` and the saved best model under
``artifacts/mlp_study/``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common import (
    MLP_DIR, SEED, Splits, class_weight_dict, evaluate_scores,
    load_splits, save_json, set_global_seed,
)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    optimizer: str = "sgd"            # 'sgd' | 'adam' | 'rmsprop'
    learning_rate: float = 0.01
    activation: str = "relu"          # 'relu' | 'leaky_relu' | 'elu'
    initializer: str = "glorot_uniform"
    hidden_units: tuple[int, ...] = (64, 32, 16)
    use_batchnorm: bool = False
    dropout: float = 0.0
    l2: float = 0.0
    use_class_weight: bool = False
    use_reduce_lr: bool = False
    description: str = ""


VARIANTS: list[Variant] = [
    Variant(
        name="1_baseline",
        description="Plain MLP, SGD, ReLU, no tricks",
    ),
    Variant(
        name="2_adam",
        optimizer="adam", learning_rate=1e-3,
        description="Swap SGD -> Adam",
    ),
    Variant(
        name="3_classweights",
        optimizer="adam", learning_rate=1e-3,
        use_class_weight=True,
        description="Adam + inverse-frequency class weights",
    ),
    Variant(
        name="4_batchnorm",
        optimizer="adam", learning_rate=1e-3,
        use_class_weight=True, use_batchnorm=True,
        description="+ BatchNormalization after each hidden layer",
    ),
    Variant(
        name="5_dropout_l2",
        optimizer="adam", learning_rate=1e-3,
        use_class_weight=True, use_batchnorm=True,
        dropout=0.3, l2=1e-4,
        description="+ Dropout(0.3) + L2(1e-4) regularisation",
    ),
    Variant(
        name="6_leakyrelu",
        optimizer="adam", learning_rate=1e-3,
        activation="leaky_relu",
        use_class_weight=True, use_batchnorm=True,
        dropout=0.3, l2=1e-4,
        description="Swap ReLU -> LeakyReLU(0.1)",
    ),
    Variant(
        name="7_best",
        optimizer="adam", learning_rate=1e-3,
        activation="leaky_relu",
        initializer="he_normal",
        use_class_weight=True, use_batchnorm=True,
        dropout=0.3, l2=1e-4,
        use_reduce_lr=True,
        description="Best config: He init + LR scheduler on top of everything",
    ),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_mlp(n_features: int, cfg: Variant):
    """Build the Keras MLP described by ``cfg``."""
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers, regularizers

    reg = regularizers.l2(cfg.l2) if cfg.l2 > 0 else None
    init = cfg.initializer

    def _act(x, name):
        if cfg.activation == "relu":
            return layers.ReLU(name=name)(x)
        if cfg.activation == "leaky_relu":
            return layers.LeakyReLU(negative_slope=0.1, name=name)(x)
        if cfg.activation == "elu":
            return layers.ELU(name=name)(x)
        raise ValueError(f"Unknown activation: {cfg.activation}")

    inputs = Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(cfg.hidden_units):
        x = layers.Dense(
            units, kernel_initializer=init, kernel_regularizer=reg,
            name=f"dense_{i+1}",
        )(x)
        if cfg.use_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = _act(x, name=f"act_{i+1}")
        if cfg.dropout > 0:
            x = layers.Dropout(cfg.dropout, name=f"drop_{i+1}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)
    model = Model(inputs, outputs, name=f"mlp_{cfg.name}")

    if cfg.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Training & evaluation per variant
# ---------------------------------------------------------------------------
def train_variant(
    cfg: Variant,
    splits: Splits,
    epochs: int = 30,
    batch_size: int = 2048,
    out_dir: Path = MLP_DIR,
    verbose: int = 0,
) -> dict:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    set_global_seed(SEED)
    model = build_mlp(splits.n_features, cfg)

    callbacks = [
        EarlyStopping(monitor="val_pr_auc", mode="max", patience=5,
                      restore_best_weights=True, verbose=verbose),
    ]
    if cfg.use_reduce_lr:
        callbacks.append(ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5, patience=3,
            min_lr=1e-6, verbose=verbose,
        ))

    cw = class_weight_dict(splits.y_train) if cfg.use_class_weight else None

    t0 = time.time()
    history = model.fit(
        splits.X_train, splits.y_train,
        validation_data=(splits.X_val, splits.y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, class_weight=cw, verbose=verbose,
    )
    train_time = time.time() - t0

    # Tune decision threshold on validation set (max F1), then evaluate on test.
    val_scores  = model.predict(splits.X_val,  batch_size=8192, verbose=0).ravel()
    test_scores = model.predict(splits.X_test, batch_size=8192, verbose=0).ravel()

    val_metrics  = evaluate_scores(splits.y_val,  val_scores,  threshold=None)
    test_metrics = evaluate_scores(splits.y_test, test_scores, threshold=val_metrics["threshold"])

    record = {
        "variant": cfg.name,
        "description": cfg.description,
        "config": asdict(cfg),
        "n_params": int(model.count_params()),
        "train_time_sec": float(train_time),
        "epochs_run": int(len(history.history["loss"])),
        "val": val_metrics,
        "test": test_metrics,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }

    save_json(out_dir / f"{cfg.name}.json", record)
    np.save(out_dir / f"{cfg.name}_test_scores.npy", test_scores)
    model.save(out_dir / f"mlp_{cfg.name}.keras")
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    set_global_seed(SEED)
    splits = load_splits()
    print("Data splits:")
    print(splits.describe())

    rows = []
    for cfg in VARIANTS:
        print(f"\n=== Training {cfg.name}: {cfg.description} ===")
        rec = train_variant(cfg, splits)
        t = rec["test"]
        v = rec["val"]
        print(
            f"  params={rec['n_params']:>6}  epochs={rec['epochs_run']:>2}"
            f"  time={rec['train_time_sec']:>5.1f}s"
            f"  | test  P={t['precision']:.3f} R={t['recall']:.3f}"
            f" F1={t['f1']:.3f}  ROC-AUC={t['roc_auc']:.4f}  PR-AUC={t['pr_auc']:.4f}"
        )
        rows.append({
            "variant": cfg.name,
            "description": cfg.description,
            "params": rec["n_params"],
            "epochs": rec["epochs_run"],
            "train_time_sec": round(rec["train_time_sec"], 2),
            "val_threshold":  round(v["threshold"], 6),
            "test_precision": round(t["precision"], 4),
            "test_recall":    round(t["recall"],    4),
            "test_f1":        round(t["f1"],        4),
            "test_roc_auc":   round(t["roc_auc"],   4),
            "test_pr_auc":    round(t["pr_auc"],    4),
        })

    summary = pd.DataFrame(rows)
    summary_path = MLP_DIR / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print(summary.to_string(index=False))

    # Promote the best variant (by validation PR-AUC) to ``mlp_best.keras`` so
    # the comparison script picks up the truly best classifier, not whichever
    # one happens to be last in the configuration list.
    import shutil
    val_pr_aucs = {}
    for cfg in VARIANTS:
        rec = json.loads((MLP_DIR / f"{cfg.name}.json").read_text())
        val_pr_aucs[cfg.name] = rec["val"]["pr_auc"]
    best_name = max(val_pr_aucs, key=val_pr_aucs.get)
    shutil.copyfile(MLP_DIR / f"mlp_{best_name}.keras", MLP_DIR / "mlp_best.keras")
    shutil.copyfile(MLP_DIR / f"{best_name}.json",       MLP_DIR / "7_best.json")
    (MLP_DIR / "best_variant.txt").write_text(best_name)
    print(f"\nBest variant by validation PR-AUC: {best_name}  (PR-AUC={val_pr_aucs[best_name]:.4f})")
    print(f"  -> copied to mlp_best.keras (consumed by experiments/compare_models.py)")

    save_json(MLP_DIR / "splits_meta.json", {
        "feature_names": splits.feature_names,
        "n_features": splits.n_features,
        "shapes": {
            "train": list(splits.X_train.shape),
            "val":   list(splits.X_val.shape),
            "test":  list(splits.X_test.shape),
        },
        "fraud_counts": {
            "train": int(splits.y_train.sum()),
            "val":   int(splits.y_val.sum()),
            "test":  int(splits.y_test.sum()),
        },
        "seed": SEED,
    })


if __name__ == "__main__":
    main()
