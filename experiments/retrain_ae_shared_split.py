"""Retrain the autoencoder on the SAME train/val/test split used by the MLP.

The autoencoder in ``artifacts/autoencoder_fraud.keras`` was trained on a
different normals-only split inside ``fraud_detection_autoencoder.ipynb``, so
comparing it directly against the MLP on the MLP's test set would mix two
different evaluation regimes (and could even leak training samples into the
test set).

Here we:
    1. Take the SAME stratified train/val/test split that the MLP uses.
    2. Train an autoencoder on the train-set *normals only* (its original
       unsupervised setup).
    3. Use the reconstruction error on the validation set to pick an F1-optimal
       threshold — same threshold-selection rule the comparison script uses
       for the MLP.
    4. Save the model + threshold under
       ``artifacts/mlp_study/ae_shared.keras`` / ``ae_shared.json`` so the
       comparison script can evaluate both models under identical conditions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import (
    MLP_DIR, SEED, evaluate_scores, load_splits, save_json, set_global_seed,
)


def build_autoencoder(n_features: int):
    import tensorflow as tf
    from tensorflow.keras import Input, Model, regularizers
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

    reg = regularizers.l2(1e-5)
    inputs = Input(shape=(n_features,), name="features")
    x = inputs

    # Encoder
    for i, units in enumerate([28, 20, 12]):
        x = Dense(units, activation="relu", kernel_regularizer=reg,
                  name=f"enc_{i+1}")(x)
        x = BatchNormalization(name=f"bn_enc_{i+1}")(x)
        if i < 2:
            x = Dropout(0.1, name=f"drop_enc_{i+1}")(x)

    # Bottleneck
    x = Dense(7, activation="relu", name="bottleneck")(x)

    # Decoder
    for i, units in enumerate([12, 20, 28]):
        x = Dense(units, activation="relu", kernel_regularizer=reg,
                  name=f"dec_{i+1}")(x)
        x = BatchNormalization(name=f"bn_dec_{i+1}")(x)
        if i > 0:
            x = Dropout(0.1, name=f"drop_dec_{i+1}")(x)

    outputs = Dense(n_features, activation="linear", name="reconstruction")(x)
    model = Model(inputs, outputs, name="autoencoder_shared")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


def main():
    set_global_seed(SEED)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    splits = load_splits()
    X_train_normal = splits.X_train[splits.y_train == 0]
    X_val_normal   = splits.X_val[splits.y_val == 0]

    print(f"AE training on normals only: {X_train_normal.shape}  "
          f"(val_normal={X_val_normal.shape})")

    ae = build_autoencoder(splits.n_features)
    t0 = time.time()
    history = ae.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=50, batch_size=2048,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-6),
        ],
        verbose=0,
    )
    train_time = time.time() - t0
    print(f"AE trained in {train_time:.1f}s over {len(history.history['loss'])} epochs")

    def reconstruction_error(X):
        X_hat = ae.predict(X, batch_size=8192, verbose=0)
        return np.mean((X - X_hat) ** 2, axis=1)

    val_scores  = reconstruction_error(splits.X_val)
    test_scores = reconstruction_error(splits.X_test)

    val_metrics  = evaluate_scores(splits.y_val,  val_scores, threshold=None)
    test_metrics = evaluate_scores(splits.y_test, test_scores,
                                   threshold=val_metrics["threshold"])

    ae.save(MLP_DIR / "ae_shared.keras")
    save_json(MLP_DIR / "ae_shared.json", {
        "n_params": int(ae.count_params()),
        "train_time_sec": float(train_time),
        "epochs_run": int(len(history.history["loss"])),
        "val": val_metrics,
        "test": test_metrics,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    })
    np.save(MLP_DIR / "ae_shared_test_scores.npy", test_scores)

    print(f"  test  P={test_metrics['precision']:.3f}"
          f"  R={test_metrics['recall']:.3f}"
          f"  F1={test_metrics['f1']:.3f}"
          f"  ROC-AUC={test_metrics['roc_auc']:.4f}"
          f"  PR-AUC={test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
