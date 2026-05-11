"""Simple Flask web UI for the Credit Card Fraud Autoencoder.

Run:
    python app.py

Then open http://127.0.0.1:5000/ in your browser.

The page accepts a CSV file with the same columns the model was trained on
(Time, V1-V28, Amount). The "Class" column is optional and ignored if present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for

# TensorFlow is loaded lazily so the page can still render with a clear error
# if the artifacts are missing.
APP_DIR        = Path(__file__).parent
MODEL_PATH     = APP_DIR / "artifacts" / "autoencoder_fraud.keras"
THRESHOLD_PATH = APP_DIR / "artifacts" / "threshold.json"
ALLOWED_EXTS   = {"csv"}
MAX_PREVIEW    = 50  # rows shown back in the results table

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload cap
app.secret_key = "dev-key-change-me"


# Lazy singletons.
_model = None
_meta: dict | None = None


def get_model_and_meta():
    """Load the trained autoencoder and threshold metadata once, then cache."""
    global _model, _meta
    if _model is None:
        if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                f"Missing artifacts. Expected {MODEL_PATH} and {THRESHOLD_PATH}. "
                "Run the training notebook first."
            )
        # Import TF here so the app starts fast and surfaces a clear error if missing.
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH, compile=False)
        with open(THRESHOLD_PATH, "r") as f:
            _meta = json.load(f)
    return _model, _meta


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def run_inference(df: pd.DataFrame, feature_order: list[str], threshold: float):
    """Compute reconstruction errors and binary predictions for a DataFrame."""
    model, _ = get_model_and_meta()
    X = df[feature_order].values.astype("float32")
    X_hat = model.predict(X, batch_size=4096, verbose=0)
    errors = np.mean(np.square(X - X_hat), axis=1)
    preds = (errors > threshold).astype(int)
    return errors, preds


@app.route("/", methods=["GET"])
def index():
    try:
        _, meta = get_model_and_meta()
        artifacts_ok = True
        threshold = float(meta["threshold"])
        n_features = int(meta["n_features"])
        feature_order = meta["feature_order"]
        roc_auc = meta.get("test_metrics", {}).get("ranking", {}).get("roc_auc")
    except Exception as exc:
        artifacts_ok = False
        threshold = None
        n_features = None
        feature_order = []
        roc_auc = None
        flash(f"Could not load model artifacts: {exc}")

    return render_template(
        "index.html",
        artifacts_ok=artifacts_ok,
        threshold=threshold,
        n_features=n_features,
        feature_order=feature_order,
        roc_auc=roc_auc,
        result=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        _, meta = get_model_and_meta()
    except Exception as exc:
        flash(f"Model not loaded: {exc}")
        return redirect(url_for("index"))

    threshold     = float(meta["threshold"])
    feature_order = meta["feature_order"]
    n_features    = int(meta["n_features"])
    roc_auc       = meta.get("test_metrics", {}).get("ranking", {}).get("roc_auc")

    file = request.files.get("file")
    if file is None or file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only .csv files are accepted.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        flash(f"Failed to read CSV: {exc}")
        return redirect(url_for("index"))

    # 'Class' is optional in uploaded files; drop it if present.
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        flash(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        )
        return redirect(url_for("index"))

    if df.isnull().any().any():
        flash("Uploaded CSV contains NaN values. Please clean it first.")
        return redirect(url_for("index"))

    errors, preds = run_inference(df, feature_order, threshold)

    # Build a small preview table — keep just the most user-meaningful columns
    # so it doesn't blow up to 30+ columns on screen.
    preview_cols = ["Time", "Amount"] if "Time" in df.columns and "Amount" in df.columns else feature_order[:4]
    table_df = df[preview_cols].copy()
    table_df.insert(0, "#", np.arange(1, len(table_df) + 1))
    table_df["Recon Error"] = np.round(errors, 6)
    table_df["Prediction"]  = ["Fraud" if p == 1 else "Normal" for p in preds]

    # Surface frauds first so they're visible in the truncated preview.
    table_df = table_df.sort_values(by="Recon Error", ascending=False).reset_index(drop=True)
    preview = table_df.head(MAX_PREVIEW)

    result = {
        "filename":  file.filename,
        "total":     int(len(preds)),
        "normal":    int((preds == 0).sum()),
        "fraud":     int((preds == 1).sum()),
        "fraud_pct": float(100.0 * (preds == 1).mean()),
        "threshold": threshold,
        "max_error": float(errors.max()),
        "min_error": float(errors.min()),
        "mean_error": float(errors.mean()),
        "columns":   list(preview.columns),
        "rows":      preview.values.tolist(),
        "truncated": len(table_df) > MAX_PREVIEW,
        "preview_n": int(min(MAX_PREVIEW, len(table_df))),
    }

    return render_template(
        "index.html",
        artifacts_ok=True,
        threshold=threshold,
        n_features=n_features,
        feature_order=feature_order,
        roc_auc=roc_auc,
        result=result,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
