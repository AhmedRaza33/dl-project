# Credit Card Fraud Detection — Autoencoder (Deep Learning Project)

An end-to-end deep-learning pipeline that detects fraudulent credit-card
transactions using an **unsupervised autoencoder**. The repo ships with the
data-preprocessing notebook, the model-training notebook, the trained model
artifacts, and a Flask web app for interactive inference.

## Project structure

```
DL Project/
├── app.py                                # Flask web app (inference UI)
├── requirements.txt                      # Python dependencies
├── fraud_detection_preprocessing.ipynb   # EDA + preprocessing notebook
├── fraud_detection_autoencoder.ipynb     # Autoencoder training notebook
├── sample_test.csv                       # Small sample for trying the app
├── artifacts/                            # Trained model + thresholds
│   ├── autoencoder_fraud.keras
│   ├── autoencoder_fraud.h5
│   ├── history.json
│   └── threshold.json
├── templates/index.html                  # Flask UI template
└── static/style.css                      # Stylesheet
```

> The full preprocessed dataset (`processed_creditcard.csv`, ~158 MB) is
> excluded from version control via `.gitignore` because it exceeds GitHub's
> 100 MB per-file limit. Regenerate it by running
> `fraud_detection_preprocessing.ipynb` on the original
> [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## How it works

1. **Preprocessing** — `fraud_detection_preprocessing.ipynb` scales features,
   handles class imbalance, and saves a clean dataset.
2. **Training** — `fraud_detection_autoencoder.ipynb` trains an autoencoder
   only on legitimate transactions. Reconstruction error is used as the
   anomaly score; a threshold is chosen from the validation set and stored in
   `artifacts/threshold.json`.
3. **Inference** — `app.py` loads the saved model and threshold and serves a
   small Flask UI that predicts whether a transaction is fraudulent.

## Getting started

```bash
# 1. Clone
git clone https://github.com/AhmedRaza33/dl-project.git
cd dl-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
python app.py
```

Then open <http://127.0.0.1:5000> in your browser and upload `sample_test.csv`
to try it out.

## Re-training from scratch

1. Download `creditcard.csv` from the Kaggle dataset linked above.
2. Run `fraud_detection_preprocessing.ipynb` end-to-end to produce
   `processed_creditcard.csv`.
3. Run `fraud_detection_autoencoder.ipynb` end-to-end to retrain the model
   and refresh the artifacts in `artifacts/`.

## Tech stack

- Python 3.10+
- TensorFlow / Keras
- scikit-learn, pandas, numpy
- Flask (for the inference UI)

## Author

**Ahmed Raza** — [@AhmedRaza33](https://github.com/AhmedRaza33)
