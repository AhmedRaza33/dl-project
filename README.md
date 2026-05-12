# Credit Card Fraud Detection — Deep Learning Study

> CS-419 Deep Learning final project. An experimental comparison of an
> **unsupervised deep autoencoder** and a **supervised MLP classifier**
> on the Kaggle Credit Card Fraud Detection dataset, with a Flask
> inference UI.

**Headline result (same 42 722-sample test set, same threshold-selection rule):**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **MLP (supervised, best variant)** | **0.902** | 0.743 | **0.815** | **0.979** | **0.833** |
| Autoencoder (unsupervised, retrained on shared split) | 0.329 | 0.716 | 0.451 | 0.938 | 0.492 |

### Deliverables

- **[`Final_Report.pdf`](Final_Report.pdf)** — formatted PDF report (7 pages, with embedded figures).
- **[`Final_Presentation.pptx`](Final_Presentation.pptx)** — 12-slide deck for the project presentation.
- [`REPORT.md`](REPORT.md) / [`PRESENTATION.md`](PRESENTATION.md) — the same content as Markdown sources.
- [`fraud_detection_mlp.ipynb`](fraud_detection_mlp.ipynb) — executed analysis notebook.

## Repository layout

```
DL Project/
├── Final_Report.pdf                       Polished PDF report (deliverable B)
├── Final_Presentation.pptx                12-slide PPTX deck (deliverable C)
├── REPORT.md                              Markdown source of the report
├── PRESENTATION.md                        Markdown source of the slides
├── README.md
├── requirements.txt
├── app.py                                 Flask UI for the autoencoder
├── fraud_detection_preprocessing.ipynb    EDA + scaling -> processed_creditcard.csv
├── fraud_detection_autoencoder.ipynb      Unsupervised AE pipeline
├── fraud_detection_mlp.ipynb              Supervised MLP study + comparison
├── experiments/                           CLI versions of the analyses
│   ├── common.py                          shared splits + metric helpers
│   ├── train_mlp_study.py                 7-variant additive sweep
│   ├── retrain_ae_shared_split.py         AE retrained on the MLP's train normals
│   ├── compare_models.py                  Head-to-head on shared test set
│   ├── ablation_study.py                  Subtractive ablation
│   ├── build_notebook.py                  Regenerates the MLP notebook
│   ├── build_report_pdf.py                Generates Final_Report.pdf
│   └── build_presentation_pptx.py         Generates Final_Presentation.pptx
└── artifacts/
    ├── autoencoder_fraud.keras            Original AE model
    ├── threshold.json                     Original AE threshold + metadata
    ├── mlp_study/                         all MLP variants, ablation, retrained AE
    └── comparison/                        head-to-head plots and metrics
```

> The full preprocessed dataset (`processed_creditcard.csv`, ~158 MB) is
> excluded from version control because it exceeds GitHub's 100 MB
> per-file limit. Regenerate it by running
> `fraud_detection_preprocessing.ipynb` on the
> [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## How it works

1. **Preprocessing** (`fraud_detection_preprocessing.ipynb`) — scales
   `Time` and `Amount`, persists the cleaned dataset.
2. **Autoencoder** (`fraud_detection_autoencoder.ipynb`) — symmetric
   30→28→20→12→**7**→12→20→28→30 AE trained on legitimate transactions
   only; reconstruction error becomes the anomaly score.
3. **MLP study** (`fraud_detection_mlp.ipynb` + `experiments/`) —
   supervised binary classifier with an additive sweep across
   optimisers, activations, weight init, BatchNorm, Dropout, L2, and
   class weighting, plus a subtractive ablation and a head-to-head
   comparison against the autoencoder on the same test set.
4. **Inference UI** (`app.py`) — Flask app that serves the autoencoder
   behind a small web form.

## Reproducing the experiments

```bash
# 0. Environment
pip install -r requirements.txt

# 1. Regenerate the preprocessed dataset (only if missing — 158 MB)
python -m nbconvert --to notebook --execute fraud_detection_preprocessing.ipynb \
    --output fraud_detection_preprocessing.ipynb

# 2. Train the unsupervised autoencoder (its own pipeline)
python -m nbconvert --to notebook --execute fraud_detection_autoencoder.ipynb \
    --output fraud_detection_autoencoder.ipynb

# 3. Run the MLP study end-to-end (7 variants -> ~75 seconds on CPU)
python experiments/train_mlp_study.py

# 4. Retrain the AE on the *same* train split and compare head-to-head
python experiments/retrain_ae_shared_split.py
python experiments/compare_models.py

# 5. Subtractive ablation on the full-stack MLP
python experiments/ablation_study.py

# 6. Refresh the executed MLP notebook with the latest numbers
python experiments/build_notebook.py
python -m nbconvert --to notebook --execute fraud_detection_mlp.ipynb \
    --output fraud_detection_mlp.ipynb

# 7. Rebuild the PDF report and the PPTX presentation from the artifacts
python experiments/build_report_pdf.py
python experiments/build_presentation_pptx.py

# 8. Run the inference UI
python app.py    # then open http://127.0.0.1:5000
```

## Tech stack

- Python 3.10+, TensorFlow / Keras
- scikit-learn, pandas, numpy, matplotlib
- Flask (for the inference UI)

## Author

**Ahmed Raza** — [@AhmedRaza33](https://github.com/AhmedRaza33)
