# Credit Card Fraud Detection — Final Report

**Course:** CS-419 Deep Learning — Spring 2026
**Section:** 13D / 13E (BSCS 2k23)
**Author:** Ahmed Raza
**Repository:** <https://github.com/AhmedRaza33/dl-project>

---

## 1. Introduction & Problem Statement

Credit-card fraud causes billions of dollars in losses every year and is
one of the canonical real-world applications of binary classification.
The problem is **extremely imbalanced** — fraudulent transactions are
typically <0.2 % of the traffic — which means naive metrics like accuracy
are meaningless (predicting "never fraud" already gets 99.83 % accuracy).

We frame this as a **binary classification** task: given a transaction
described by 30 numerical features, output a probability that it is
fraudulent. We then study how each deep-learning design choice (optimiser,
activation, weight initialisation, regularisation, normalisation, class
weighting) affects model quality, and compare a supervised MLP against an
unsupervised autoencoder anomaly detector on the **same** test split.

Concretely, this report covers the deliverables from the project brief:

| Deliverable | Section |
|---|---|
| A. Problem definition                | §1, §2 |
| B. Baseline model                    | §4.1 |
| C. Experimental deep-learning study  | §4.2 |
| D. Analysis (curves, CM, ablation, overfitting, efficiency) | §5 |
| E. Real-world reflection             | §7 |

---

## 2. Dataset Description

We use the **Kaggle Credit Card Fraud Detection dataset**: 284 807
transactions made by European cardholders in September 2013, with **492
fraud cases (0.172 %)**.

| Attribute | Value |
|---|---|
| Number of samples | 284 807 |
| Number of features | 30 (`Time`, `V1..V28`, `Amount`) |
| Target | `Class` ∈ {0, 1}; 1 = fraud |
| Positive rate | 0.172 % |

`V1..V28` are anonymised PCA components of the original raw features. We
scale `Time` and `Amount` to zero mean / unit variance in
`fraud_detection_preprocessing.ipynb` and persist the cleaned dataset as
`processed_creditcard.csv` (~158 MB — excluded from version control and
regenerable from the preprocessing notebook).

### Split strategy

A **stratified 70 / 15 / 15 train / validation / test split** with seed 42
preserves the 0.17 % fraud rate in every partition. The exact same split
is consumed by every variant in the MLP study **and** by the retrained
autoencoder, so every reported number in this report is directly
comparable.

```
train  : n=199 364   fraud=344 (0.173 %)
val    : n= 42 721   fraud= 74 (0.173 %)
test   : n= 42 722   fraud= 74 (0.173 %)
```

---

## 3. Methodology

Two model families are studied:

1. **MLP supervised classifier** (`fraud_detection_mlp.ipynb`).
   3 hidden layers (64 → 32 → 16), sigmoid head, binary cross-entropy
   loss, ~5 k parameters. Decision threshold tuned to argmax-F1 on the
   validation set.
2. **Deep autoencoder anomaly detector** (`fraud_detection_autoencoder.ipynb`).
   Symmetric 30→28→20→12→**7**→12→20→28→30 architecture trained on
   *normals only* to minimise reconstruction MSE. At inference time,
   high reconstruction error = anomaly = fraud. ~4 k parameters.

Both notebooks now use the **same stratified 70 / 15 / 15 split** defined
in `experiments/common.py`. The autoencoder still trains on the *normals
only* (its original unsupervised paradigm), but those normals are drawn
from the same training partition the MLP uses. The two models therefore
make their final predictions on byte-identical validation and test sets,
and the comparison is genuinely apples-to-apples.

### Why these models?

| Architecture | Suitability for tabular fraud detection |
|---|---|
| **MLP** | ✓ Natural fit — features have no spatial / temporal structure. |
| 1D CNN | ✗ Features are PCA-decorrelated; no local correlation for kernels to exploit. |
| RNN / LSTM | ✗ Each transaction is a single vector; no sequence to unroll. |
| **Autoencoder** | ✓ Excellent for unlabelled / novel-fraud regimes. |

We deliberately *do not* train a CNN or RNN because there is no
inductive bias they could exploit on this dataset — they would be a
strictly worse MLP. The MLP-vs-AE comparison is the meaningful one for
this task.

### Evaluation metrics

Accuracy is **not** reported as a headline number — predicting "always
normal" already scores 99.83 %. We use:

- **Precision, Recall, F1** at a tuned decision threshold.
- **ROC-AUC** — threshold-free ranking quality.
- **PR-AUC (Average Precision)** — the right ranking metric under
  extreme class imbalance.
- **Confusion matrix** (TP / FP / FN / TN).
- **Parameters** and **training time** for efficiency.

---

## 4. Experiments & Results

### 4.1 Baseline (Deliverable B)

A plain 3-layer MLP with SGD (lr=0.01, momentum=0.9), ReLU, no
regularisation, no class weighting, EarlyStopping (patience=5 on
validation PR-AUC).

| Metric | Value |
|---|---|
| Test precision | 0.806 |
| Test recall    | 0.784 |
| **Test F1**    | **0.795** |
| Test ROC-AUC   | 0.959 |
| Test PR-AUC    | 0.700 |
| Parameters     | 4 609 |
| Train time     | 8 s |

This is already strong — but it is the floor; everything below has to
beat it.

### 4.2 Experimental study (Deliverable C)

An *additive* sweep: each variant takes the previous configuration and
adds one technique. All variants use the same data split and the same
random seed (42), so differences are attributable to the change made.

| # | Variant | Adds | Params | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---:|---:|---:|---:|
| 1 | `1_baseline`      | SGD, ReLU, no extras | 4 609 | 0.795 | 0.9593 | 0.6998 |
| 2 | `2_adam`          | **Swap SGD → Adam** | 4 609 | **0.815** | **0.9791** | **0.8329** |
| 3 | `3_classweights`  | + Inverse-frequency class weights | 4 609 | 0.760 | 0.9750 | 0.6552 |
| 4 | `4_batchnorm`     | + BatchNormalization | 5 057 | 0.767 | 0.9651 | 0.6448 |
| 5 | `5_dropout_l2`    | + Dropout(0.3) + L2(1e-4) | 5 057 | 0.789 | 0.9577 | 0.6671 |
| 6 | `6_leakyrelu`     | Swap ReLU → LeakyReLU(0.1) | 5 057 | 0.776 | 0.9615 | 0.6600 |
| 7 | `7_best`          | + He init + ReduceLROnPlateau | 5 057 | 0.795 | 0.9701 | 0.6658 |

**Best variant (by validation PR-AUC): `2_adam`** — the single change
from SGD to Adam delivers the largest improvement, and adding more
techniques on top *hurts*. This is a meaningful finding for a low-feature,
high-sample-count tabular problem: the model is already low-variance,
so further regularisation removes signal.

The headline numbers for the production MLP are therefore:

| Test metric | MLP (`2_adam`) |
|---|---:|
| Precision  | **0.902** |
| Recall     | 0.743 |
| F1         | **0.815** |
| ROC-AUC    | **0.9791** |
| PR-AUC     | **0.8329** |
| Confusion matrix | TN=42 642, FP=6, FN=19, TP=55 |

### 4.3 Comparative analysis: MLP vs. Autoencoder

Both models evaluated on the same 42 722-sample test set with the same
F1-optimal threshold-selection rule on the same validation set.

| Model | Params | Train time | Inference (ms/1k) | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLP (best)**     | 4 609 |  7.9 s | 1.44 | **0.902** | 0.743 | **0.815** | **0.979** | **0.833** |
| Autoencoder        | 4 085 | 96.8 s | 1.92 | 0.413 | 0.676 | 0.513 | 0.940 | 0.468 |

The MLP wins decisively across **every** metric. The autoencoder still
ranks frauds reasonably well (ROC-AUC 0.94) but its precision is poor:
because it never saw a fraud during training, its reconstruction error
overlaps heavily between fraud and normal traffic, so the precision /
recall trade-off is much worse.

Plots produced by `experiments/compare_models.py`:

- `artifacts/comparison/roc_pr.png` — ROC and PR curves overlaid.
- `artifacts/comparison/cm_mlp.png` / `cm_ae.png` — confusion matrices.
- `artifacts/comparison/efficiency.png` — bar charts of params, training
  time, and inference latency.

---

## 5. Analysis (Deliverable D)

### 5.1 Training vs validation curves

Validation loss curves are plotted for every variant in §4 of
`fraud_detection_mlp.ipynb`. Two observations:

- **SGD → Adam is dramatic.** The SGD baseline takes a full 30 epochs to
  reach its plateau; Adam converges to a strictly better PR-AUC inside 25
  epochs.
- **Train / val gap stays small.** Even the fastest-converging variants
  (`3_classweights`, `4_batchnorm`) show parallel train / val curves —
  no sign of overfitting before EarlyStopping fires.

### 5.2 Confusion matrix (best MLP, test set)

```
                  Predicted
                  Normal     Fraud
        Normal    42 642        6
True
        Fraud         19       55
```

- **6 false positives** out of 42 648 legitimate transactions → real
  customers seeing a card block: 0.014 % false-positive rate.
- **19 false negatives** out of 74 frauds → 25.7 % of frauds slipped
  through. In production this is the lever we would tune: lower the
  threshold to trade some precision for more recall. The PR curve in the
  comparison plot quantifies exactly how much.

### 5.3 Ablation study

Subtractive ablation on the full-stack variant: start from the model
with every technique on, then remove **one** component at a time. The
difference reveals each technique's marginal contribution **on top of
everything else**.

| Removed | Test F1 | ΔF1 | Test ROC-AUC | ΔROC-AUC | Test PR-AUC | ΔPR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| *nothing* (full stack) | 0.7945 |  0.0000 | 0.9701 |  0.0000 | 0.6658 |  0.0000 |
| LeakyReLU              | 0.7389 | **−0.0556** | 0.9451 | −0.0250 | 0.6142 | −0.0516 |
| Class weights          | 0.7724 | −0.0221 | 0.8988 | **−0.0713** | 0.5903 | **−0.0755** |
| He init                | 0.7755 | −0.0190 | 0.9615 | −0.0086 | 0.6600 | −0.0058 |
| BatchNorm              | 0.7808 | −0.0137 | 0.9377 | −0.0324 | 0.6243 | −0.0415 |
| Dropout                | 0.8000 | +0.0055 | 0.9628 | −0.0073 | 0.6690 | +0.0032 |
| L2                     | 0.8027 | +0.0082 | 0.9699 | −0.0002 | 0.6628 | −0.0030 |

**Reading the ablation:**

- The single largest contributor is **LeakyReLU** (−0.056 F1 if
  removed). On a dataset with PCA-decorrelated, zero-centred features
  many activations land in the negative half-plane; ReLU's hard zero is
  costing us gradient.
- **Class weights** matter most for **ROC-AUC** (−0.07 if removed) —
  i.e. they reshape the score distribution rather than the eventual
  argmax.
- **Dropout(0.3)** and **L2(1e-4)** show *positive* ΔF1 if removed —
  they were *over-regularising* the model. With 199 k training examples
  per 5 k parameters, the model is naturally low-variance and needs less
  regularisation, not more.

### 5.4 Overfitting / underfitting

The full-stack model has **5 057 parameters** trained on **199 364
examples** — roughly **39 examples per parameter**. That is a very low
risk of overfitting, which the ablation confirms (more regularisation →
worse generalisation).

The opposite failure — **underfitting / undertraining** — is what
EarlyStopping protects us from. The class-weighted variants converge in
7–10 epochs because the heavily-weighted minority class blows up the
gradients early; once `val_pr_auc` stops improving for 5 epochs the
weights are restored to the best epoch.

### 5.5 Efficiency

| Model | Params | Training time | Inference latency |
|---|---:|---:|---:|
| MLP (best)     | 4 609 |  7.9 s | 1.44 ms / 1 000 samples |
| Autoencoder    | 4 085 | 96.8 s | 1.92 ms / 1 000 samples |

The MLP is 6× faster to train (fewer epochs needed to converge) and
slightly faster at inference. Both fit comfortably on commodity CPU.

---

## 6. Comparative Analysis Summary

> **Which deep learning techniques helped the most, and why?**

For this dataset, ranked by impact:

1. **Adam optimiser** — +0.13 PR-AUC over SGD (variant 1 → variant 2).
   The PCA-decorrelated features create a loss surface where adaptive
   per-parameter learning rates converge to a substantially better
   optimum than plain SGD-with-momentum.
2. **LeakyReLU + He initialisation** — the single largest hit in the
   subtractive ablation (−0.056 F1 if removed). On zero-mean features,
   negative-half-plane gradients matter.
3. **BatchNormalization** — modest but real contribution (~+0.03 PR-AUC
   in the ablation). Mostly stabilises training rather than improving
   peak performance.
4. **Class weighting** — primarily improves **ROC-AUC** (+0.07 in
   the ablation). It changes the model's probability calibration; once
   we tune the decision threshold the gain on F1 is smaller, but the
   underlying ranking is genuinely better.

**Techniques that did not help on this data:**

- **Dropout(0.3) + L2(1e-4)** — slightly *hurt* generalisation. With
  199 k examples per 5 k parameters the model has plenty of effective
  data; suppressing capacity removes signal.

**Cross-model:**

- The **supervised MLP beats the unsupervised autoencoder by ~0.30 F1
  and ~0.36 PR-AUC** on the same test set. The autoencoder retains a
  role as a drift sensor and as a fallback for novel fraud patterns
  unseen in the labels — but as the primary classifier the MLP is
  strictly better.

---

## 7. Real-World Reflection (Deliverable E)

### 7.1 Deployment challenges

The best MLP serialises to ~20 kB and predicts in <1.5 ms per 1 000
transactions on CPU — easy to deploy as a microservice. `app.py` in this
repository already serves the autoencoder behind a Flask UI; swapping in
the MLP is a one-line change. Realistic production concerns:

- **Latency budget.** Payment authorisation must complete within a
  few hundred milliseconds end-to-end, so the model must coexist with
  network, lookup, and feature-extraction cost. Our 1.5 µs/sample
  inference is well inside that budget.
- **Drift monitoring.** Fraud patterns are non-stationary. We need a
  drift detector — the autoencoder's reconstruction error on
  *legitimate* traffic is a natural one. When the average error on
  normals rises, retrain.
- **Versioning and rollback.** Models, thresholds, and the feature
  scaler need to be versioned together (they are coupled). A bad model
  push that doubles the false-positive rate is a customer-service
  emergency.

### 7.2 Ethical / bias considerations

The Kaggle dataset is anonymised and PCA-transformed, so we cannot
directly audit the model for bias against protected demographic groups.
In a real deployment we would need to:

- Measure false-positive rates per geography, age, and merchant
  category, not just in aggregate.
- Ensure customers blocked by the model have a **fast, free, and
  effective** way to dispute the decision.
- Treat the model output as one input to a human decision — not as a
  decision in itself — for any consequence beyond a single transaction
  decline.

### 7.3 Data limitations

- **PCA-anonymised features** mean we cannot inspect what the model is
  actually using. Two features `V14` and `V17` could be merchant ZIP or
  could be cardholder age; we cannot tell.
- **Single time window.** The Kaggle dataset covers 48 hours of one
  bank's traffic in 2013. Generalisation to other banks, geographies,
  and years is unverified.
- **Class imbalance.** With only 492 frauds in the entire dataset and
  74 in the test set, every confusion-matrix cell has high variance.
  Bootstrap confidence intervals on the F1 score should be reported in
  any production write-up.

---

## 8. Conclusion & Future Work

**Conclusions.**

- A small supervised MLP (~5 k parameters) trained with Adam reaches
  **F1 ≈ 0.81 / ROC-AUC ≈ 0.98 / PR-AUC ≈ 0.83** on a held-out 15 % test
  split — a strong baseline that beats the unsupervised autoencoder by a
  wide margin on the same data.
- For this dataset, the **optimiser choice (Adam vs SGD)** is the
  single largest design decision; classical regularisation (Dropout,
  L2) actively hurts because the data:parameter ratio is high.
- Class-imbalance handling matters more for ranking quality (ROC-AUC)
  than for thresholded F1.

**Future work.**

- **Ensembling.** Combine the MLP's probability with the autoencoder's
  reconstruction error in a logistic stacking layer; the autoencoder's
  ranking is complementary at the high-recall tail.
- **Focal loss** instead of BCE — directly down-weights easy examples,
  shown to help with extreme imbalance in image detection; worth a try
  here.
- **Gradient-boosted trees** (XGBoost / LightGBM) as a comparison
  baseline. On tabular data with this much imbalance, GBDTs are
  historically very competitive and we should know how much (or how
  little) deep learning is buying us beyond a strong classical baseline.
- **Calibration.** The MLP probabilities are uncalibrated. For
  thresholded decisions this is fine, but for downstream cost-sensitive
  decisions we should add Platt scaling or isotonic regression on the
  validation set.

---

## Appendix A — How to reproduce

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Regenerate the preprocessed dataset (only if processed_creditcard.csv
#    is missing; the file is excluded from git because it is 158 MB).
jupyter nbconvert --to notebook --execute fraud_detection_preprocessing.ipynb

# 3. Run the unsupervised autoencoder pipeline.
jupyter nbconvert --to notebook --execute fraud_detection_autoencoder.ipynb

# 4. Run the supervised MLP study end-to-end (also produces the head-to-head
#    comparison artifacts and the ablation).
python experiments/train_mlp_study.py
python experiments/compare_models.py
python experiments/ablation_study.py

# 5. Refresh the executed MLP notebook with the latest numbers.
python -m nbconvert --to notebook --execute fraud_detection_mlp.ipynb \
    --output fraud_detection_mlp.ipynb
```

## Appendix B — Repository map

```
DL Project/
├── REPORT.md                              <-- this file
├── PRESENTATION.md                        slide outline
├── README.md
├── requirements.txt
├── app.py                                 Flask UI for the autoencoder
├── fraud_detection_preprocessing.ipynb    EDA + scaling -> processed_creditcard.csv
├── fraud_detection_autoencoder.ipynb      Unsupervised AE pipeline
├── fraud_detection_mlp.ipynb              Supervised MLP study + comparison (this report's main notebook)
├── experiments/
│   ├── common.py                          Shared splits + metric helpers (consumed by both notebooks)
│   ├── train_mlp_study.py                 7-variant additive sweep
│   ├── compare_models.py                  Head-to-head on shared test set
│   ├── ablation_study.py                  Subtractive ablation
│   ├── build_notebook.py                  Regenerates fraud_detection_mlp.ipynb
│   ├── build_report_pdf.py                Regenerates Final_Report.pdf
│   └── build_presentation_pptx.py         Regenerates Final_Presentation.pptx
└── artifacts/
    ├── autoencoder_fraud.keras            Canonical AE (trained by the notebook on the shared split)
    ├── threshold.json                     AE threshold + metadata
    ├── history.json                       AE training history + time
    ├── mlp_study/                         All MLP variants + ablation
    └── comparison/                        Head-to-head plots and metrics
```
