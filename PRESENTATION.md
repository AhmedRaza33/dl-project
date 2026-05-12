# Final Presentation Outline

CS-419 Deep Learning — Credit Card Fraud Detection
**Author:** Ahmed Raza
**Repository:** <https://github.com/AhmedRaza33/dl-project>

A 10-minute talk, ~12 slides. Speaker notes are bullet points; flesh them
out into spoken sentences during practice.

---

## Slide 1 — Title

- Credit Card Fraud Detection: an experimental DL study
- MLP vs. Autoencoder on the Kaggle dataset
- Course / section / name / date
- Visual: confusion-matrix preview as background.

## Slide 2 — Why this problem?

- Real losses, real consequences for customers when we get it wrong.
- 0.17 % positive rate → accuracy is a useless metric (99.83 % from the
  trivial "always normal" baseline).
- Two competing philosophies worth comparing:
  - Supervised classifier (labels available).
  - Unsupervised anomaly detector (works without labels).

## Slide 3 — Dataset at a glance

- 284 807 European card transactions, September 2013.
- 30 features: `Time`, `V1..V28` (PCA-decorrelated), `Amount`.
- 492 frauds (0.172 %).
- Stratified 70 / 15 / 15 split with seed 42 — reused by **both** notebooks.
- Visual: bar chart of class counts (log scale).

## Slide 4 — Model A: Deep autoencoder (unsupervised)

- Architecture: 30→28→20→12→**7**→12→20→28→30 (symmetric).
- Trained on **normals only** to minimise reconstruction MSE.
- Anomaly score = reconstruction error → threshold via F1-max on val.
- Visual: encoder/decoder diagram + reconstruction error histogram by class.

## Slide 5 — Model B: MLP (supervised)

- Architecture: 30 → 64 → 32 → 16 → 1 (sigmoid).
- Binary cross-entropy loss, ~5 k parameters.
- Decision threshold tuned by argmax F1 on validation.
- Visual: block diagram of the network.

## Slide 6 — Experimental study: 7 variants

- Additive sweep: each variant adds **one** technique.
- Table on slide:
  - 1 baseline (SGD, ReLU) → F1 0.795
  - 2 +Adam              → **F1 0.815 / PR-AUC 0.833** ★ best
  - 3 +class weights     → F1 0.760
  - 4 +BatchNorm         → F1 0.767
  - 5 +Dropout + L2      → F1 0.789
  - 6 +LeakyReLU         → F1 0.776
  - 7 +He init + LR sched → F1 0.795
- Visual: bar chart of test F1 per variant, baseline highlighted.

## Slide 7 — Key insight: simpler is better here

- Best variant is `2_adam` — *plain* Adam, no extras.
- Why? **199 364 examples / 5 057 params ≈ 39 :1** — high data:parameter
  ratio, so the model is already low-variance.
- Adding Dropout(0.3) + L2 → over-regularises → loses signal.
- The lesson: don't reach for regularisation reflexively; let the data
  : parameter ratio guide you.
- Visual: validation PR-AUC curves overlaid for all 7 variants.

## Slide 8 — Ablation (subtractive)

- Start from full stack, remove **one** component.
- Biggest drops:
  - Remove **LeakyReLU**     → −0.056 F1 (largest hit)
  - Remove **class weights** → −0.071 ROC-AUC (largest ROC hit)
- *Negative* drops (techniques that were hurting):
  - Remove Dropout → +0.006 F1
  - Remove L2      → +0.008 F1
- Visual: bar chart of ΔF1 by component (red = bad to remove, green = good to remove).

## Slide 9 — Head-to-head: MLP vs Autoencoder

- Both evaluated on the same 42 722-sample test set.
- Same threshold-selection rule (argmax F1 on shared val).

| Metric | MLP | Autoencoder |
|---|---:|---:|
| Precision | **0.902** | 0.413 |
| Recall    | 0.743 | 0.676 |
| F1        | **0.815** | 0.513 |
| ROC-AUC   | **0.979** | 0.940 |
| PR-AUC    | **0.833** | 0.468 |
| Params    | 4 609 | 4 085 |
| Train time | 7.9 s | 96.8 s |

- Visual: side-by-side confusion matrices.

## Slide 10 — Why the AE is still useful

- The MLP needs labels; the AE doesn't.
- AE is a natural **drift detector**: rising reconstruction error on
  legitimate traffic = the data distribution has shifted = retrain.
- AE could catch **novel fraud types** the labelled training set never
  saw.
- → Production answer is *both*: MLP as primary classifier, AE as
  guardrail.

## Slide 11 — Biggest challenge

- Class imbalance (0.17 %) made every shortcut bite back:
  - Accuracy meaningless → forced us to PR-AUC.
  - Class weights changed probability calibration → had to re-tune
    thresholds for every variant.
  - 74 fraud cases in test set → high variance in confusion matrix.
- Making the comparison **fair**: realised partway through that the
  original AE notebook used a different split than the MLP notebook, so
  we unified both notebooks to use the **same** stratified 70/15/15 split
  defined in `experiments/common.py`.

## Slide 12 — What I would improve with more time

- Train a **GBDT baseline** (XGBoost/LightGBM) — classical tabular
  champion. Need to know how much DL is actually buying us.
- Try **focal loss** instead of BCE.
- **Ensemble** the MLP probability with the AE reconstruction error.
- **Probability calibration** (Platt scaling / isotonic).
- Bootstrap **confidence intervals** on F1 — 74 positive samples is
  small.

## Slide 13 (backup) — Live demo

- Run `app.py`, upload `sample_test.csv`, show the UI flagging fraud
  rows.
- If time permits, walk through the notebook's confusion matrix.

---

## Speaker tips

- Keep slides 6–9 anchored to **numbers** — the audience can compare
  fast.
- For slide 7, lean into the counter-intuitive finding: more
  regularisation hurt. Examiners reward demonstrated understanding of
  *when* a technique applies.
- For slide 11, frame the AE-retrain story as **methodological
  honesty** — caught a leakage risk and fixed it.
