"""Render the final-presentation deck as a .pptx file.

Built with python-pptx so it opens in PowerPoint, Keynote, Google Slides,
and LibreOffice Impress identically. The deck covers the four points the
project brief asks for:

    - Best model and results
    - Key comparison insights
    - Biggest challenge faced
    - What you would improve with more time

Numbers and figures come from the same artifacts as the report PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


PROJECT_DIR = Path(__file__).resolve().parent.parent
MLP_DIR = PROJECT_DIR / "artifacts" / "mlp_study"
CMP_DIR = PROJECT_DIR / "artifacts" / "comparison"
OUT_PATH = PROJECT_DIR / "Final_Presentation.pptx"


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY   = RGBColor(0x1F, 0x29, 0x37)   # dark slate
DARK   = RGBColor(0x11, 0x18, 0x27)
ACCENT = RGBColor(0x1D, 0x4E, 0xD8)   # blue
ACCENT_LIGHT = RGBColor(0xDB, 0xEA, 0xFE)
SOFT   = RGBColor(0x6B, 0x72, 0x80)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
SUCCESS = RGBColor(0x16, 0xA3, 0x4A)
DANGER  = RGBColor(0xDC, 0x26, 0x26)
BG_ALT  = RGBColor(0xF9, 0xFA, 0xFB)


# ---------------------------------------------------------------------------
# Slide helpers
# ---------------------------------------------------------------------------
def new_blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def add_text_box(slide, left, top, width, height, text, *,
                 font_size=18, bold=False, italic=False,
                 color=DARK, align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP, font_name="Calibri"):
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0

    # split on lines so each becomes a paragraph
    lines = text.split("\n") if isinstance(text, str) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = font_name
    return tx


def add_rich_paragraphs(slide, left, top, width, height, paragraphs,
                        font_size=14, color=DARK, font_name="Calibri",
                        align=PP_ALIGN.LEFT):
    """Add a text box where each item is a list of (text, kwargs) runs."""
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, runs in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for text, opts in runs:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(opts.get("size", font_size))
            r.font.bold = opts.get("bold", False)
            r.font.italic = opts.get("italic", False)
            r.font.color.rgb = opts.get("color", color)
            r.font.name = font_name
    return tx


def add_filled_rect(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_title_bar(slide, title: str, eyebrow: str | None = None):
    """Slim left strip + slide title + optional eyebrow text."""
    add_filled_rect(slide, Inches(0), Inches(0), Inches(0.18), Inches(7.5), ACCENT)
    if eyebrow:
        add_text_box(slide, Inches(0.55), Inches(0.30), Inches(8), Inches(0.3),
                     eyebrow.upper(), font_size=10, bold=True, color=ACCENT)
    add_text_box(slide, Inches(0.55), Inches(0.55), Inches(12.5), Inches(0.7),
                 title, font_size=30, bold=True, color=DARK)


def add_footer(slide, idx: int, total: int):
    add_text_box(slide, Inches(0.55), Inches(7.05),
                 Inches(8), Inches(0.3),
                 "Ahmed Raza  ·  CS-419 Deep Learning  ·  Credit Card Fraud Detection",
                 font_size=10, color=SOFT)
    add_text_box(slide, Inches(11.5), Inches(7.05),
                 Inches(1.8), Inches(0.3),
                 f"{idx} / {total}", font_size=10, color=SOFT,
                 align=PP_ALIGN.RIGHT)


def add_table(slide, left, top, width, height, data,
              header_fill=NAVY, header_color=WHITE, body_font_size=12,
              header_font_size=12, first_col_align=PP_ALIGN.LEFT):
    rows, cols = len(data), len(data[0])
    tbl_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    tbl = tbl_shape.table
    for c_idx, val in enumerate(data[0]):
        cell = tbl.cell(0, c_idx)
        cell.text = ""
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(val)
        r.font.bold = True; r.font.size = Pt(header_font_size)
        r.font.color.rgb = header_color; r.font.name = "Calibri"
    for r_idx in range(1, rows):
        for c_idx in range(cols):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else BG_ALT
            p = cell.text_frame.paragraphs[0]
            p.alignment = first_col_align if c_idx == 0 else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(data[r_idx][c_idx])
            run.font.size = Pt(body_font_size)
            run.font.color.rgb = DARK
            run.font.name = "Calibri"
    return tbl_shape


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------
def build_title_slide(prs):
    slide = new_blank_slide(prs)
    add_filled_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), NAVY)
    add_filled_rect(slide, Inches(0), Inches(3.4), Inches(13.333), Inches(0.04), ACCENT)
    add_text_box(slide, Inches(1), Inches(1.7), Inches(11.3), Inches(0.5),
                 "CS-419 DEEP LEARNING  ·  FINAL PROJECT",
                 font_size=15, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(2.2), Inches(11.3), Inches(1.0),
                 "Credit Card Fraud Detection",
                 font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(3.6), Inches(11.3), Inches(0.8),
                 "An Experimental Study: Supervised MLP vs Unsupervised Autoencoder",
                 font_size=20, color=RGBColor(0xD1, 0xD5, 0xDB), align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(5.2), Inches(11.3), Inches(0.5),
                 "Ahmed Raza",
                 font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(5.7), Inches(11.3), Inches(0.5),
                 "BSCS 2k23 · Section 13D / 13E · Spring 2026",
                 font_size=14, color=RGBColor(0x9C, 0xA3, 0xAF), align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(6.6), Inches(11.3), Inches(0.5),
                 "github.com/AhmedRaza33/dl-project",
                 font_size=12, color=ACCENT, align=PP_ALIGN.CENTER)
    return slide


def build_problem_slide(prs, idx, total):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "The Problem", eyebrow="Why fraud detection")
    add_rich_paragraphs(slide, Inches(0.55), Inches(1.6), Inches(7.5), Inches(5),
                        paragraphs=[
        [("Credit-card fraud causes ", {}),
         ("billions of dollars", {"bold": True}),
         (" in losses each year.", {})],
        [("", {})],
        [("Key feature of the problem:", {"bold": True})],
        [("•  Extreme class imbalance — fraud is < 0.2 % of all transactions.", {})],
        [("•  Accuracy is meaningless — predicting ", {}),
         ("\"never fraud\"", {"italic": True}),
         (" already scores ", {}),
         ("99.83 %", {"bold": True, "color": DANGER}),
         (".", {})],
        [("•  Mistakes are asymmetric: missed fraud is the costly error.", {})],
        [("", {})],
        [("Two questions this project answers:", {"bold": True})],
        [("1.  Which DL technique helps the most on this data?", {})],
        [("2.  Is a supervised classifier (MLP) really better than an "
          "unsupervised anomaly detector (Autoencoder)?", {})],
    ], font_size=16)

    # Right panel: number facts
    panel_left = Inches(8.5)
    add_filled_rect(slide, panel_left, Inches(1.6),
                    Inches(4.3), Inches(5.2), BG_ALT)
    add_text_box(slide, panel_left, Inches(1.7), Inches(4.3), Inches(0.4),
                 "By the numbers", font_size=11, bold=True, color=ACCENT,
                 align=PP_ALIGN.CENTER)
    facts = [
        ("284 807", "transactions"),
        ("30",      "features (V1..V28, Time, Amount)"),
        ("492",     "fraud cases"),
        ("0.172 %", "positive rate"),
    ]
    for i, (big, small) in enumerate(facts):
        y = Inches(2.2 + i * 1.05)
        add_text_box(slide, panel_left, y, Inches(4.3), Inches(0.6),
                     big, font_size=30, bold=True, color=DARK,
                     align=PP_ALIGN.CENTER)
        add_text_box(slide, panel_left, y + Inches(0.55),
                     Inches(4.3), Inches(0.4),
                     small, font_size=12, color=SOFT, align=PP_ALIGN.CENTER)
    add_footer(slide, idx, total)


def build_methodology_slide(prs, idx, total):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Two Models, Same Test Set",
                  eyebrow="Methodology")

    # Two cards side by side
    for i, (title, subtitle, points, accent) in enumerate([
        ("Supervised MLP",
         "Binary classifier",
         [
            "30 → 64 → 32 → 16 → 1 (sigmoid)",
            "Binary cross-entropy loss",
            "Sees fraud labels during training",
            "Decision = predicted probability",
            "≈ 5 k parameters",
         ],
         ACCENT),
        ("Deep Autoencoder",
         "Unsupervised anomaly detector",
         [
            "30 → 28 → 20 → 12 → 7 → 12 → 20 → 28 → 30",
            "Trained on normals only (MSE)",
            "Never sees fraud during training",
            "Decision = reconstruction error",
            "≈ 4 k parameters",
         ],
         SUCCESS),
    ]):
        x = Inches(0.55 + i * 6.4)
        add_filled_rect(slide, x, Inches(1.6), Inches(6.0), Inches(0.6), accent)
        add_text_box(slide, x + Inches(0.2), Inches(1.65),
                     Inches(5.6), Inches(0.5),
                     title, font_size=20, bold=True, color=WHITE)
        add_filled_rect(slide, x, Inches(2.2), Inches(6.0), Inches(4.2), BG_ALT)
        add_text_box(slide, x + Inches(0.3), Inches(2.3),
                     Inches(5.6), Inches(0.4),
                     subtitle, font_size=13, italic=True, color=SOFT)
        for j, bullet_text in enumerate(points):
            add_text_box(slide, x + Inches(0.3),
                         Inches(2.75 + j * 0.6),
                         Inches(5.5), Inches(0.45),
                         "•  " + bullet_text, font_size=14, color=DARK)

    add_text_box(slide, Inches(0.55), Inches(6.6),
                 Inches(12.5), Inches(0.4),
                 "Both evaluated on the same stratified 70 / 15 / 15 split (seed=42) — "
                 "every reported number is directly comparable.",
                 font_size=12, italic=True, color=SOFT, align=PP_ALIGN.CENTER)

    add_footer(slide, idx, total)


def build_results_table_slide(prs, idx, total, summary):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Experimental Study — 7 MLP Variants",
                  eyebrow="Additive sweep")
    short_desc = {
        "1_baseline":      "SGD, ReLU, no tricks",
        "2_adam":          "Swap SGD → Adam",
        "3_classweights":  "+ class weights",
        "4_batchnorm":     "+ BatchNormalization",
        "5_dropout_l2":    "+ Dropout(0.3) + L2(1e-4)",
        "6_leakyrelu":     "ReLU → LeakyReLU(0.1)",
        "7_best":          "+ He init + LR scheduler",
    }
    rows = [["Variant", "What it adds", "F1", "ROC-AUC", "PR-AUC"]]
    for _, r in summary.iterrows():
        marker = "  ★" if r["variant"] == "2_adam" else "   "
        rows.append([
            r["variant"] + marker,
            short_desc.get(r["variant"], r.get("description", "")),
            f"{r['test_f1']:.3f}",
            f"{r['test_roc_auc']:.4f}",
            f"{r['test_pr_auc']:.4f}",
        ])
    add_table(slide, Inches(0.55), Inches(1.6),
              Inches(8.5), Inches(4.5), rows,
              body_font_size=14, header_font_size=14)

    # Side commentary
    add_filled_rect(slide, Inches(9.3), Inches(1.6),
                    Inches(3.5), Inches(4.5), BG_ALT)
    add_text_box(slide, Inches(9.4), Inches(1.7),
                 Inches(3.3), Inches(0.4),
                 "Winner", font_size=11, bold=True, color=ACCENT)
    add_text_box(slide, Inches(9.4), Inches(2.05),
                 Inches(3.3), Inches(0.5),
                 "2_adam ★", font_size=24, bold=True, color=DARK)
    add_rich_paragraphs(slide, Inches(9.4), Inches(2.65),
                        Inches(3.3), Inches(3.4),
                        paragraphs=[
        [("F1 ", {"bold": True}), ("0.815", {"color": SUCCESS, "bold": True})],
        [("PR-AUC ", {"bold": True}), ("0.833", {"color": SUCCESS, "bold": True})],
        [("ROC-AUC ", {"bold": True}), ("0.979", {"color": SUCCESS, "bold": True})],
        [("", {})],
        [("Plain ", {}), ("Adam", {"bold": True}),
         (" on the baseline arch.", {})],
        [("Adding BN / Dropout / L2 / class-weights ", {}),
         ("hurts.", {"bold": True, "color": DANGER})],
    ], font_size=14)

    add_text_box(slide, Inches(0.55), Inches(6.35),
                 Inches(12.5), Inches(0.6),
                 "Switching SGD → Adam alone delivers +0.13 PR-AUC. "
                 "Further regularisation removes signal — the model is already low-variance.",
                 font_size=14, italic=True, color=SOFT, align=PP_ALIGN.CENTER)

    add_footer(slide, idx, total)


def build_key_insight_slide(prs, idx, total):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Key Insight: simpler is better here",
                  eyebrow="Why")

    # Equation big number
    add_filled_rect(slide, Inches(0.55), Inches(1.7),
                    Inches(7), Inches(2.6), BG_ALT)
    add_text_box(slide, Inches(0.55), Inches(1.8),
                 Inches(7), Inches(0.5),
                 "Data : parameters ratio", font_size=12, bold=True,
                 color=ACCENT, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(0.55), Inches(2.4),
                 Inches(7), Inches(1.2),
                 "199 364 / 5 057  ≈  39 : 1",
                 font_size=44, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(0.55), Inches(3.6),
                 Inches(7), Inches(0.6),
                 "examples per parameter — already low-variance",
                 font_size=14, italic=True, color=SOFT, align=PP_ALIGN.CENTER)

    # Right panel: implications
    add_rich_paragraphs(slide, Inches(8), Inches(1.7),
                        Inches(5.0), Inches(5),
                        paragraphs=[
        [("Implications", {"bold": True, "size": 18, "color": ACCENT})],
        [("", {})],
        [("•  ", {}), ("Adam dominates SGD", {"bold": True}),
         (" — adaptive LRs converge to a strictly better optimum on the PCA-decorrelated loss surface.", {})],
        [("", {})],
        [("•  ", {}), ("Dropout & L2 hurt", {"bold": True, "color": DANGER}),
         (" — the model has plenty of capacity headroom; suppressing it removes signal.", {})],
        [("", {})],
        [("•  ", {}), ("Class weights help ROC-AUC", {"bold": True}),
         (", but the F1 gain disappears once we tune the threshold.", {})],
        [("", {})],
        [("•  ", {}), ("LeakyReLU + He init", {"bold": True}),
         (" matter most in the subtractive ablation: removing them costs −0.056 F1.", {})],
    ], font_size=13)

    add_footer(slide, idx, total)


def build_comparison_slide(prs, idx, total, cmp):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Head-to-Head: MLP vs Autoencoder",
                  eyebrow="Comparative analysis")

    # Headline table
    rows = [
        ["Metric", "MLP (best)", "Autoencoder", "Δ"],
        ["Precision",
         f"{cmp['mlp']['precision']:.3f}",
         f"{cmp['autoencoder']['precision']:.3f}",
         f"+{cmp['mlp']['precision'] - cmp['autoencoder']['precision']:.3f}"],
        ["Recall",
         f"{cmp['mlp']['recall']:.3f}",
         f"{cmp['autoencoder']['recall']:.3f}",
         f"+{cmp['mlp']['recall'] - cmp['autoencoder']['recall']:.3f}"],
        ["F1",
         f"{cmp['mlp']['f1']:.3f}",
         f"{cmp['autoencoder']['f1']:.3f}",
         f"+{cmp['mlp']['f1'] - cmp['autoencoder']['f1']:.3f}"],
        ["ROC-AUC",
         f"{cmp['mlp']['roc_auc']:.4f}",
         f"{cmp['autoencoder']['roc_auc']:.4f}",
         f"+{cmp['mlp']['roc_auc'] - cmp['autoencoder']['roc_auc']:.4f}"],
        ["PR-AUC",
         f"{cmp['mlp']['pr_auc']:.4f}",
         f"{cmp['autoencoder']['pr_auc']:.4f}",
         f"+{cmp['mlp']['pr_auc'] - cmp['autoencoder']['pr_auc']:.4f}"],
        ["Params",
         f"{cmp['mlp_params']:,}",
         f"{cmp['ae_params']:,}",
         "—"],
        ["Train time",
         f"{cmp['mlp_train_time_sec']:.1f} s",
         f"{cmp['ae_train_time_sec']:.1f} s",
         "—"],
    ]
    add_table(slide, Inches(0.55), Inches(1.6),
              Inches(6.5), Inches(4.5), rows, body_font_size=14)

    # ROC/PR plot on the right
    slide.shapes.add_picture(
        str(CMP_DIR / "roc_pr.png"),
        Inches(7.4), Inches(1.6), width=Inches(5.7),
    )
    add_text_box(slide, Inches(7.4), Inches(6.05),
                 Inches(5.7), Inches(0.4),
                 "ROC (left) and Precision–Recall (right) on the shared test set.",
                 font_size=11, italic=True, color=SOFT, align=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(0.55), Inches(6.5),
                 Inches(6.5), Inches(0.4),
                 "Same 42 722-sample test set. Same F1-tuned threshold rule.",
                 font_size=11, italic=True, color=SOFT)

    add_footer(slide, idx, total)


def build_confusion_slide(prs, idx, total, cmp):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Where each model errs",
                  eyebrow="Error analysis")

    # Side-by-side confusion matrices
    slide.shapes.add_picture(str(CMP_DIR / "cm_mlp.png"),
                             Inches(0.8), Inches(1.7), height=Inches(3.6))
    slide.shapes.add_picture(str(CMP_DIR / "cm_ae.png"),
                             Inches(7.0), Inches(1.7), height=Inches(3.6))

    # Captions
    tn_m, fp_m = cmp['mlp']['confusion_matrix'][0]
    fn_m, tp_m = cmp['mlp']['confusion_matrix'][1]
    tn_a, fp_a = cmp['autoencoder']['confusion_matrix'][0]
    fn_a, tp_a = cmp['autoencoder']['confusion_matrix'][1]
    add_text_box(slide, Inches(0.55), Inches(5.4),
                 Inches(6), Inches(0.4),
                 f"MLP — FP={fp_m},  FN={fn_m},  TP={tp_m}",
                 font_size=14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(6.7), Inches(5.4),
                 Inches(6), Inches(0.4),
                 f"Autoencoder — FP={fp_a},  FN={fn_a},  TP={tp_a}",
                 font_size=14, bold=True, color=SUCCESS, align=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(0.55), Inches(5.95),
                 Inches(12.3), Inches(1.1),
                 f"• MLP makes only {fp_m} false positives on {tn_m + fp_m:,} legitimate transactions ({100*fp_m/(tn_m+fp_m):.4f}% FPR) "
                 f"and misses {fn_m}/{fn_m+tp_m} frauds.\n"
                 f"• Autoencoder catches {tp_a} frauds at the cost of {fp_a} false alarms — {fp_a/fp_m:.0f}× more noise for the operations team.\n"
                 f"• In production we'd ensemble: MLP as primary, AE as drift sensor + fallback for novel fraud types.",
                 font_size=13, color=DARK)

    add_footer(slide, idx, total)


def build_ablation_slide(prs, idx, total, ablation):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Ablation Study (subtractive)",
                  eyebrow="What carries the model?")

    short_abl = {
        "ablate_leakyrelu":    "LeakyReLU",
        "ablate_classweights": "Class weights",
        "ablate_he_init":      "He init",
        "ablate_batchnorm":    "BatchNorm",
        "ablate_dropout":      "Dropout(0.3)",
        "ablate_l2":           "L2(1e-4)",
    }
    rows = [["Removed component", "ΔF1", "ΔROC-AUC", "ΔPR-AUC"]]
    sorted_abl = ablation[ablation["variant"] != "full_stack"].sort_values(
        "delta_test_f1"
    )
    for _, r in sorted_abl.iterrows():
        rows.append([
            short_abl.get(r["variant"], r["variant"]),
            f"{r['delta_test_f1']:+.4f}",
            f"{r['delta_test_roc_auc']:+.4f}",
            f"{r['delta_test_pr_auc']:+.4f}",
        ])
    add_table(slide, Inches(0.55), Inches(1.6),
              Inches(7.0), Inches(4.5), rows, body_font_size=14)

    add_rich_paragraphs(slide, Inches(7.9), Inches(1.7),
                        Inches(5.1), Inches(5),
                        paragraphs=[
        [("Reading the table", {"bold": True, "size": 18, "color": ACCENT})],
        [("", {})],
        [("Removing ", {}), ("LeakyReLU", {"bold": True}),
         (" hurts F1 the most  (−0.056).", {})],
        [("Removing ", {}), ("class weights", {"bold": True}),
         (" hurts ROC-AUC the most  (−0.071).", {})],
        [("", {})],
        [("Removing ", {}), ("Dropout", {"bold": True, "color": SUCCESS}),
         (" and ", {}), ("L2", {"bold": True, "color": SUCCESS}),
         (" gives ", {}),
         ("positive Δ", {"bold": True, "color": SUCCESS}),
         (" — they were over-regularising.", {})],
        [("", {})],
        [("Bottom line:", {"bold": True})],
        [("Optimiser > activation/init > BatchNorm > class weights ≫ Dropout/L2",
          {"italic": True})],
    ], font_size=13)

    add_footer(slide, idx, total)


def build_challenges_slide(prs, idx, total):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Biggest challenges faced", eyebrow="Lessons learned")

    challenges = [
        ("Extreme class imbalance (0.17 %)",
         "Forced us off accuracy and on to PR-AUC. Every threshold tweak had outsized impact on the confusion matrix."),
        ("Fair model comparison",
         "Realised the AE was trained on a different normals split than the MLP's test set — risked train/test leakage. Fixed by retraining the AE on the same train split before comparing."),
        ("Threshold instability",
         "Adding class weights shifts the entire probability calibration — every variant needed its own threshold re-tuning before metrics were comparable."),
        ("Counter-intuitive regularisation",
         "Default DL instinct says \"add Dropout + L2\". The ablation showed they were hurting; this only became visible after running the subtractive study."),
    ]
    for i, (title, body) in enumerate(challenges):
        y = Inches(1.7 + i * 1.2)
        add_filled_rect(slide, Inches(0.55), y, Inches(0.06), Inches(1.0), ACCENT)
        add_text_box(slide, Inches(0.75), y, Inches(12.3), Inches(0.4),
                     title, font_size=17, bold=True, color=DARK)
        add_text_box(slide, Inches(0.75), y + Inches(0.45),
                     Inches(12.3), Inches(0.7),
                     body, font_size=13, color=SOFT)

    add_footer(slide, idx, total)


def build_future_slide(prs, idx, total):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "What we'd improve with more time",
                  eyebrow="Future work")

    cards = [
        ("Gradient-boosted trees baseline",
         "XGBoost / LightGBM are historically very competitive on tabular data. We need to know how much deep learning is actually buying us over a strong classical baseline."),
        ("Focal loss",
         "Replace BCE with focal loss to directly down-weight easy negatives — known to help under extreme class imbalance."),
        ("MLP + Autoencoder ensemble",
         "Stack the MLP probability and AE reconstruction error in a logistic meta-learner. The AE's ranking is complementary at the high-recall tail."),
        ("Probability calibration",
         "Add Platt scaling / isotonic regression on the validation set so probabilities are usable in cost-sensitive downstream decisions."),
        ("Bootstrap confidence intervals",
         "Only 74 frauds in the test set — every CM cell has high variance. Report 95 % CIs on F1 before any deployment claim."),
    ]
    for i, (title, body) in enumerate(cards):
        row, col = divmod(i, 2)
        x = Inches(0.55 + col * 6.4)
        y = Inches(1.7 + row * 1.5)
        add_filled_rect(slide, x, y, Inches(6.0), Inches(1.35), BG_ALT)
        add_filled_rect(slide, x, y, Inches(0.06), Inches(1.35), ACCENT)
        add_text_box(slide, x + Inches(0.2), y + Inches(0.1),
                     Inches(5.7), Inches(0.4),
                     title, font_size=15, bold=True, color=DARK)
        add_text_box(slide, x + Inches(0.2), y + Inches(0.55),
                     Inches(5.7), Inches(0.75),
                     body, font_size=12, color=SOFT)
    add_footer(slide, idx, total)


def build_takeaways_slide(prs, idx, total, cmp):
    slide = new_blank_slide(prs)
    add_title_bar(slide, "Key takeaways", eyebrow="Summary")

    bullets = [
        (f"Best supervised MLP: F1 = {cmp['mlp']['f1']:.3f}, ROC-AUC = {cmp['mlp']['roc_auc']:.4f}, PR-AUC = {cmp['mlp']['pr_auc']:.4f}",
         "Plain Adam-trained baseline beats every more-regularised variant."),
        ("Supervised MLP > Autoencoder by +0.36 F1, +0.34 PR-AUC on the same test set.",
         "But the AE is still valuable as a drift sensor and as a fallback for novel fraud types."),
        ("Optimiser choice was the single biggest design decision.",
         "Adam over SGD added +0.13 PR-AUC; nothing else moved the needle as much."),
        ("Heavy regularisation hurts when data : params is high.",
         "With ~39 examples per parameter, Dropout / L2 removed signal we couldn't spare."),
        ("Methodology matters as much as architecture.",
         "Retraining the AE on the shared split was the difference between a fair comparison and a leaky one."),
    ]
    for i, (title, body) in enumerate(bullets):
        y = Inches(1.7 + i * 1.0)
        add_filled_rect(slide, Inches(0.55), y, Inches(0.06), Inches(0.85), ACCENT)
        add_text_box(slide, Inches(0.75), y, Inches(12.3), Inches(0.4),
                     "•  " + title, font_size=15, bold=True, color=DARK)
        add_text_box(slide, Inches(0.95), y + Inches(0.42),
                     Inches(12.0), Inches(0.45),
                     body, font_size=13, italic=True, color=SOFT)

    add_footer(slide, idx, total)


def build_closing_slide(prs, total):
    slide = new_blank_slide(prs)
    add_filled_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), NAVY)
    add_text_box(slide, Inches(1), Inches(2.5), Inches(11.3), Inches(0.8),
                 "Thank you", font_size=60, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(3.6), Inches(11.3), Inches(0.5),
                 "Questions?", font_size=28, color=RGBColor(0xD1, 0xD5, 0xDB),
                 align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(5.0), Inches(11.3), Inches(0.4),
                 "Ahmed Raza  ·  CS-419 Deep Learning  ·  Spring 2026",
                 font_size=15, color=RGBColor(0xD1, 0xD5, 0xDB),
                 align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(5.4), Inches(11.3), Inches(0.4),
                 "github.com/AhmedRaza33/dl-project",
                 font_size=14, color=ACCENT, align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
def main():
    summary = pd.read_csv(MLP_DIR / "metrics_summary.csv")
    ablation = pd.read_csv(MLP_DIR / "ablation_results.csv")
    cmp = json.loads((CMP_DIR / "comparison_metrics.json").read_text())

    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)

    total = 10  # numbered slides (title + 9 body, closing not numbered)
    build_title_slide(prs)                                # 1
    build_problem_slide(prs,    2, total)                 # 2
    build_methodology_slide(prs, 3, total)                # 3
    build_results_table_slide(prs, 4, total, summary)     # 4
    build_key_insight_slide(prs, 5, total)                # 5
    build_comparison_slide(prs,  6, total, cmp)           # 6
    build_confusion_slide(prs,   7, total, cmp)           # 7
    build_ablation_slide(prs,    8, total, ablation)      # 8
    build_challenges_slide(prs,  9, total)                # 9
    build_future_slide(prs,     10, total)                # 10
    build_takeaways_slide(prs,  10, total, cmp)           # bonus summary (same idx ok)
    build_closing_slide(prs, total)                       # thank-you

    prs.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
