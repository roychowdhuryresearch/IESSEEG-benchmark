#!/usr/bin/env python
"""
eval_classification_metrics.py

Reads an inference CSV (e.g., from inference_xgb_30min.py) with columns:
  - file_id
  - npz_path
  - known_label (may be NaN if unknown)
  - pred_prob (float)
  - pred_label (0 or 1)

Computes classification metrics (accuracy, AUC, precision, recall, F1, confusion matrix)
for rows where known_label is available (not NaN), and saves a confusion matrix plot.

Example usage:
  python eval_classification_metrics.py \
    --prediction_csv "inference_results.csv" \
    --confusion_fig  "confusion_matrix.png"
"""

import argparse
import pandas as pd
import numpy as np
import os
import json

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_csv", type=str, required=True,
                        help="Path to CSV with columns: [file_id, known_label, pred_label, pred_prob, ...].")
    parser.add_argument("--output_folder", type=str,
                        help="Path to save the confusion matrix plot and metrics.")
    parser.add_argument("--human_label_meta", type=str, default="../data/final_test.csv",
                        help="Path to the human label meta data.")
    args = parser.parse_args()

    # Load human label meta data
    human_label_df = pd.read_csv(args.human_label_meta)
    human_label_df = human_label_df[["short_recording_id", "human_label"]]
    human_label_df["short_recording_id"] = human_label_df["short_recording_id"].astype(str)

    # Load predictions
    df = pd.read_csv(args.prediction_csv)
    # find column with name recording_id as substring
    recording_id_col = [col for col in df.columns if "recording_id" in col][0]
    df = df.rename(columns={recording_id_col: "short_recording_id"})

    df["short_recording_id"] = df["short_recording_id"].astype(str)
    valid_df = df.merge(human_label_df, on="short_recording_id", how="left")
    
    if "case_vs_control" in args.prediction_csv.split("/")[-2]:
        y_true = valid_df["human_label"]
    elif "immediate_responder" in args.prediction_csv.split("/")[-2]:
        y_true = valid_df["known_label"]
    elif "meaningful_responder" in args.prediction_csv.split("/")[-2]:
        y_true = valid_df["known_label"]

    # Extract arrays
    y_true = y_true.astype(int)
    y_pred = valid_df["pred_label"].astype(int)

    # For AUC, we need pred_prob
    if "pred_prob" in valid_df.columns:
        y_prob = valid_df["pred_prob"].astype(float)
    else:
        y_prob = None

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sens    = recall_score(y_true, y_pred, zero_division=0)          # sensitivity
    f1      = f1_score(y_true, y_pred, zero_division=0)

    # specificity from confusion-matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    auc_value = None
    if y_prob is not None and len(set(y_true)) > 1:
        auc_value = roc_auc_score(y_true, y_prob)
    # ────────────────────────────────────────────────────────────────

    # ─────────────────────── print / save ───────────────────────────
    print("\nClassification Metrics:")
    print(f"AUROC            = {auc_value:.4f}" if auc_value else "AUROC            = N/A")
    print(f"Balanced Accuracy= {bal_acc:.4f}")
    print(f"Sensitivity      = {sens:.4f}")
    print(f"Specificity      = {spec:.4f}")
    print(f"F1-score         = {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # classification report
    class_report = classification_report(y_true, y_pred, zero_division=0)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:\n", class_report)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(10,8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix")
    confusion_fig_path = os.path.join(args.output_folder, "confusion_matrix.png")
    plt.savefig(confusion_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_fig_path}")

    # Round every float to three decimals (without breaking None values)
    f3 = lambda v: None if v is None else f"{v:.3f}"

    metrics = dict(
        auroc            = f3(auc_value),
        balanced_accuracy= f3(bal_acc),
        sensitivity      = f3(sens),
        specificity      = f3(spec),
        f1               = f3(f1),
    )
    with open(os.path.join(args.output_folder, "metrics.json"), "w") as jf:
        json.dump(metrics, jf, indent=2)
    print(f"\nAll outputs written to: {os.path.abspath(args.output_folder)}")

if __name__ == "__main__":
    main()
