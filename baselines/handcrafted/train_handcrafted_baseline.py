#!/usr/bin/env python
"""
train_handcrafted_baseline.py

Trains an XGBoost model on EEG recordings described in a metadata CSV.
Computes EEG features PER WINDOW (no pooling across windows -- the
classifier is a genuine window-level classifier, one training row per
window, grouped by patient_id so no patient's windows cross train/test),
and saves the trained model, confusion matrix plot, and a text file with
final metrics.

Feature sets (--feature_set):
  - "regional" (default): 5 anatomical regions, each with band powers,
    ratios, Hjorth, broadband + per-band Shannon entropy, SEF90, DFA/LRTC;
    plus hemispheric asymmetry features and pairwise delta-band PLI
    connectivity (kept un-averaged). See features.py for full rationale.
  - "flat": legacy per-channel flatten (n_channels * 10 dims).
  - "avg": legacy channel-averaged (10 dims).

XGBoost setup:
  - scale_pos_weight is computed automatically from the training label
    distribution (addresses majority-class collapse on imbalanced tasks
    like immediate_responder, rather than relying on dimensionality
    reduction alone to fix it).
  - Regularized by default (shallow trees, min_child_weight, subsample,
    colsample_bytree, L1/L2) -- appropriate for the small-n/high-dim
    regime here.
  - Early stopping against the held-out group-split test set.

Example usage:
  python train_handcrafted_baseline.py \
    --train_meta_csv "fold_0/train.csv" \
    --data_root "../data/scalp_eeg_200hz_npz" \
    --epoch_length 30 \
    --sfreq 200 \
    --test_size 0.1 \
    --model_out "brain_baseline.joblib" \
    --cv_folds 1 \
    --confusion_fig "cm.png" \
    --metrics_out "final_metrics.txt" \
    --label_key case_control_label
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat_mod
import legacy_features


def extract_window_features_for_recording(npz_path, sfreq, epoch_length, feature_set="regional", cache_dir=None,
                                           include_asymmetry=True, include_pli=True):
    """
    Loads entire .npz, epochs into non-overlapping windows, computes a
    feature vector PER WINDOW (no pooling across windows).
    Returns an (n_windows, feat_dim) array, or None if too short.
    """
    if feature_set in ("regional", "global_avg"):
        mode = "regional" if feature_set == "regional" else "global_avg"
        return feat_mod.extract_window_features_for_npz(npz_path, sfreq, epoch_length, cache_dir=cache_dir, mode=mode,
                                                          include_asymmetry=include_asymmetry, include_pli=include_pli)

    if not os.path.isfile(npz_path):
        return None
    loaded = np.load(npz_path)
    data_array = loaded["data"]
    ep_list = legacy_features.epoch_data(data_array, sfreq, epoch_length)
    if len(ep_list) == 0:
        return None
    avg_channels = feature_set == "avg"
    all_feats = [legacy_features.compute_epoch_features(ep, sfreq, avg_channels=avg_channels) for ep in ep_list]
    return np.array(all_feats)


def build_xgb_classifier(scale_pos_weight, eval_set=None):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20 if eval_set is not None else None,
        use_label_encoder=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epoch_length", type=float, default=30.0)
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--model_out", type=str, default="brain_baseline.joblib")
    parser.add_argument("--cv_folds", type=int, default=1,
                         help="Number of cross-validation folds (k). <=1 means skip CV.")
    parser.add_argument("--confusion_fig", type=str, default="cm.png")
    parser.add_argument("--metrics_out", type=str, default="metrics.txt")
    parser.add_argument("--label_key", type=str, default="case_control_label")
    parser.add_argument("--feature_set", type=str, default="regional",
                         choices=["regional", "global_avg", "flat", "avg"],
                         help="regional=grouped+asymmetry+PLI features (default); "
                              "global_avg=same feature types averaged across all 22 channels "
                              "into one global vector (no regions/asymmetry/PLI); "
                              "flat/avg=legacy per-channel feature sets.")
    parser.add_argument("--feature_cache_dir", type=str, default=None,
                         help="If set, per-recording window features are memoized here "
                              "(same recording is reused across CV folds instead of "
                              "being recomputed for each one).")
    parser.add_argument("--exclude_asymmetry", action="store_true",
                         help="(feature_set=regional only) drop the hemispheric asymmetry block.")
    parser.add_argument("--exclude_pli", action="store_true",
                         help="(feature_set=regional only) drop the PLI connectivity block.")
    args = parser.parse_args()

    df = pd.read_csv(args.train_meta_csv)
    if "short_recording_id" not in df.columns or "case_control_label" not in df.columns or "patient_id" not in df.columns:
        print("metadata_csv must contain short_recording_id, case_control_label, patient_id columns.")
        return

    labels = create_label_from_meta_csv(df, args.label_key)
    df["numeric_label"] = labels

    short_ids = df["short_recording_id"].values
    labels = df["numeric_label"].values
    patient_ids = df["patient_id"].values

    X_list, y_list, pid_list = [], [], []

    print(f"Extracting per-window features (feature_set={args.feature_set})...")
    n_recordings_used = 0
    for sid, lbl, pid in tqdm(zip(short_ids, labels, patient_ids), total=len(short_ids)):
        npz_filename = str(sid) if str(sid).lower().endswith(".npz") else f"{sid}.npz"
        fullpath = os.path.join(args.data_root, npz_filename)

        window_feats = extract_window_features_for_recording(
            npz_path=fullpath,
            sfreq=args.sfreq,
            epoch_length=args.epoch_length,
            feature_set=args.feature_set,
            cache_dir=args.feature_cache_dir,
            include_asymmetry=not args.exclude_asymmetry,
            include_pli=not args.exclude_pli,
        )
        if window_feats is None:
            continue
        n_recordings_used += 1
        for feats in window_feats:
            X_list.append(feats)
            y_list.append(lbl)
            pid_list.append(pid)

    X = np.array(X_list)
    y = np.array(y_list)
    pids = np.array(pid_list)
    print(f"Collected {X.shape[0]} windows from {n_recordings_used} valid recordings after feature extraction.")
    if X.shape[0] == 0:
        print("No valid recordings. Exiting.")
        return
    print(f"Feature dimension = {X.shape[1]}")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"Label balance: pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.3f}")

    if args.cv_folds > 1:
        print(f"\nPerforming {args.cv_folds}-fold group-based cross validation...")
        gkf = GroupKFold(n_splits=args.cv_folds)
        acc_scores, auc_scores, f1_scores, recall_scores, precision_scores = [], [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=pids), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            fold_spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            xgb_cv = build_xgb_classifier(fold_spw, eval_set=[(X_val, y_val)])
            xgb_cv.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = xgb_cv.predict(X_val)
            y_proba = xgb_cv.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            f1v = f1_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            pre = precision_score(y_val, y_pred, zero_division=0)

            print(f"Fold {fold_idx}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1v:.4f}, Rec={rec:.4f}, Pre={pre:.4f}")
            acc_scores.append(acc); auc_scores.append(auc); f1_scores.append(f1v)
            recall_scores.append(rec); precision_scores.append(pre)

        print(f"\nGroupKFold {args.cv_folds}-fold CV Results:")
        print(f"Accuracy => mean={np.mean(acc_scores):.4f}, std={np.std(acc_scores):.4f}")
        print(f"AUC      => mean={np.mean(auc_scores):.4f}, std={np.std(auc_scores):.4f}")
        print(f"F1       => mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=pids))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nFinal group-based train size: {X_train.shape}, test size: {X_test.shape}")

    final_spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = build_xgb_classifier(final_spw, eval_set=[(X_test, y_test)])
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"FINAL Test Accuracy = {acc:.4f}")
    print(f"FINAL Test AUC      = {auc:.4f}")

    from joblib import dump
    dump(clf, args.model_out)
    print(f"\nModel saved to {args.model_out}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(args.confusion_fig, bbox_inches="tight")
    plt.close()

    cls_report = classification_report(y_test, y_pred, zero_division=0)
    with open(args.metrics_out, "w") as f:
        f.write("=== Final Test Metrics ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC:      {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report + "\n")

    print(f"Metrics saved to {args.metrics_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
