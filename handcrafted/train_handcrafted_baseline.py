#!/usr/bin/env python
"""
train_brain_features.py

Trains an XGBoost model (or logistic regression) on EEG recordings described in a metadata CSV.
Computes 'classic brain-science' EEG features for each recording, aggregates them, 
and saves the trained model, confusion matrix plot, and a text file with final metrics.

Now includes:
  - K-fold cross validation (accuracy, AUC, etc.)
  - Final train-test split (20%) for a final model checkpoint
  - Confusion matrix plot & metrics text file

Example usage:
  python train_brain_features.py \
    --metadata_csv "final_open_source_metadata.csv" \
    --data_root "../data/scalp_eeg_200hz_npz" \
    --scenario "awake" \
    --epoch_length 10 \
    --sfreq 200 \
    --test_size 0.2 \
    --model_out "brain_baseline.joblib" \
    --cv_folds 5 \
    --confusion_fig "cm.png" \
    --metrics_out "final_metrics.txt"
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump

# ML
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# For feature extraction
from scipy.signal import welch

# For confusion matrix plotting
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

# -----------------------------------------------------------------------
# 1) EEG Feature Extraction Helpers
# -----------------------------------------------------------------------
def spectral_entropy(signal, sf, band=None):
    """Compute the spectral entropy of a 1D signal over optional band-limited range."""
    freqs, psd = welch(signal, sf, nperseg=len(signal))
    if band is not None:
        (f_low, f_high) = band
        mask = (freqs >= f_low) & (freqs <= f_high)
        psd = psd[mask]
    psd_norm = psd / (psd.sum() + 1e-12)
    s_entropy = -(psd_norm * np.log2(psd_norm + 1e-12)).sum()
    return s_entropy

def hjorth_params(signal):
    """Compute Hjorth mobility and complexity for a single 1D signal."""
    dx = np.diff(signal)
    var0 = np.var(signal)
    var1 = np.var(dx)
    if var0 < 1e-12:
        return (0.0, 0.0)
    mobility = np.sqrt(var1 / var0)
    ddx = np.diff(dx)
    var2 = np.var(ddx)
    complexity = 0.0 if var1 < 1e-12 else np.sqrt(var2/var1)
    return (mobility, complexity)

def compute_epoch_features(epoch_data, sfreq):
    """
    epoch_data: shape (n_channels, n_samples)
    Returns a 1D feature vector, flattening across channels.
    """
    n_channels, n_samples = epoch_data.shape
    band_ranges = {
        "delta": (1,4),
        "theta": (4,8),
        "alpha": (8,12),
        "beta":  (12,30),
        "gamma": (30,50),
    }

    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=n_samples)

    # band powers
    bp_list = []
    for (f_low, f_high) in band_ranges.values():
        mask = (freqs >= f_low) & (freqs <= f_high)
        bp = np.mean(psd[:, mask], axis=1)  # shape (n_channels,)
        bp_list.append(bp)
    band_powers = np.stack(bp_list, axis=1)  # shape => (n_channels, 5)

    # band ratio examples
    delta, theta, alpha, beta, gamma = [band_powers[:, i] + 1e-12 for i in range(5)]
    theta_alpha = theta / alpha
    delta_beta  = delta / beta
    ratio_arr   = np.stack([theta_alpha, delta_beta], axis=1)

    # Hjorth + Spectral Entropy
    hjorth_mob = []
    hjorth_comp= []
    spec_ents  = []

    for ch_idx in range(n_channels):
        mob, comp = hjorth_params(epoch_data[ch_idx, :])
        hjorth_mob.append(mob)
        hjorth_comp.append(comp)

        se = spectral_entropy(epoch_data[ch_idx, :], sfreq, band=None)
        spec_ents.append(se)

    hjorth_mob  = np.array(hjorth_mob)
    hjorth_comp = np.array(hjorth_comp)
    spec_ents   = np.array(spec_ents)

    combined = np.concatenate([
        band_powers,       # shape (n_channels, 5)
        ratio_arr,         # shape (n_channels, 2)
        hjorth_mob[:,None],
        hjorth_comp[:,None],
        spec_ents[:,None],
    ], axis=1)  # => (n_channels, 5+2+1+1+1=10)

    return combined.flatten()  # flatten => (n_channels*10,)

def epoch_data(data, sfreq, epoch_length=10.0):
    """Non-overlapping epoching => list of (n_channels, n_samps_per_epoch)."""
    n_channels, n_times = data.shape
    samp_per_epoch = int(epoch_length * sfreq)
    epochs = []
    start = 0
    while start + samp_per_epoch <= n_times:
        ep = data[:, start:start+samp_per_epoch]
        epochs.append(ep)
        start += samp_per_epoch
    return epochs

def extract_features_for_recording(npz_path, sfreq, epoch_length):
    """
    Loads entire .npz, epochs, computes epoch features, aggregates by (mean,std).
    Returns final feature vector or None if too short for one epoch.
    """
    loaded = np.load(npz_path)
    data_array = loaded['data']  # shape => (n_channels,n_times)
    ep_list = epoch_data(data_array, sfreq, epoch_length)
    if len(ep_list) == 0:
        return None
    all_feats = []
    for ep in ep_list:
        feat = compute_epoch_features(ep, sfreq)
        all_feats.append(feat)
    all_feats = np.array(all_feats)  # (n_epochs, feat_dim)
    mean_f = all_feats.mean(axis=0)
    std_f  = all_feats.std(axis=0)
    return np.concatenate([mean_f, std_f], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta_csv", type=str, required=True,
                        help="CSV with columns: short_recording_id, case_control_label, patient_id, etc.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing .npz training files.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Epoch length in seconds.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction for test split.")
    parser.add_argument("--model_out", type=str, default="brain_baseline.joblib",
                        help="Output path for model checkpoint.")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds (k). 0 means skip CV.")
    parser.add_argument("--confusion_fig", type=str, default="cm.png",
                        help="Where to save confusion matrix plot.")
    parser.add_argument("--metrics_out", type=str, default="metrics.txt",
                        help="Where to save text metrics (accuracy, AUC, classification report).")
    parser.add_argument("--label_key", type=str, default="case_control_label",
                        help="Key to use for labeling (case_control_label, immediate_responder, meaningful_responder).")
    args = parser.parse_args()

    # 1) Load metadata
    df = pd.read_csv(args.train_meta_csv)
    if "short_recording_id" not in df.columns or "case_control_label" not in df.columns or "patient_id" not in df.columns:
        print("metadata_csv must contain short_recording_id, case_control_label, patient_id columns.")
        return

    # Convert "CASE"/"CONTROL" => 1/0
    labels = create_label_from_meta_csv(df, args.label_key)
    df["numeric_label"] = labels

    # We'll store them in arrays
    short_ids   = df["short_recording_id"].values
    labels      = df["numeric_label"].values
    patient_ids = df["patient_id"].values

    # 2) Extract features => We'll build X, y by iterating over short_ids
    X_list = []
    y_list = []
    pid_list = []
    valid_short_ids = []

    print("Extracting features...")
    for sid, lbl, pid in tqdm(zip(short_ids, labels, patient_ids), total=len(short_ids)):
        # Construct .npz path => data_root/sid.npz
        if str(sid).lower().endswith(".npz"):
            npz_filename = str(sid)
        else:
            npz_filename = f"{sid}.npz"
        fullpath = os.path.join(args.data_root, npz_filename)

        feats = extract_features_for_recording(
            npz_path=fullpath,
            sfreq=args.sfreq,
            epoch_length=args.epoch_length
        )

        if feats is None:
            continue  # skip if too short or file not found
        X_list.append(feats)
        y_list.append(lbl)
        pid_list.append(pid)
        valid_short_ids.append(sid)

    X = np.array(X_list)
    y = np.array(y_list)
    pids = np.array(pid_list)
    print(f"Collected {X.shape[0]} valid recordings after feature extraction.")
    if X.shape[0] == 0:
        print("No valid recordings. Exiting.")
        return

    print(f"Feature dimension = {X.shape[1]}")

    # (A) CROSS VALIDATION (Group-based)
    # We'll do group-based K-fold if user wants cv_folds>1
    if args.cv_folds > 1:
        print(f"\nPerforming {args.cv_folds}-fold group-based cross validation...")

        gkf = GroupKFold(n_splits=args.cv_folds)
        xgb_cv = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        acc_scores = []
        auc_scores = []
        f1_scores  = []
        recall_scores = []
        precision_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=pids), start=1):
            print(f"\n=== Fold {fold_idx} ===")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            xgb_cv.fit(X_train, y_train)

            y_pred  = xgb_cv.predict(X_val)
            y_proba = xgb_cv.predict_proba(X_val)[:,1]

            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            f1v = f1_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            pre = precision_score(y_val, y_pred)

            print(f"Fold {fold_idx}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1v:.4f}, Rec={rec:.4f}, Pre={pre:.4f}")

            acc_scores.append(acc)
            auc_scores.append(auc)
            f1_scores.append(f1v)
            recall_scores.append(rec)
            precision_scores.append(pre)

        print(f"\nGroupKFold {args.cv_folds}-fold CV Results:")
        print(f"Accuracy => mean={np.mean(acc_scores):.4f}, std={np.std(acc_scores):.4f}")
        print(f"AUC      => mean={np.mean(auc_scores):.4f}, std={np.std(auc_scores):.4f}")
        print(f"F1       => mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}")
        print(f"Recall   => mean={np.mean(recall_scores):.4f}, std={np.std(recall_scores):.4f}")
        print(f"Precision=> mean={np.mean(precision_scores):.4f}, std={np.std(precision_scores):.4f}")

    # (B) FINAL TRAIN-TEST SPLIT (Group-based)
    # We'll do a 20% test group-split by default
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=pids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nFinal group-based train size: {X_train.shape}, test size: {X_test.shape}")

    # Train XGB
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"FINAL Test Accuracy = {acc:.4f}")
    print(f"FINAL Test AUC      = {auc:.4f}")

    # (C) Save model
    from joblib import dump
    dump(clf, args.model_out)
    print(f"\nModel saved to {args.model_out}")

    # (D) Additional metrics + confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(args.confusion_fig, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix figure saved to {args.confusion_fig}")

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