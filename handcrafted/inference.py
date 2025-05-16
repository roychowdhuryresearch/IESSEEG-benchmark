#!/usr/bin/env python
"""
inference_xgb_30min.py

Reads:
  - A trained XGBoost model (.joblib)
  - A CSV (produced by gen_inference_meta.py) with columns [file_id, npz_path, label]
    * "npz_path" is the path to a .npz file (e.g. "0.npz"), each containing data_array shape (n_channels, n_times)
    * "label" might be None/NaN or 0/1 if known
  - epoch_length, sfreq, etc.

For each row, we:
  1. Load the .npz
  2. Epoch it (like in training)
  3. Extract mean/std of "classic brain features" (band powers, spectral entropy, etc.)
  4. Use the loaded XGBoost model to predict probability => case vs control
  5. Write out an output CSV with [file_id, npz_path, known_label, pred_label, pred_prob].

Example usage:
  python inference_xgb_30min.py \
    --model_file "brain_xgb.joblib" \
    --inference_csv "case_control_test.csv" \
    --out_csv "inference_results.csv" \
    --epoch_length 10 \
    --sfreq 200
"""

import os
import argparse
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

# XGBoost
from xgboost import XGBClassifier

# If you used the same feature extraction code from training:
from scipy.signal import welch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv


def epoch_data(data, sfreq, epoch_length=10.0):
    """Non-overlapping epoching, shape => list of (n_channels, n_samps_per_epoch)."""
    n_channels, n_times = data.shape
    samp_per_epoch = int(epoch_length * sfreq)
    epochs = []
    start = 0
    while start + samp_per_epoch <= n_times:
        ep = data[:, start:start+samp_per_epoch]
        epochs.append(ep)
        start += samp_per_epoch
    return epochs

def spectral_entropy(signal, sf):
    freqs, psd = welch(signal, sf, nperseg=len(signal))
    psd_norm = psd / (psd.sum() + 1e-12)
    return -(psd_norm * np.log2(psd_norm + 1e-12)).sum()

def hjorth_params(signal):
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
    Returns a 1D feature vector across all channels.
    Same approach as your training script: band powers, ratios, Hjorth, spectral entropy.
    """
    # Define bands
    band_ranges = {
        "delta": (1,4),
        "theta": (4,8),
        "alpha": (8,12),
        "beta":  (12,30),
        "gamma": (30,50),
    }
    n_channels, n_samples = epoch_data.shape

    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=n_samples)

    # band powers
    bp_list = []
    for (f_low, f_high) in band_ranges.values():
        mask = (freqs >= f_low) & (freqs <= f_high)
        bp = np.mean(psd[:, mask], axis=1)  # shape (n_channels,)
        bp_list.append(bp)
    band_powers = np.stack(bp_list, axis=1)  # => (n_channels, 5)

    # ratios
    delta, theta, alpha, beta, gamma = [band_powers[:, i] + 1e-12 for i in range(5)]
    theta_alpha = theta / alpha
    delta_beta  = delta / beta
    ratio_arr   = np.stack([theta_alpha, delta_beta], axis=1)

    # Hjorth & spectral entropy
    hjorth_mob, hjorth_comp, spec_ents = [], [], []
    for ch_idx in range(n_channels):
        mob, comp = hjorth_params(epoch_data[ch_idx, :])
        hjorth_mob.append(mob)
        hjorth_comp.append(comp)

        se = spectral_entropy(epoch_data[ch_idx, :], sfreq)
        spec_ents.append(se)

    hjorth_mob  = np.array(hjorth_mob)
    hjorth_comp = np.array(hjorth_comp)
    spec_ents   = np.array(spec_ents)

    combined = np.concatenate([
        band_powers,      # (n_channels, 5)
        ratio_arr,        # (n_channels, 2)
        hjorth_mob[:,None],
        hjorth_comp[:,None],
        spec_ents[:,None],
    ], axis=1)  # shape => (n_channels, 5+2+1+1+1=10)

    return combined.flatten()

def extract_features_for_npz(npz_path, sfreq, epoch_length):
    """
    Loads .npz => data (n_channels, n_times),
    epochs, compute features for each epoch, aggregates mean/std => final.
    """
    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        return None
    loaded = np.load(npz_path)
    data_array = loaded["data"]  # shape => (n_channels,n_times)

    # epoch
    ep_list = epoch_data(data_array, sfreq, epoch_length)
    if len(ep_list) == 0:
        # too short
        return None

    all_features = []
    for ep in ep_list:
        feat = compute_epoch_features(ep, sfreq)
        all_features.append(feat)

    all_features = np.array(all_features)  # => (n_epochs, feat_dim)
    mean_f = all_features.mean(axis=0)
    std_f  = all_features.std(axis=0)
    return np.concatenate([mean_f, std_f], axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the XGBoost .joblib model (from training).")
    parser.add_argument("--inference_csv", type=str, required=True,
                        help="CSV with [file_id, npz_path, label]. e.g. from gen_inference_meta.py")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder with .npz wavelet data (like training).")
    parser.add_argument("--out_csv", type=str, default="inference_results.csv",
                        help="Output CSV with predictions.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Epoch length in seconds.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency (should match training).")
    parser.add_argument("--label_key", type=str, default="case_control_label",
                        help="Key to use for labeling (case_control_label, immediate_responder, meaningful_responder).")

    args = parser.parse_args()

    print(f"Loading model from {args.model_file}...")
    model = load(args.model_file)  # XGBClassifier from joblib

    print(f"Reading inference CSV from {args.inference_csv}...")
    df_infer = pd.read_csv(args.inference_csv)  # columns => file_id, npz_path, label

    labels = create_label_from_meta_csv(df_infer, args.label_key)
    df_infer["binary_label"] = labels
    
    results = []
    for row in tqdm(df_infer.itertuples(), total=len(df_infer), desc="Inference"):
        short_id = row.short_recording_id
        patient_ids = row.patient_id
        known_label = row.binary_label

        feats = extract_features_for_npz(f"{args.data_root}/{short_id}.npz", args.sfreq, args.epoch_length)
        if feats is None:
            print(f"Skipping file_id={short_id} due to short/no data: {args.data_root}/{short_id}.npz")
            continue

        feats_2d = feats.reshape(1, -1)
        proba = model.predict_proba(feats_2d)[0,1]
        pred_label = int(proba >= 0.5)

        results.append({
            "short_recording_id": short_id,
            "known_label": known_label,
            "pred_prob": proba,
            "pred_label": pred_label
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nDone. Saved inference results to {args.out_csv}. Sample:\n{out_df.head(10)}")

if __name__ == "__main__":
    main()
# This script is designed to run inference using a pre-trained XGBoost model on EEG data.   