#!/usr/bin/env python
"""
inference_brain_3dresnet.py

Inference script for a 3D ResNet trained on wavelet-based EEG.
Parallels train_brain_3dresnet.py but uses a meta CSV with columns:
  short_recording_id,long_recording_id,case_control_label,
  pre_post_treatment_label,sleep_awake_label,patient_id,
  Interval,AgeAtEEG1y,AOOmo,LeadtimeD,LeadtimeUKISS

We do:
  - Convert "CASE"/"CONTROL" => label=1/0 if known
  - Build dataset (ScalpEEG_Wavelet3D_Dataset) in "continuous" mode for inference
  - Load trained model from .pt
  - Run inference => store predictions to an output CSV

Example usage:
  python inference_brain_3dresnet.py \
    --inference_csv "short_inference.csv" \
    --data_root "path/to/npz/files" \
    --model_file "brain_3dresnet.pt" \
    --out_csv "inference_results.csv" \
    --epoch_length 10 \
    --sfreq 200
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

# Import the same model & dataset code used in training
from models import resnet18_3d
from dataset import ScalpEEG_Wavelet3D_Dataset
from morlet_feature import make_wavelet_3d_batch_gpu  # Updated to use GPU version

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

class Brain3DResNetInference:
    """
    A class encapsulating the inference pipeline for wavelet-based 3D ResNet.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1) Load inference CSV
        df_infer = pd.read_csv(args.inference_csv)
        # Must have columns => short_recording_id, case_control_label, etc.
        
        labels = create_label_from_meta_csv(df_infer, args.label_key)
        df_infer["binary_label"] = labels

        self.short_ids   = df_infer["short_recording_id"].values
        self.labels      = df_infer["binary_label"].values
        self.patient_ids = df_infer["patient_id"].values if "patient_id" in df_infer.columns else np.zeros_like(self.labels)

        # Keep the entire DataFrame for reference if needed
        self.infer_df = df_infer

        print(f"Found {len(self.short_ids)} recordings in inference CSV.")

        # 2) Load the trained model
        print(f"Loading model from {args.model_file}...")
        self.model = resnet18_3d(in_channels=22, num_classes=2)  # same in_channels as training
        self.model.load_state_dict(torch.load(args.model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def build_concat_dataset(self):
        """
        Builds a ConcatDataset for inference in 'continuous' mode.
        """
        from torch.utils.data import ConcatDataset

        ds_list = []
        for sid, pid, lbl in tqdm(list(zip(self.short_ids, self.patient_ids, self.labels)), desc="Building dataset"):
            ds = ScalpEEG_Wavelet3D_Dataset(
                data_dir = self.args.data_root,
                patient_id = pid,
                eeg_recording_id = sid,    # short recording ID
                label = int(lbl if lbl>=0 else 0),  # use 0 if unknown
                window_in_sec = self.args.epoch_length,
                sample_rate   = self.args.sfreq,
                mode = "continuous"  # or "random" if you want
            )
            ds_list.append(ds)
        return ConcatDataset(ds_list)

    def run_inference(self):
        """
        Runs inference, writes out a CSV with predictions.
        """
        dataset = self.build_concat_dataset()
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # We'll store window-level predictions. If you want to average by short_id,
        # you can do so after collecting predictions in a dict.
        results = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", leave=True):
                # Get raw waveform data and move to GPU
                waveform = batch["waveform"].to(self.device, non_blocking=True)
                known_lbls = batch["label"].numpy()
                rec_ids = batch["recording_id"].numpy()

                # Convert to wavelet on GPU using optimized implementation
                wavelet_3d = make_wavelet_3d_batch_gpu(
                    waveform,
                    sampling_rate=self.args.sfreq,
                    freq_seg=64,
                    min_freq=1,
                    max_freq=100,
                    stdev_cycles=3,
                    use_channel_as_in_dim=True,  # ResNet expects channels as input dimension
                    device=self.device
                )

                # Run model inference
                logits = self.model(wavelet_3d)
                probs = nn.functional.softmax(logits, dim=1)[:,1].cpu().numpy()  # prob for class=1
                preds = (probs >= 0.5).astype(int)

                # Clear GPU memory
                del wavelet_3d
                torch.cuda.empty_cache()

                for i, rid in enumerate(rec_ids):
                    results.append({
                        "short_recording_id": rid,
                        "pred_prob": float(probs[i]),
                        "pred_label": int(preds[i]),
                        "known_label": int(known_lbls[i]) if known_lbls[i]>=0 else None
                    })

        # Option 1) If you want per-window results => just save
        out_df = pd.DataFrame(results)
        # Option 2) If you want to average across all windows that correspond to the same short_id,
        # do a groupby. For example:
        grouped = out_df.groupby("short_recording_id")
        final_rows = []
        for sid, group in grouped:
            mean_prob = group["pred_prob"].mean()
            final_pred = int(mean_prob >= 0.5)
            known_lbl = group["known_label"].iloc[0]  # all the same anyway
            final_rows.append({
                "short_recording_id": sid,
                "pred_prob": mean_prob,
                "pred_label": final_pred,
                "known_label": known_lbl
            })
        final_df = pd.DataFrame(final_rows)

        # Save final aggregated predictions
        if not os.path.exists(os.path.dirname(self.args.out_csv)):
            os.makedirs(os.path.dirname(self.args.out_csv))
        final_df.to_csv(self.args.out_csv, index=False)
        print(f"Saved inference predictions to {self.args.out_csv}. Sample:")
        print(final_df.head(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_csv", type=str, required=True,
                        help="CSV with columns => short_recording_id,case_control_label,...")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder with .npz wavelet data (like training).")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the trained 3D ResNet checkpoint.")
    parser.add_argument("--out_csv", type=str, default="inference_results.csv",
                        help="Where to save predictions.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Seconds for each EEG window.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency (for wavelet transform).")
    parser.add_argument("--label_key", type=str, default="case_control_label",
                        help="Key to use for labeling (case_control_label, immediate_responder, meaningful_responder).")
    parser.add_argument("--cuda", type=int, default=0, help="GPU to use for inference.")
    args = parser.parse_args()

    infer_obj = Brain3DResNetInference(args)
    infer_obj.run_inference()

if __name__ == "__main__":
    main()
