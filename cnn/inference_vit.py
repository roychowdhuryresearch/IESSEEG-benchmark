#!/usr/bin/env python
"""
inference_vit.py

Inference script for a 3D Vision Transformer trained on wavelet-based EEG.
Similar to inference.py but uses ViT.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

# Import the ViT instead of ResNet
from models_vit import vit3d
from dataset import ScalpEEG_Wavelet3D_Dataset
from morlet_feature import make_wavelet_3d_batch_gpu  # Updated to use GPU version

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

class Brain3DViTInference:
    """
    Inference pipeline for the 3D ViT model.
    """
    def __init__(self, args):
        self.args = args
        # Explicitly set the CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(args.cuda)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1) Load inference CSV
        df_infer = pd.read_csv(args.inference_csv)
        df_infer["binary_label"] = create_label_from_meta_csv(df_infer, args.label_key)

        self.short_ids   = df_infer["short_recording_id"].values
        self.labels      = df_infer["binary_label"].values
        self.patient_ids = df_infer["patient_id"].values if "patient_id" in df_infer.columns else np.zeros_like(self.labels)
        self.infer_df    = df_infer

        print(f"Found {len(self.short_ids)} recordings in inference CSV.")

        # 2) Load the trained model
        print(f"Loading model from {args.model_file}...")
        self.model = vit3d(
            in_channels=1,   # Must match how we trained (we unsqueeze(1) for the 3D dimension)
            num_classes=2,
            patch_size=(11, 8, 32), # since it is (1, 22, 64, 2000)
            embed_dim=96,
            depth=3,
            num_heads=6,
            mlp_ratio=2.0,
            dropout=0.3
        ).to(self.device)

        self.model.load_state_dict(torch.load(args.model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def build_concat_dataset(self):
        from torch.utils.data import ConcatDataset
        ds_list = []
        for sid, pid, lbl in tqdm(list(zip(self.short_ids, self.patient_ids, self.labels)), desc="Building dataset"):
            ds = ScalpEEG_Wavelet3D_Dataset(
                data_dir=self.args.data_root,
                patient_id=pid,
                eeg_recording_id=sid,
                label=int(lbl if lbl >= 0 else 0),  # if unknown, use 0
                window_in_sec=self.args.epoch_length,
                sample_rate=self.args.sfreq,
                mode="continuous",
            )
            ds_list.append(ds)
        return ConcatDataset(ds_list)

    def run_inference(self):
        dataset = self.build_concat_dataset()
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        results = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", leave=True):
                # Get raw waveform data and move to GPU
                waveform = batch["waveform"].to(self.device)
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
                    use_channel_as_in_dim=False
                ).to(self.device)

                # Run model inference
                logits = self.model(wavelet_3d)
                probs = nn.functional.softmax(logits, dim=1)[:,1].cpu().numpy()
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

        # Per-window results
        out_df = pd.DataFrame(results)

        # Optional groupby for final aggregated predictions by short_id
        grouped = out_df.groupby("short_recording_id")
        final_rows = []
        for sid, group in grouped:
            mean_prob = group["pred_prob"].mean()
            final_pred = int(mean_prob >= 0.5)
            known_lbl = group["known_label"].iloc[0]
            final_rows.append({
                "short_recording_id": sid,
                "pred_prob": mean_prob,
                "pred_label": final_pred,
                "known_label": known_lbl
            })
        final_df = pd.DataFrame(final_rows)

        if not os.path.exists(os.path.dirname(self.args.out_csv)):
            os.makedirs(os.path.dirname(self.args.out_csv), exist_ok=True)

        final_df.to_csv(self.args.out_csv, index=False)
        print(f"Saved inference predictions to {self.args.out_csv}. Sample:")
        print(final_df.head(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_csv", type=str, required=True,
                        help="CSV with columns => short_recording_id, case_control_label,...")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder with .npz wavelet data (like training).")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the trained 3D ViT checkpoint.")
    parser.add_argument("--out_csv", type=str, default="inference_results_vit.csv",
                        help="Where to save predictions.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Seconds for each EEG window.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency (for wavelet transform).")
    parser.add_argument('--label_key', type=str, default="case_control_label",
                        help="Key to use for label column in inference CSV.")
    parser.add_argument('--cuda', type=int, default=0,
                        help="CUDA device to use.")
    args = parser.parse_args()

    infer_obj = Brain3DViTInference(args)
    infer_obj.run_inference()

if __name__ == "__main__":
    main()
