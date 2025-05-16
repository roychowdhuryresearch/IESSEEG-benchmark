#!/usr/bin/env python
"""
train_vit_baseline.py

High-level code for training a 3D Vision Transformer on wavelet-based EEG,
with train/validation/test split and early stopping.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# Import your 3D ViT
from models_vit import vit3d
from dataset import ScalpEEG_Wavelet3D_Dataset
from morlet_feature import make_wavelet_3d_batch_gpu

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

class Brain3DViTTrainer:
    """
    Encapsulates the training/evaluation pipeline for a 3D Vision Transformer,
    using a train/validation/test split and early stopping.
    """
    def __init__(self, args):
        self.args = args

        # 1) Load metadata CSV
        df = pd.read_csv(args.train_csv)
        self.filenames   = df["short_recording_id"].values
        self.patient_ids = df["patient_id"].values
        self.labels      = create_label_from_meta_csv(df, args.label_key)

        print(f"Found {len(self.filenames)} recordings.")

        # 2) Basic config
        #
        # IMPORTANT: in_channels=1 for the 3D model, because we'll unsqueeze dimension 1
        # to shape => (batch, 1, 22, 64, 2000).
        #
        self.in_channels = 1
        self.num_classes = 2
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self):
        """
        Train with an internal train/validation split (args.val_size),
        then evaluate on test set, and do early stopping based on validation loss.
        """
        # Split => train + test
        all_indices = np.arange(len(self.filenames))
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=self.args.test_size,
            stratify=self.labels,
            random_state=42
        )
        print(f"\nFinal train-test split => train={len(train_idx)}, test={len(test_idx)}")

        # Further split train => train + validation
        if self.args.val_size > 0:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=self.args.val_size,
                stratify=self.labels[train_idx],
                random_state=42
            )
            print(f"Train-Validation split => train={len(train_idx)}, val={len(val_idx)}")
        else:
            # If val_size = 0, no separate validation set
            val_idx = []

        # Build datasets
        train_ds = self.build_concat_dataset(
            self.filenames[train_idx],
            self.patient_ids[train_idx],
            self.labels[train_idx],
            mode="random"
        )
        test_ds  = self.build_concat_dataset(
            self.filenames[test_idx],
            self.patient_ids[test_idx],
            self.labels[test_idx],
            mode="continuous"
        )

        # If we have a validation split, build it; otherwise fallback to test_ds
        if len(val_idx) > 0:
            val_ds = self.build_concat_dataset(
                self.filenames[val_idx],
                self.patient_ids[val_idx],
                self.labels[val_idx],
                mode="continuous"
            )
        else:
            val_ds = test_ds  # Not recommended for real usage, but for code completeness.

        # Build DataLoaders
        train_loader = DataLoader(
            train_ds, 
            batch_size=64,  # Increase batch size
            shuffle=True, 
            num_workers=8,  # Reduce workers to prevent memory issues
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=32)
        test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=32)

        # Build model
        model = vit3d(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            patch_size=(11, 8, 32),  # This will give us 11 * 8 * 100 = 8800 patches
            embed_dim=96,  # Reduced from 128 to prevent overfitting
            depth=3,  # Keep depth at 3
            num_heads=6,  # Reduced from 8 to match embed_dim
            mlp_ratio=2.0,  # Reduced from 4.0
            dropout=0.3,  # Increased dropout for better regularization
        ).to(self.device)

        # Initialize weights with smaller values to handle large input scale
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        model.apply(init_weights)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-6,  # Even smaller learning rate for stability
            weight_decay=0.1,  # Stronger weight decay
            betas=(0.9, 0.999),  # Default betas
            eps=1e-8  # Larger epsilon for numerical stability
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-7  # Set minimum learning rate
        )

        # Early stopping setup
        best_model_state = None
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = self.args.patience
        max_epochs = self.args.max_epochs

        # Main training loop
        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch+1}/{max_epochs}")

            # Train for one epoch
            train_loss = self.train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate_one_epoch(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")

            # Step scheduler based on validation loss
            scheduler.step(val_loss)

            # Check improvement and save best model
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()  # Make a copy of the state dict
                epochs_no_improve = 0
                # Save the best model immediately
                torch.save(best_model_state, self.args.model_out)
                print(f"New best model saved to {self.args.model_out}")
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve. Best loss: {best_val_loss:.4f}")

            # Early stopping condition
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

        # Evaluate on test
        acc_test, auc_test, y_pred, y_true = self.evaluate_loader(model, test_loader, return_preds=True)
        print(f"\nFINAL Test Accuracy = {acc_test:.4f}")
        print(f"FINAL Test AUC      = {auc_test:.4f}")

        # Save the model
        torch.save(model.state_dict(), self.args.model_out)
        print(f"Model saved to {self.args.model_out}")

        # Confusion matrix, classification report, etc.
        self.compute_confusion_metrics(y_true, y_pred, acc_test, auc_test)

    def build_concat_dataset(self, file_list, patient_id_list, label_list, mode):
        ds_list = []
        for fn, pid, lbl in tqdm(zip(file_list, patient_id_list, label_list),
                                 desc="Building dataset", total=len(file_list)):
            ds = ScalpEEG_Wavelet3D_Dataset(
                data_dir=self.args.data_root,
                patient_id=pid,
                eeg_recording_id=fn,
                label=lbl,
                window_in_sec=self.args.epoch_length,
                sample_rate=self.args.sfreq,
                mode=mode,
                random_sample_num=256,
                # use_channel_as_in_dim=False  # 3D model expects shape => (B, 1, 22, 64, 2000)
            )
            ds_list.append(ds)
        return ConcatDataset(ds_list)

    def train_one_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Training batch", leave=False):
            # Get raw waveform data
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            y_label = batch["label"].to(self.device)

            # Convert to wavelet on GPU using optimized implementation
            wavelet_3d = make_wavelet_3d_batch_gpu(
                waveform,
                sampling_rate=self.args.sfreq,
                freq_seg=64,
                min_freq=1,
                max_freq=100,
                stdev_cycles=3,
                use_channel_as_in_dim=False,
                device=self.device
            )

            optimizer.zero_grad()
            logits = model(wavelet_3d)
            loss = criterion(logits, y_label)
            loss.backward()
            
            # Tighter gradient clipping for stability with large values
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            total_loss += loss.item()

            # Clear GPU memory
            del wavelet_3d
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(loader)
        return avg_loss

    def validate_one_epoch(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation batch", leave=False):
                waveform = batch["waveform"].to(self.device)  # Move to GPU immediately
                y_label = batch["label"].to(self.device)

                # Convert to wavelet on GPU using optimized implementation
                wavelet_3d = make_wavelet_3d_batch_gpu(
                    waveform,
                    sampling_rate=self.args.sfreq,
                    freq_seg=64,
                    min_freq=1,
                    max_freq=100,
                    stdev_cycles=3,
                    use_channel_as_in_dim=False,
                    device=self.device
                )

                logits = model(wavelet_3d)
                loss = criterion(logits, y_label)
                total_loss += loss.item()

                # Clear GPU memory
                del wavelet_3d
                torch.cuda.empty_cache()

        return total_loss / len(loader)

    def evaluate_loader(self, model, loader, return_preds=False):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating batch", leave=False):
                waveform = batch["waveform"].to(self.device)  # Move to GPU immediately
                labels_cpu = batch["label"].cpu().numpy()

                # Convert to wavelet on GPU using optimized implementation
                wavelet_3d = make_wavelet_3d_batch_gpu(
                    waveform,
                    sampling_rate=self.args.sfreq,
                    freq_seg=64,
                    min_freq=1,
                    max_freq=100,
                    stdev_cycles=3,
                    use_channel_as_in_dim=False,
                    device=self.device
                )

                logits = model(wavelet_3d)
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.append(preds)
                all_probs.append(probs[:, 1])  # prob for class=1
                all_labels.append(labels_cpu)

        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        try:
            auc_ = roc_auc_score(all_labels, all_probs)
        except:
            auc_ = 0.0

        if return_preds:
            return acc, auc_, all_preds, all_labels
        else:
            return acc, auc_

    def compute_confusion_metrics(self, y_true, y_pred, acc, auc_):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title("Confusion Matrix (Test Set)")
        plt.savefig(self.args.confusion_fig, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix figure saved to {self.args.confusion_fig}")

        cls_report = classification_report(y_true, y_pred, zero_division=0)
        with open(self.args.metrics_out, "w") as f:
            f.write("=== Final Test Metrics (3D ViT) ===\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC:      {auc_:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(cls_report + "\n")

        print(f"Metrics saved to {self.args.metrics_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True,
                        help="CSV with columns: short_recording_id, patient_id, case_control_label, etc.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing .npz files.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Seconds for each EEG window.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency for wavelet transform.")
    parser.add_argument("--test_size", type=float, default=0.0,
                        help="Fraction for the test split.")
    parser.add_argument("--val_size", type=float, default=0.0,
                        help="Fraction of the *training* set to reserve for validation.")
    parser.add_argument("--model_out", type=str, default="brain_vit3d.pt",
                        help="Output path for 3D ViT checkpoint.")
    parser.add_argument("--confusion_fig", type=str, default="cm_vit.png",
                        help="Where to save confusion matrix plot.")
    parser.add_argument("--metrics_out", type=str, default="vit3d_metrics.txt",
                        help="Where to save text metrics.")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum training epochs before stopping.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience. Stop if val loss doesn't improve for this many epochs.")
    parser.add_argument("--cuda", type=int, default=0,
                        help="Which GPU to use (0-based) if available.")
    parser.add_argument("--label_key", type=str, default="case_control_label",
                        help="Key to use for labeling (case_control_label, immediate_responder, meaningful_responder).")

    args = parser.parse_args()

    # Instantiate trainer and run
    trainer = Brain3DViTTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
