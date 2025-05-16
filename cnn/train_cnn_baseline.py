#!/usr/bin/env python
"""
train_brain_3dresnet.py

High-level modular code for training a 3D ResNet on wavelet-based EEG,
with a simple train/validation/test split and early stopping.

Usage example:
  python train_brain_3dresnet.py \
    --train_csv "final_open_source_metadata.csv" \
    --data_root "../data/scalp_eeg_200hz_npz" \
    --epoch_length 10 \
    --sfreq 200 \
    --test_size 0.2 \
    --val_size 0.1 \
    --model_out "brain_3dresnet.pt" \
    --confusion_fig "cm.png" \
    --metrics_out "3dresnet_metrics.txt" \
    --patience 5 \
    --max_epochs 50
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

# Import your model & dataset definitions
from models import resnet18_3d
from dataset import ScalpEEG_Wavelet3D_Dataset
from morlet_feature import make_wavelet_3d_batch_gpu  # Updated to use GPU version

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

class Brain3DResNetTrainer:
    """
    A class to encapsulate the training and evaluation pipeline
    for wavelet-based 3D ResNet classification, now with early stopping.
    """
    def __init__(self, args):
        """
        Loads CSV, sets up config, and stores arguments.
        """
        self.args = args

        # 1) Load metadata CSV
        train_df = pd.read_csv(args.train_csv)

        self.filenames   = train_df["short_recording_id"].values
        self.patient_ids = train_df["patient_id"].values
        self.labels      = create_label_from_meta_csv(train_df, args.label_key)

        print(f"Found {len(self.filenames)} recordings.")

        # 2) Basic config
        self.in_channels = 22   # # of EEG channels
        self.num_classes = 2    # binary classification
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self):
        """
        Single train-test split, then split off a validation set from the training set,
        train the model with early stopping, and evaluate on the test set.
        """
        # (A) Final train-test split
        full_indices = np.arange(len(self.filenames))
        train_idx, test_idx = train_test_split(
            full_indices,
            test_size=self.args.test_size,
            stratify=self.labels,
            random_state=42
        )

        print(f"\nFinal train-test split => train={len(train_idx)}, test={len(test_idx)}")

        # (B) Further split train -> train + validation
        #     We do this only if self.args.val_size > 0
        if self.args.val_size > 0:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=self.args.val_size,
                stratify=self.labels[train_idx],
                random_state=42
            )
            print(f"Train-Validation split => train={len(train_idx)}, val={len(val_idx)}")
        else:
            val_idx = []

        # (C) Build the datasets
        train_ds = self.build_concat_dataset(
            self.filenames[train_idx],
            self.patient_ids[train_idx],
            self.labels[train_idx],
            mode="random"  # random windows for training
        )

        test_ds  = self.build_concat_dataset(
            self.filenames[test_idx],
            self.patient_ids[test_idx],
            self.labels[test_idx],
            mode="continuous"
        )

        if len(val_idx) > 0:
            val_ds = self.build_concat_dataset(
                self.filenames[val_idx],
                self.patient_ids[val_idx],
                self.labels[val_idx],
                mode="continuous"
            )
        else:
            # If no validation split is desired, use test set as a placeholder (not ideal for real training)
            val_ds = test_ds

        # (D) Build DataLoaders
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                                  num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                                  num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                                  num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        # (E) Build final model
        model = resnet18_3d(in_channels=self.in_channels, num_classes=self.num_classes).to(self.device)
        
        # Initialize weights with smaller values to handle large input scale
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Smaller learning rate for stability
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
            min_lr=1e-6  # Set minimum learning rate
        )

        # (F) Train with early stopping
        best_model_state = None
        best_val_loss = float('inf')
        epochs_no_improve = 0
        max_epochs = self.args.max_epochs
        patience = self.args.patience

        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch+1}/{max_epochs}")
            self.train_one_epoch(model, train_loader, criterion, optimizer)

            # Validate on validation set
            val_loss = self.validate_one_epoch(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")

            # Step scheduler based on validation loss
            scheduler.step(val_loss)

            # Check improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping condition
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # (G) Load best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # (H) Evaluate on the test set
        acc_test, auc_test, y_pred, y_true = self.evaluate_loader(model, test_loader, return_preds=True)
        print(f"\nFINAL Test Accuracy = {acc_test:.4f}")
        print(f"FINAL Test AUC      = {auc_test:.4f}")

        # (I) Save model
        torch.save(model.state_dict(), self.args.model_out)
        print(f"Model saved to {self.args.model_out}")

        # (J) Confusion matrix & metrics
        self.compute_confusion_metrics(y_true, y_pred, acc_test, auc_test)

    def build_concat_dataset(self, file_list, patient_id_list, label_list, mode):
        """
        Builds a ConcatDataset of ScalpEEG_Wavelet3D_Dataset from
        multiple recordings => (filename, label).
        """
        ds_list = []
        for fn, pid, lbl in tqdm(list(zip(file_list, patient_id_list, label_list)),
                                 desc="Building dataset"):
            ds = ScalpEEG_Wavelet3D_Dataset(
                data_dir=self.args.data_root,
                patient_id=pid,
                eeg_recording_id=fn,
                label=lbl,
                window_in_sec=self.args.epoch_length,
                sample_rate=self.args.sfreq,
                mode=mode,
                random_sample_num=256
            )
            ds_list.append(ds)
        return ConcatDataset(ds_list)

    def train_one_epoch(self, model, loader, criterion, optimizer):
        """
        Train for one epoch on a given loader.
        """
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Training batch", leave=False):
            # Get raw waveform data and move to GPU
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
                use_channel_as_in_dim=True,  # ResNet expects channels as input dimension
                device=self.device
            )

            optimizer.zero_grad()
            logits = model(wavelet_3d)
            loss = criterion(logits, y_label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Clear GPU memory
            del wavelet_3d
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(loader)
        print(f"Train Loss: {avg_loss:.4f}")

    def validate_one_epoch(self, model, loader, criterion):
        """
        Validate for one epoch on the validation set,
        returning the average validation loss.
        """
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Valid. batch", leave=False):
                # Get raw waveform data and move to GPU
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
                    use_channel_as_in_dim=True,  # ResNet expects channels as input dimension
                    device=self.device
                )

                logits = model(wavelet_3d)
                loss = criterion(logits, y_label)
                total_loss += loss.item()

                # Clear GPU memory
                del wavelet_3d
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(loader)
        return avg_loss

    def evaluate_loader(self, model, loader, return_preds=False):
        """
        Evaluate on a given loader => returns (accuracy, auc, [preds, labels] optional).
        """
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating batch", leave=False):
                # Get raw waveform data and move to GPU
                waveform = batch["waveform"].to(self.device, non_blocking=True)
                labels_cpu = batch["label"].cpu().numpy()

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

                logits = model(wavelet_3d)
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.append(preds)
                all_probs.append(probs[:,1])  # prob for class=1
                all_labels.append(labels_cpu)

                # Clear GPU memory
                del wavelet_3d
                torch.cuda.empty_cache()

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
        """
        Produce confusion matrix plot, classification report,
        and write them to files.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title("Confusion Matrix (Test Set)")
        plt.savefig(self.args.confusion_fig, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix figure saved to {self.args.confusion_fig}")

        cls_report = classification_report(y_true, y_pred, zero_division=0)
        with open(self.args.metrics_out, "w") as f:
            f.write("=== Final Test Metrics (3D ResNet) ===\n")
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
                        help="CSV with columns including short_recording_id, patient_id, case_control_label, etc.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing .npz files.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Seconds for each EEG window.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency (for wavelet transform).")
    parser.add_argument("--test_size", type=float, default=0.0,
                        help="Fraction for test split.")
    parser.add_argument("--val_size", type=float, default=0.0,
                        help="Fraction of (original) training set to reserve for validation.")
    parser.add_argument("--model_out", type=str, default="brain_3dresnet.pt",
                        help="Output path for 3D ResNet checkpoint.")
    parser.add_argument("--confusion_fig", type=str, default="cm.png",
                        help="Where to save confusion matrix plot.")
    parser.add_argument("--metrics_out", type=str, default="3dresnet_metrics.txt",
                        help="Where to save text metrics (accuracy, AUC, classification report).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait for validation loss to improve before early stopping.")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of epochs for training.")
    parser.add_argument("--label_key", type=str, default="case_control_label",
                        help="Key to use for labeling (case_control_label, immediate_responder, meaningful_responder).")
    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU device ID to use (0-based). Set to -1 for CPU.")

    args = parser.parse_args()

    # Instantiate the trainer and run training
    trainer = Brain3DResNetTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
