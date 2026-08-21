#!/usr/bin/env python
"""Run benchmark models on the expert-scored BASED epochs.

For each (recording, rater-chosen epoch) this records two things per
model: the clip-level case probability the benchmark would produce, and
the mean penultimate embedding. The first asks whether a model's decision
tracks expert severity; the second asks whether its representation
encodes severity even if the decision head discards it.

Leakage control: every recording belongs to a patient who is held out in
exactly one cross-validation fold, and each epoch is scored with that
fold's model. No epoch is ever scored by a model that trained on its
patient.

The models were trained on case/control labels only, and every BASED
recording is a confirmed case, so any relationship with BASED severity is
something the model learned implicitly rather than being supervised on.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# Root of the benchmark repository; its baselines/ subtree holds both the
# model code and the per-fold checkpoints this reads.
BENCH = os.environ.get("IESSEEG_BENCH_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_ROOT = os.path.join(BENCH, "baselines")
sys.path.insert(0, BENCH)

# Montage tree each model consumes, and the window it expects.
MODEL_SPEC = {
    "luna":  dict(montage="ref19_unused", window_sec=30, sfreq=256),
    "reve":  dict(montage="ref19", window_sec=30, sfreq=200),
    "eegpt": dict(montage="ref19", window_sec=4, sfreq=250),
}


def windows_from(data, window_frames):
    """Tile (C, T) into non-overlapping windows, dropping any remainder."""
    n = data.shape[1] // window_frames
    if n == 0:
        return None
    return np.stack([data[:, i * window_frames:(i + 1) * window_frames] for i in range(n)])


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def load_luna(fold, device):
    sys.path.insert(0, os.path.join(CKPT_ROOT, "luna"))
    from models.LUNA import LUNA
    from luna_data import channel_locations, make_recording_transform
    model = LUNA(patch_size=40, embed_dim=64, num_heads=2, depth=8,
                 num_queries=4, drop_path=0.1, num_classes=2)
    ck = torch.load(os.path.join(CKPT_ROOT, "luna", "ckpts", f"case_control_fold{fold}.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    pos = torch.from_numpy(channel_locations()).to(device)
    return model, dict(positions=pos, transform=make_recording_transform())


def run_luna(model, aux, data, device):
    """LUNA: 22-ch bipolar, resampled to 256 Hz, z-scored, 30 s windows."""
    resampled = aux["transform"](data, aux["channels"])
    win = windows_from(resampled, 30 * 256)
    if win is None:
        return None, None
    x = torch.from_numpy(zscore(win)).float().to(device)
    pos = aux["positions"].unsqueeze(0).expand(x.shape[0], -1, -1)
    with torch.no_grad():
        # Reproduce the classifier head to get the pooled feature it sees.
        tokens, _ = model.prepare_tokens(x, pos, mask=None)
        h, _ = model.cross_attn(tokens)
        from einops import rearrange
        h = rearrange(h, "(b t) q d -> b t (q d)", b=x.shape[0])
        for blk in model.blocks:
            h = blk(h)
        latent = model.norm(h)
        logits = model.classifier(latent)
        emb = latent.mean(dim=1)
    prob = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return prob, emb.float().cpu().numpy()


def load_reve(fold, device):
    sys.path.insert(0, os.path.join(CKPT_ROOT, "reve"))
    from reve_model import ReveClassifier
    from reve_data import REVE_CHANNELS, channel_positions
    model = ReveClassifier(n_classes=2)
    ck = torch.load(os.path.join(CKPT_ROOT, "reve", "ckpts", f"case_control_fold{fold}.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    return model, dict(positions=channel_positions(REVE_CHANNELS).to(device))


def run_reve(model, aux, data, device):
    """REVE: 19-ch referential at native 200 Hz, z-scored, 30 s windows."""
    win = windows_from(data, 30 * 200)
    if win is None:
        return None, None
    x = torch.from_numpy(zscore(win)).float().to(device)
    pos = aux["positions"].unsqueeze(0).expand(x.shape[0], -1, -1)
    with torch.no_grad():
        feats = model.encoder(x, pos)
        pooled = model.encoder.attention_pooling(feats)
        logits = model.head(pooled)
    prob = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return prob, pooled.float().cpu().numpy()


def load_eegpt(fold, device):
    from braindecode.models import EEGPT
    model = EEGPT(n_outputs=2, n_chans=19, n_times=1000, sfreq=250,
                  chan_proj_type="conv1d_constraint", n_chans_target=19)
    ck = torch.load(os.path.join(CKPT_ROOT, "eegpt", "ckpts", f"case_control_fold{fold}.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), {}


def run_eegpt(model, aux, data, device):
    """EEGPT: 19-ch referential resampled to 250 Hz, 4 s windows."""
    from scipy.signal import resample_poly
    from braindecode.preprocessing import exponential_moving_standardize
    resampled = resample_poly(data, 5, 4, axis=-1)
    resampled = exponential_moving_standardize(np.ascontiguousarray(resampled, dtype=np.float64))
    win = windows_from(resampled, 4 * 250)
    if win is None:
        return None, None
    x = torch.from_numpy(win).float().to(device)
    feats = {}

    def capture(module, inputs, output):
        # Must return None: a forward hook that returns a value REPLACES
        # the module's output, which silently turned the logits into the
        # pre-head activation the first time this was written.
        feats["emb"] = inputs[0].detach()

    handle = model.final_layer.register_forward_hook(capture)
    with torch.no_grad():
        logits = model(x)
    handle.remove()
    prob = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    # The head sees (B, patches, summary tokens, features); average over
    # patches and tokens so the embedding is comparable in size to the
    # other models' rather than a 63k-dim flattening that no probe with
    # ~100 samples could fit.
    emb = feats["emb"].mean(dim=(1, 2)).float().cpu().numpy()
    return prob, emb


RUNNERS = {
    "luna": (load_luna, run_luna, "bipolar22"),
    "reve": (load_reve, run_reve, "ref19"),
    "eegpt": (load_eegpt, run_eegpt, "ref19"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch_dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epoch_pairs", default=os.environ.get("IESSEEG_BASED_EPOCH_PAIRS"),
                        help="CSV of (recording, rater, BASED, offset) pairs.")
    parser.add_argument("--subject_meta", default=os.environ.get("IESSEEG_SUBJECT_META"),
                        help="Subject-level metadata linking recordings to patients.")
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    segments = pd.read_csv(os.path.join(args.epoch_dir, "segments.csv"))
    pairs = pd.read_csv(args.epoch_pairs)

    # Each recording's patient is held out in exactly one fold; use that
    # fold's model so no epoch is scored by a model trained on its patient.
    manifest = pd.read_csv(os.path.join(BENCH, "splits", "case_control", "fold_manifest.csv"))
    meta = pd.read_csv(args.subject_meta)
    rec2pat = meta.drop_duplicates("long_recording_id").set_index("long_recording_id")["patient_id"]
    pat2fold = manifest.set_index("patient_id")["fold"]

    for model_name in args.models:
        loader, runner, montage = RUNNERS[model_name]
        print(f"\n=== {model_name} (montage {montage}) ===")

        rows, embeddings = [], []
        cache_fold, cached = None, None

        for _, seg in segments.iterrows():
            rec, uid = seg.recording_id, seg.segment_uid
            if rec not in rec2pat.index:
                print(f"  [skip] {uid}: recording not in metadata"); continue
            fold = int(pat2fold.get(rec2pat[rec], -1))
            if fold < 0:
                print(f"  [skip] {uid}: patient not in fold manifest"); continue

            if cache_fold != fold:
                cached = loader(fold, device); cache_fold = fold
            model, aux = cached

            npz = np.load(os.path.join(args.epoch_dir, montage, f"{uid}.npz"), allow_pickle=True)
            data = npz["data"].astype(np.float64)
            aux = dict(aux, channels=[str(c) for c in npz["channel"]])

            prob, emb = runner(model, aux, data, device)
            if prob is None:
                print(f"  [skip] {uid}: epoch too short for a window"); continue

            raters = pairs[(pairs.recording_id == rec) &
                           (np.isclose(pairs.offset_sec, seg.offset_sec))]
            for _, r in raters.iterrows():
                rows.append(dict(segment_uid=uid, recording_id=rec, offset_sec=seg.offset_sec,
                                 fold=fold, rater=r.rater, role=r.role, based=r.based,
                                 model=model_name, mean_prob=float(prob.mean()),
                                 median_prob=float(np.median(prob)), n_windows=len(prob)))
            embeddings.append(dict(segment_uid=uid, recording_id=rec, fold=fold,
                                   emb=emb.mean(axis=0)))

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out_dir, f"{model_name}_scores.csv"), index=False)
        if embeddings:
            np.savez_compressed(
                os.path.join(args.out_dir, f"{model_name}_embeddings.npz"),
                segment_uid=np.array([e["segment_uid"] for e in embeddings]),
                recording_id=np.array([e["recording_id"] for e in embeddings]),
                fold=np.array([e["fold"] for e in embeddings]),
                emb=np.stack([e["emb"] for e in embeddings]),
            )
        print(f"  {len(df)} (epoch, rater) rows; {len(embeddings)} epoch embeddings "
              f"(dim {embeddings[0]['emb'].shape[0] if embeddings else 0})")


if __name__ == "__main__":
    main()
