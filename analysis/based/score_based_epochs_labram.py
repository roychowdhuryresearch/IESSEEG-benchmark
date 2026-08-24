#!/usr/bin/env python
"""Score the expert-rated BASED epochs with the fine-tuned LaBraM models.

Extends the LUNA/REVE/EEGPT epoch scoring to the benchmark's strongest
Task-1 model. Fold routing is reused verbatim from the existing scores
file (each epoch is scored by the model from the fold in which that
patient was held out), so the leakage control is identical by
construction. Embeddings come from `forward_features` (the mean-pooled
penultimate representation) called directly -- no forward hooks, which
silently replace module outputs when they return a value.

Preprocessing replicates the benchmark inference path exactly:
ref19 montage, 10-s windows, x(1/100) dataset scaling then /100 at the
rearrange, softmax over two logits.

Env:
  IESSEEG_BASED_EPOCHS   dir with ref19/{segment_uid}.npz
  IESSEEG_BASED_SCORES   an existing {model}_scores.csv (fold routing)
  IESSEEG_LABRAM_DIR     kfold labram dir (code + checkpoints/)
  IESSEEG_OUT            output dir for labram_scores.csv / _embeddings.npz
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

CH_NAMES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'FZ', 'CZ', 'PZ']
WIN = 2000          # 10 s at 200 Hz
PATCH = 200


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def load_labram(labram_dir, fold, device):
    sys.path.insert(0, labram_dir)
    import utils                              # noqa: E402
    from timm.models import create_model      # noqa: E402
    import modeling_finetune                  # noqa: E402,F401

    model = create_model(
        "labram_base_patch200_200", pretrained=False, num_classes=2,
        drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
        drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
        use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1,
        qkv_bias=False)
    ck = torch.load(os.path.join(labram_dir, "checkpoints",
                                 f"case_control_fold{fold}",
                                 "checkpoint-best.pth"),
                    map_location="cpu", weights_only=False)
    utils.load_state_dict(model, ck.get("model", ck), prefix="")
    model.to(device).eval()
    input_chans = utils.get_input_chans(CH_NAMES)
    return model, input_chans


def main():
    want = os.environ.get("IESSEEG_DEVICE", "cuda:0")
    device = torch.device(want if torch.cuda.is_available() else "cpu")
    out_dir = env("IESSEEG_OUT")
    labram_dir = env("IESSEEG_LABRAM_DIR")
    epoch_dir = os.path.join(env("IESSEEG_BASED_EPOCHS"), "ref19")

    # one row per rating; raters chose different windows, so the 115
    # segment_uids are already unique (one epoch per rating)
    ref = pd.read_csv(env("IESSEEG_BASED_SCORES")) \
            .sort_values("fold").reset_index(drop=True)
    assert ref.segment_uid.is_unique

    rows, embs, uids = [], [], []
    model = input_chans = None
    cache_fold = None
    with torch.no_grad():
        for _, r in ref.iterrows():
            if cache_fold != r.fold:
                model, input_chans = load_labram(labram_dir, int(r.fold),
                                                 device)
                cache_fold = r.fold
            z = np.load(os.path.join(epoch_dir, f"{r.segment_uid}.npz"),
                        allow_pickle=True)
            data = z["data"].astype(np.float32)          # (19, 60000), uV
            n_win = data.shape[1] // WIN
            x = data[:, :n_win * WIN].reshape(19, n_win, WIN) \
                    .transpose(1, 0, 2)                  # (n_win, 19, 2000)
            x = torch.from_numpy(x).to(device) * (1.0 / 100)
            x = rearrange(x, 'B N (A T) -> B N A T', T=PATCH) / 100
            feat = model.forward_features(x, input_chans=input_chans)
            logits = model.head(feat)
            p1 = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            prob = float(p1.mean())
            med = float(np.median(p1))  # np.median: matches the sibling scorer
            rows.append(dict(segment_uid=r.segment_uid,
                             recording_id=r.recording_id,
                             offset_sec=r.offset_sec, fold=int(r.fold),
                             rater=r.rater, role=r.role, based=r.based,
                             model="labram", mean_prob=prob,
                             median_prob=med, n_windows=n_win))
            embs.append(feat.mean(0).cpu().numpy())
            uids.append(r.segment_uid)
            if len(rows) % 25 == 0:
                print(f"  {len(rows)}/{len(ref)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "labram_scores.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "labram_embeddings.npz"),
                        emb=np.stack(embs), segment_uid=np.array(uids))
    print(f"wrote labram_scores.csv ({len(df)}) and labram_embeddings.npz "
          f"(dim {embs[0].shape[0]})")
    # sanity: probabilities must vary (the hook trap produced constants)
    assert df.mean_prob.std() > 1e-4, "degenerate probabilities"


if __name__ == "__main__":
    main()
