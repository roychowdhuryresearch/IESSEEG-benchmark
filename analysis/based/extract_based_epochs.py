#!/usr/bin/env python
"""Extract the EEG epochs that experts actually scored with BASED.

Each of 7 raters independently chose a 5-minute window of each of 20 long
recordings and assigned it a BASED score, so the workbook yields
independent (epoch, score) pairs rather than repeated readings of one
stimulus. This pulls exactly those epochs out of the source EDFs and
writes them in each montage the benchmark models expect.

Only the scored epochs are extracted, not whole recordings: 115 epochs of
5 minutes is a few hundred megabytes, where the full 20 recordings in
three montages would be ~90 GB.

Filtering is applied to the epoch plus a padding margin which is then
trimmed, so the filter's edge transient falls outside the returned
window rather than inside it.

Montages follow preprocessing_release/eeg_preprocess exactly:
  bipolar22  0.5-50 Hz,  no notch   (GBDT, 3D ResNet, 3D ViT, LUNA)
  biot18     0.5-50 Hz,  no notch   (BIOT)
  ref19      0.3-75 Hz,  60 Hz notch (LaBraM, CBraMod, EEGPT, REVE)
"""

import argparse
import os

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

# Paths come from the environment so this runs outside the machine it was
# written on; both are site-specific inputs, not part of the release.
EDF_ROOT = os.environ.get("IESSEEG_LONG_EDF_ROOT", "")
EPOCH_PAIRS = os.environ.get("IESSEEG_BASED_EPOCH_PAIRS", "")

# 21 electrodes in the order the release pipeline indexes into.
ALLOWED = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
           'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2']

# 22-channel longitudinal bipolar ("double banana"), as index pairs.
BIPOLAR22 = [[1, 11], [11, 13], [13, 15], [15, 9],
             [0, 10], [10, 12], [12, 14], [14, 8],
             [20, 13], [13, 5], [5, 17], [17, 4],
             [4, 12], [12, 19], [1, 3], [3, 5],
             [5, 7], [7, 9], [0, 2], [2, 4],
             [4, 6], [6, 8]]

# BIOT's 18-channel bipolar montage, written with 10-20 equivalents
# (T7/T8/P7/P8 in BIOT's naming are T3/T4/T5/T6 here).
BIOT18 = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
          ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
          ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
          ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
          ("FZ", "CZ"), ("CZ", "PZ")]

REF19 = ALLOWED[:19]

MONTAGES = {
    "bipolar22": dict(l_freq=0.5, h_freq=50.0, notch=None),
    "biot18": dict(l_freq=0.5, h_freq=50.0, notch=None),
    "ref19": dict(l_freq=0.3, h_freq=75.0, notch=60.0),
}

SFREQ = 200
PAD_SEC = 30.0


def canon(name):
    n = name.upper().replace("EEG", "").replace("POL", "").replace(" ", "")
    return n.replace("-REF", "").replace("REF", "")


def load_epoch(edf_path, offset_sec, duration_sec):
    """Read one padded epoch from an EDF, in microvolts."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    total = raw.n_times / raw.info["sfreq"]

    start = max(0.0, offset_sec - PAD_SEC)
    stop = min(total, offset_sec + duration_sec + PAD_SEC)
    if stop - start < duration_sec:
        return None, None, None

    raw = raw.crop(tmin=start, tmax=stop).load_data(verbose=False)
    return raw, start, total


def to_montage(raw, kind, offset_sec, duration_sec, epoch_start):
    """Filter, resample, and build one montage; returns (C, T) in uV."""
    cfg = MONTAGES[kind]
    work = raw.copy()
    work.filter(l_freq=cfg["l_freq"], h_freq=cfg["h_freq"], picks="all", verbose=False)
    if cfg["notch"]:
        work.notch_filter(freqs=[cfg["notch"]], picks="all", verbose=False)
    if int(work.info["sfreq"]) != SFREQ:
        work.resample(SFREQ, npad="auto", verbose=False)

    names = [canon(c) for c in work.ch_names]
    data = work.get_data(units="uV")

    index = {}
    for i, n in enumerate(names):
        index.setdefault(n, i)

    if kind == "ref19":
        missing = [c for c in REF19 if c not in index]
        if missing:
            return None, None
        out = np.stack([data[index[c]] for c in REF19])
        labels = [f"{c}-REF" for c in REF19]
    elif kind == "biot18":
        missing = {e for pair in BIOT18 for e in pair if e not in index}
        if missing:
            return None, None
        out = np.stack([data[index[a]] - data[index[b]] for a, b in BIOT18])
        # BIOT's released montage names these T7/T8/P7/P8; the electrode
        # pairs are identical, and matching its labels keeps the extracted
        # files interchangeable with the benchmark's preprocessed tree.
        rename = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
        labels = [f"{rename.get(a, a)}-{rename.get(b, b)}" for a, b in BIOT18]
    else:
        missing = [c for c in ALLOWED if c not in index]
        if missing:
            return None, None
        ordered = np.stack([data[index[c]] for c in ALLOWED])
        out = np.stack([ordered[a] - ordered[b] for a, b in BIPOLAR22])
        labels = [f"{ALLOWED[a]}-{ALLOWED[b]}" for a, b in BIPOLAR22]

    # Trim the padding that absorbed the filter transient.
    lead = int(round((offset_sec - epoch_start) * SFREQ))
    want = int(round(duration_sec * SFREQ))
    out = out[:, lead:lead + want]
    if out.shape[1] < want:
        return None, None
    return out.astype(np.float32), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch_pairs", default=EPOCH_PAIRS or None, required=not EPOCH_PAIRS,
                        help="CSV of (recording, rater, BASED, offset) pairs; "
                             "or set IESSEEG_BASED_EPOCH_PAIRS.")
    parser.add_argument("--edf_root", default=EDF_ROOT or None, required=not EDF_ROOT,
                        help="Directory of source long-recording EDFs; "
                             "or set IESSEEG_LONG_EDF_ROOT.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--duration_sec", type=float, default=300.0,
                        help="BASED is scored on a 5-minute window.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = pd.read_csv(args.epoch_pairs)
    if args.limit:
        pairs = pairs.head(args.limit)

    # Several raters can land on nearly the same offset; extract each
    # (recording, offset) once and let the manifest map raters onto it.
    unique = pairs[["recording_id", "offset_sec"]].drop_duplicates()
    print(f"{len(pairs)} rater epochs -> {len(unique)} unique (recording, offset) extractions")

    for kind in MONTAGES:
        os.makedirs(os.path.join(args.out_dir, kind), exist_ok=True)

    manifest, failures = [], []
    for n, (_, row) in enumerate(unique.iterrows(), 1):
        rec, offset = row.recording_id, float(row.offset_sec)
        edf = os.path.join(args.edf_root, f"{rec}.edf")
        uid = f"{rec}_off{int(offset)}"

        if not os.path.isfile(edf):
            failures.append((uid, "no EDF")); continue

        raw, start, total = load_epoch(edf, offset, args.duration_sec)
        if raw is None:
            failures.append((uid, f"offset {offset:.0f}s outside {total:.0f}s recording")); continue

        wrote = {}
        for kind in MONTAGES:
            data, labels = to_montage(raw, kind, offset, args.duration_sec, start)
            if data is None:
                failures.append((uid, f"{kind}: montage unavailable")); continue
            path = os.path.join(args.out_dir, kind, f"{uid}.npz")
            np.savez_compressed(path, data=data, channel=np.array(labels))
            wrote[kind] = data.shape
        raw.close() if hasattr(raw, "close") else None

        if len(wrote) == len(MONTAGES):
            manifest.append(dict(segment_uid=uid, recording_id=rec, offset_sec=offset,
                                 n_samples=wrote["ref19"][1]))
        if n % 10 == 0 or n == len(unique):
            print(f"  [{n}/{len(unique)}] {uid} {wrote.get('ref19')}")

    man = pd.DataFrame(manifest)
    man.to_csv(os.path.join(args.out_dir, "segments.csv"), index=False)
    print(f"\nextracted {len(man)} segments -> {args.out_dir}/segments.csv")
    if failures:
        print(f"{len(failures)} failures:")
        for uid, why in failures[:10]:
            print(f"  {uid}: {why}")


if __name__ == "__main__":
    main()
