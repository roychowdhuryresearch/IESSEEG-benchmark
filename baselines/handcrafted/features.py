"""
Shared EEG feature-extraction module for the handcrafted baseline.

Feature design:
  - The 22-channel double-banana bipolar montage is grouped into 5
    anatomical regions (right/left temporal chains, central chain,
    right/left parasagittal chains). Per-channel features are averaged
    within each region. This trades raw per-channel resolution for a much
    lower-dimensional, less overfit-prone feature space, while still
    preserving broad spatial localization -- unlike flat all-channel
    averaging, which collapses everything down to one global number.
  - Regional features: band powers (delta/theta/alpha/beta/gamma), two
    band ratios, Hjorth mobility/complexity, broadband spectral entropy,
    per-band Shannon entropy, and spectral edge frequency (SEF90). Follows
    the IESS treatment-response QEEG literature (PMC12525723) on
    band-limited entropy being predictive of response.
    (Long-range temporal correlation / DFA was tried and removed: a
    per-recording-constant version risked letting XGBoost shortcut on
    recording identity rather than genuine window-level signal, and a
    per-window/30s version's estimate was judged too unreliable relative
    to the rest of the feature set -- worth revisiting with a properly
    sized window later.)
  - Hemispheric asymmetry features (right-minus-left regional features)
    for the two mirrored region pairs (temporal, parasagittal), since
    hypsarrhythmia is classically multifocal/asymmetric.
  - Phase-Lag Index (PLI) functional connectivity in the delta band,
    computed between every pair of the 5 regions and kept as separate
    features -- never averaged into a single connectivity scalar, since
    PLI is only meaningful pairwise.
"""

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt

BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 50),
}

# Channel order as stored in the .npz files (double-banana bipolar
# montage) -- see data/scalp_eeg_data_200HZ_np_format/*.npz 'channel' field.
CHANNEL_NAMES = [
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "A2-T4", "T4-C4", "C4-CZ", "CZ-C3", "C3-T3", "T3-A1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
]

REGIONS = {
    "right_temporal":     [0, 1, 2, 3],
    "left_temporal":      [4, 5, 6, 7],
    "central":            [8, 9, 10, 11, 12, 13],
    "right_parasagittal": [14, 15, 16, 17],
    "left_parasagittal":  [18, 19, 20, 21],
}
REGION_NAMES = list(REGIONS.keys())

# Mirrored region pairs used for hemispheric asymmetry features.
ASYMMETRY_PAIRS = [
    ("right_temporal", "left_temporal"),
    ("right_parasagittal", "left_parasagittal"),
]

DELTA_BAND = BAND_RANGES["delta"]

N_FEATS_PER_REGION = 5 + 2 + 2 + 1 + 5 + 1  # band powers, ratios, hjorth, broadband ent, per-band ent, sef90
# DFA/LRTC removed: a per-recording-constant version risked letting XGBoost
# shortcut on recording identity, and a per-window (30s) version's estimate
# is much noisier than the rest of the feature set warrants revisiting later.
N_REGION_PAIRS = len(REGION_NAMES) * (len(REGION_NAMES) - 1) // 2
FEATURE_DIM = N_FEATS_PER_REGION * len(REGION_NAMES) + N_FEATS_PER_REGION * len(ASYMMETRY_PAIRS) + N_REGION_PAIRS

_PER_REGION_FEAT_NAMES = (
    [f"bandpower_{b}" for b in BAND_RANGES] +
    ["ratio_theta_alpha", "ratio_delta_beta"] +
    ["hjorth_mobility", "hjorth_complexity"] +
    ["entropy_broadband"] +
    [f"entropy_{b}" for b in BAND_RANGES] +
    ["sef90"]
)  # must match the concatenation order in _region_features()

FEATURE_NAMES = (
    [f"{region}__{feat}" for region in REGION_NAMES for feat in _PER_REGION_FEAT_NAMES] +
    [f"asym_{right}_minus_{left}__{feat}"
     for right, left in ASYMMETRY_PAIRS for feat in _PER_REGION_FEAT_NAMES] +
    [f"pli_delta__{REGION_NAMES[i]}__{REGION_NAMES[j]}"
     for i in range(len(REGION_NAMES)) for j in range(i + 1, len(REGION_NAMES))]
)  # must match the concatenation order in compute_window_features()
assert len(FEATURE_NAMES) == FEATURE_DIM


def hjorth_params(signal):
    dx = np.diff(signal)
    var0 = np.var(signal)
    var1 = np.var(dx)
    if var0 < 1e-12:
        return 0.0, 0.0
    mobility = np.sqrt(var1 / var0)
    ddx = np.diff(dx)
    var2 = np.var(ddx)
    complexity = 0.0 if var1 < 1e-12 else np.sqrt(var2 / var1)
    return mobility, complexity


def spectral_entropy(signal, sf, band=None):
    freqs, psd = welch(signal, sf, nperseg=len(signal))
    return spectral_entropy_from_psd(freqs, psd, band=band)


def spectral_entropy_from_psd(freqs, psd, band=None):
    """Same as spectral_entropy but reuses an already-computed PSD, avoiding
    a redundant welch() call -- important since _region_features already
    computes one welch per region and would otherwise recompute it once per
    channel per band."""
    if band is not None:
        f_low, f_high = band
        mask = (freqs >= f_low) & (freqs <= f_high)
        psd = psd[mask]
    psd_norm = psd / (psd.sum() + 1e-12)
    return -(psd_norm * np.log2(psd_norm + 1e-12)).sum()


def spectral_edge_frequency(freqs, psd, edge=0.9):
    """Frequency below which `edge` fraction of total spectral power falls."""
    cum_power = np.cumsum(psd)
    total = cum_power[-1] + 1e-12
    idx = np.searchsorted(cum_power, edge * total)
    idx = min(idx, len(freqs) - 1)
    return freqs[idx]


def phase_lag_index(sig1, sig2, sf, band):
    """PLI between two signals, band-limited to `band` = (f_low, f_high)."""
    f_low, f_high = band
    nyq = sf / 2.0
    b, a = butter(4, [f_low / nyq, f_high / nyq], btype="band")
    filt1 = filtfilt(b, a, sig1)
    filt2 = filtfilt(b, a, sig2)
    phase1 = np.angle(hilbert(filt1))
    phase2 = np.angle(hilbert(filt2))
    phase_diff = phase1 - phase2
    return float(np.abs(np.mean(np.sign(np.sin(phase_diff)))))


def _region_features(epoch_data, channel_idxs, sfreq):
    """
    Computes the regional feature vector (18-dim) for one region, and
    returns the region-averaged waveform for reuse by PLI (which needs one
    representative signal per region node, not per-channel signals).
    """
    sub = epoch_data[channel_idxs, :]  # (n_ch_in_region, n_samples)
    n_ch, n_samples = sub.shape
    region_signal = sub.mean(axis=0)

    freqs, psd = welch(sub, fs=sfreq, nperseg=n_samples)  # (n_ch, n_freqs)
    bp_list = []
    for f_low, f_high in BAND_RANGES.values():
        mask = (freqs >= f_low) & (freqs <= f_high)
        bp_list.append(np.mean(psd[:, mask], axis=1))
    band_powers = np.stack(bp_list, axis=1)         # (n_ch, 5)
    region_band_powers = band_powers.mean(axis=0)    # (5,)

    delta, theta, alpha, beta, gamma = [region_band_powers[i] + 1e-12 for i in range(5)]
    ratios = np.array([theta / alpha, delta / beta])

    hjorth_vals = np.array([hjorth_params(sub[ch]) for ch in range(n_ch)])  # (n_ch, 2)
    region_hjorth = hjorth_vals.mean(axis=0)

    # Reuse the region's already-computed per-channel PSD (`psd`, from the
    # welch() call above) instead of recomputing welch per channel per band.
    broadband_ent = np.mean([spectral_entropy_from_psd(freqs, psd[ch]) for ch in range(n_ch)])
    per_band_ent = np.array([
        np.mean([spectral_entropy_from_psd(freqs, psd[ch], band=b) for ch in range(n_ch)])
        for b in BAND_RANGES.values()
    ])

    region_psd_mean = psd.mean(axis=0)
    sef90 = spectral_edge_frequency(freqs, region_psd_mean, edge=0.9)

    feats = np.concatenate([
        region_band_powers, ratios, region_hjorth,
        [broadband_ent], per_band_ent, [sef90],
    ])
    return feats, region_signal


def compute_window_features(epoch_data, sfreq, include_asymmetry=True, include_pli=True):
    """
    epoch_data: (n_channels=22, n_samples)
    Returns a 1D feature vector:
      - 5 regions x 16 regional features            = 80  (always included)
      - 2 asymmetry pairs x 16 diff features         = 32  (if include_asymmetry)
      - 10 region-pair PLI (delta band)              = 10  (if include_pli)
    """
    region_feats, region_signals = {}, {}
    for region, idxs in REGIONS.items():
        feats, sig = _region_features(epoch_data, idxs, sfreq)
        region_feats[region] = feats
        region_signals[region] = sig

    blocks = [np.concatenate([region_feats[r] for r in REGION_NAMES])]

    if include_asymmetry:
        blocks.append(np.concatenate([
            region_feats[right] - region_feats[left]
            for right, left in ASYMMETRY_PAIRS
        ]))

    if include_pli:
        pli_vals = []
        for i in range(len(REGION_NAMES)):
            for j in range(i + 1, len(REGION_NAMES)):
                r1, r2 = REGION_NAMES[i], REGION_NAMES[j]
                pli_vals.append(phase_lag_index(region_signals[r1], region_signals[r2], sfreq, DELTA_BAND))
        blocks.append(np.array(pli_vals))

    return np.concatenate(blocks)


def feature_names_for(include_asymmetry=True, include_pli=True):
    names = [f"{region}__{feat}" for region in REGION_NAMES for feat in _PER_REGION_FEAT_NAMES]
    if include_asymmetry:
        names += [f"asym_{right}_minus_{left}__{feat}"
                  for right, left in ASYMMETRY_PAIRS for feat in _PER_REGION_FEAT_NAMES]
    if include_pli:
        names += [f"pli_delta__{REGION_NAMES[i]}__{REGION_NAMES[j]}"
                  for i in range(len(REGION_NAMES)) for j in range(i + 1, len(REGION_NAMES))]
    return names


GLOBAL_AVG_FEATURE_DIM = N_FEATS_PER_REGION
GLOBAL_AVG_FEATURE_NAMES = [f"global__{feat}" for feat in _PER_REGION_FEAT_NAMES]
_ALL_CHANNEL_IDXS = list(range(len(CHANNEL_NAMES)))


def compute_window_features_global_avg(epoch_data, sfreq):
    """
    epoch_data: (n_channels=22, n_samples)
    Same feature types as compute_window_features (band powers, ratios,
    Hjorth, broadband + per-band entropy, SEF90), but averaged across ALL
    22 channels into one global 16-dim vector -- no regional split, and
    consequently no hemispheric asymmetry (needs a left/right distinction)
    and no PLI connectivity (needs multiple distinct nodes to pair).
    """
    feats, _ = _region_features(epoch_data, _ALL_CHANNEL_IDXS, sfreq)
    return feats


def epoch_data_windows(data, sfreq, epoch_length=30.0):
    """Non-overlapping epoching -> list of (n_channels, n_samps_per_epoch)."""
    n_channels, n_times = data.shape
    samp_per_epoch = int(epoch_length * sfreq)
    epochs = []
    start = 0
    while start + samp_per_epoch <= n_times:
        epochs.append(data[:, start:start + samp_per_epoch])
        start += samp_per_epoch
    return epochs


def extract_window_features_for_npz(npz_path, sfreq, epoch_length, cache_dir=None, mode="regional",
                                     include_asymmetry=True, include_pli=True):
    """
    Loads a .npz, epochs into non-overlapping windows, computes the
    feature vector PER WINDOW (no pooling across windows). Returns
    (n_windows, feat_dim), or None if too short.

    mode="regional" (default): regional features, plus asymmetry/PLI blocks
    per include_asymmetry/include_pli. mode="global_avg": single global
    channel-averaged features (GLOBAL_AVG_FEATURE_DIM dims), no
    regions/asymmetry/PLI (include_asymmetry/include_pli ignored).

    Per-recording features only depend on the recording + (sfreq,
    epoch_length, mode, include_asymmetry, include_pli), never on which CV
    fold it's assigned to -- the same recording appears in the training set
    of 4 of 5 folds under k-fold CV, so without caching we'd recompute
    identical features up to 4x. When cache_dir is given, results are
    memoized to disk keyed on all of the above.
    """
    import os
    import hashlib
    if not os.path.isfile(npz_path):
        return None

    if mode == "regional":
        def compute_fn(w, sf):
            return compute_window_features(w, sf, include_asymmetry=include_asymmetry, include_pli=include_pli)
        variant_tag = f"regional_asym{int(include_asymmetry)}_pli{int(include_pli)}"
    else:
        compute_fn = compute_window_features_global_avg
        variant_tag = "global_avg"

    cache_path = None
    if cache_dir is not None:
        # Key on the absolute path (not just basename) so recordings with
        # the same filename under different data_root dirs (e.g. train vs.
        # baseline_test) can never collide in the cache. variant_tag is
        # included so different feature configs can never be mixed up even
        # if a caller points them at the same cache_dir by mistake.
        abspath = os.path.abspath(npz_path)
        digest = hashlib.md5(abspath.encode()).hexdigest()[:16]
        key = f"{os.path.basename(npz_path)}__{digest}__sf{sfreq}_el{epoch_length}_{variant_tag}.npy"
        cache_path = os.path.join(cache_dir, key)
        if os.path.isfile(cache_path):
            return np.load(cache_path)

    loaded = np.load(npz_path)
    data_array = loaded["data"]  # (n_channels, n_times)
    windows = epoch_data_windows(data_array, sfreq, epoch_length)
    if len(windows) == 0:
        return None
    feats = np.array([compute_fn(w, sfreq) for w in windows])

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        # Write-then-rename so a crash mid-save can't leave a truncated .npy
        # that a later run would load as valid cache. np.save always
        # appends ".npy" to whatever base path it's given.
        tmp_base = cache_path + f".tmp{os.getpid()}"
        np.save(tmp_base, feats)
        os.replace(tmp_base + ".npy", cache_path)

    return feats
