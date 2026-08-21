"""
Legacy per-channel EEG feature extraction (pre-regional-grouping baseline).
Kept only so --feature_set flat/avg remain available for comparison against
the new "regional" feature set in features.py.
"""

import numpy as np
from scipy.signal import welch

BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 50),
}


def spectral_entropy(signal, sf, band=None):
    freqs, psd = welch(signal, sf, nperseg=len(signal))
    if band is not None:
        f_low, f_high = band
        mask = (freqs >= f_low) & (freqs <= f_high)
        psd = psd[mask]
    psd_norm = psd / (psd.sum() + 1e-12)
    return -(psd_norm * np.log2(psd_norm + 1e-12)).sum()


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


def compute_epoch_features(epoch_data, sfreq, avg_channels=False):
    """epoch_data: (n_channels, n_samples). Returns flattened (n_channels*10)
    or channel-averaged (10,) feature vector."""
    n_channels, n_samples = epoch_data.shape
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=n_samples)

    bp_list = []
    for f_low, f_high in BAND_RANGES.values():
        mask = (freqs >= f_low) & (freqs <= f_high)
        bp_list.append(np.mean(psd[:, mask], axis=1))
    band_powers = np.stack(bp_list, axis=1)  # (n_channels, 5)

    delta, theta, alpha, beta, gamma = [band_powers[:, i] + 1e-12 for i in range(5)]
    ratio_arr = np.stack([theta / alpha, delta / beta], axis=1)

    hjorth_mob, hjorth_comp, spec_ents = [], [], []
    for ch_idx in range(n_channels):
        mob, comp = hjorth_params(epoch_data[ch_idx, :])
        hjorth_mob.append(mob)
        hjorth_comp.append(comp)
        spec_ents.append(spectral_entropy(epoch_data[ch_idx, :], sfreq))

    combined = np.concatenate([
        band_powers, ratio_arr,
        np.array(hjorth_mob)[:, None],
        np.array(hjorth_comp)[:, None],
        np.array(spec_ents)[:, None],
    ], axis=1)  # (n_channels, 10)

    if avg_channels:
        return combined.mean(axis=0)
    return combined.flatten()


def epoch_data(data, sfreq, epoch_length=10.0):
    n_channels, n_times = data.shape
    samp_per_epoch = int(epoch_length * sfreq)
    epochs = []
    start = 0
    while start + samp_per_epoch <= n_times:
        epochs.append(data[:, start:start + samp_per_epoch])
        start += samp_per_epoch
    return epochs
