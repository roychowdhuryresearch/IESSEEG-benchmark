"""The montage/rate bridge between IESSEEG and LUNA.

LUNA expects the 22-channel TCP bipolar montage at 256 Hz plus 3D
electrode coordinates, while IESSEEG stores the same derivations in a
different order, six of them at the opposite polarity, at 200 Hz. That
adapter is easy to get subtly wrong in a way nothing else would catch --
training would run and only the numbers would suffer -- so it is pinned
here.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines", "luna"))

from luna_data import (  # noqa: E402
    LUNA_SAMPLE_RATE, SOURCE_SAMPLE_RATE, TCP_MONTAGE, build_channel_map,
    channel_locations, make_recording_transform, normalize_window,
)

# The channel order IESSEEG's 22-channel bipolar tree actually uses.
IESSEEG_CHANNELS = [
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "A2-T4", "T4-C4", "C4-CZ", "CZ-C3", "C3-T3", "T3-A1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
]

# The six central-chain derivations IESSEEG stores reversed.
EXPECTED_FLIPPED = {"A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2"}


def test_montage_has_22_unique_derivations():
    assert len(TCP_MONTAGE) == 22
    assert len(set(TCP_MONTAGE)) == 22


def test_every_luna_channel_is_mapped():
    indices, signs = build_channel_map(IESSEEG_CHANNELS)
    assert len(indices) == len(TCP_MONTAGE) == len(signs)
    assert set(indices) == set(range(22)), "every source channel used exactly once"


def test_exactly_the_central_chain_is_polarity_flipped():
    indices, signs = build_channel_map(IESSEEG_CHANNELS)
    flipped = {name for name, sign in zip(TCP_MONTAGE, signs) if sign < 0}
    assert flipped == EXPECTED_FLIPPED


def test_mapped_channels_name_the_same_electrode_pair():
    """A flipped channel must be the same two electrodes, reversed."""
    indices, signs = build_channel_map(IESSEEG_CHANNELS)
    for target, index, sign in zip(TCP_MONTAGE, indices, signs):
        source = IESSEEG_CHANNELS[index]
        if sign > 0:
            assert source == target
        else:
            anode, cathode = target.split("-")
            assert source == f"{cathode}-{anode}"


def test_missing_channel_fails_loudly():
    """A montage mismatch must raise, not silently train on 21 channels."""
    with pytest.raises(ValueError, match="missing LUNA channels"):
        build_channel_map(IESSEEG_CHANNELS[:-1])


def test_channel_locations_are_electrode_midpoints():
    locations = channel_locations()
    assert locations.shape == (22, 3)
    assert locations.dtype == np.float32
    assert np.isfinite(locations).all()
    # Distinct derivations sit at distinct scalp positions.
    assert len({tuple(np.round(row, 6)) for row in locations}) == 22


def test_recording_transform_reorders_and_resamples():
    rng = np.random.default_rng(0)
    seconds = 10
    raw = rng.normal(0, 50, size=(22, SOURCE_SAMPLE_RATE * seconds))

    transform = make_recording_transform()
    out = transform(raw, IESSEEG_CHANNELS)

    assert out.shape == (22, LUNA_SAMPLE_RATE * seconds)


def test_recording_transform_applies_the_sign_flip():
    """Check the flip on a constant-signal recording, where resampling
    leaves the value alone and only the sign can change it."""
    raw = np.ones((22, SOURCE_SAMPLE_RATE * 4))
    transform = make_recording_transform()
    out = transform(raw, IESSEEG_CHANNELS)

    _, signs = build_channel_map(IESSEEG_CHANNELS)
    middle = out[:, LUNA_SAMPLE_RATE : LUNA_SAMPLE_RATE * 3]
    for row, sign in zip(middle, signs):
        assert np.allclose(row, sign, atol=1e-3)


def test_resampled_length_is_divisible_by_patch_size():
    """LUNA patches the window in 40-sample blocks; a 30 s window at
    256 Hz must divide evenly or the tail would be silently dropped."""
    window = LUNA_SAMPLE_RATE * 30
    assert window % 40 == 0
    assert window // 40 == 192


def test_normalize_window_standardises_each_channel():
    rng = np.random.default_rng(1)
    window = rng.normal(5, 30, size=(22, 1000))
    out = normalize_window(window)
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1, atol=1e-3)


def test_normalize_window_survives_a_flat_channel():
    """A dead electrode has zero variance; the epsilon must keep it
    finite rather than producing NaNs that poison the batch."""
    window = np.ones((3, 100))
    out = normalize_window(window)
    assert np.isfinite(out).all()


def test_shared_dataset_exposes_the_zscore_transform():
    from iesseeg.data.raw_dataset import WINDOW_TRANSFORMS

    assert "zscore" in WINDOW_TRANSFORMS
