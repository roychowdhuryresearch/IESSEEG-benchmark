"""The montage/rate bridge between IESSEEG and EEGPT.

EEGPT expects a referential 10-20 montage at 250 Hz; IESSEEG stores the
same 19 electrodes with a `-REF` suffix at 200 Hz. As with the LUNA
adapter, a mistake here would not raise -- training would run on the
wrong channels and only the numbers would suffer.

These tests do not import braindecode, so they run in the benchmark's
main environment even though EEGPT itself runs from its own venv.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines", "eegpt"))

from eegpt_data import (  # noqa: E402
    EEGPT_CHANNELS, EEGPT_SAMPLE_RATE, SOURCE_SAMPLE_RATE, build_channel_map,
    make_recording_transform, normalize_channel_name,
)

# How the referential tree names its channels.
IESSEEG_CHANNELS = [f"{name}-REF" for name in EEGPT_CHANNELS]


def test_montage_is_the_standard_nineteen():
    assert len(EEGPT_CHANNELS) == 19
    assert len(set(EEGPT_CHANNELS)) == 19


@pytest.mark.parametrize("raw,expected", [
    ("FP1-REF", "FP1"), ("fp1-ref", "FP1"), ("T3-LE", "T3"),
    ("CZ", "CZ"), (" O2-REF ", "O2"),
])
def test_channel_name_normalisation(raw, expected):
    assert normalize_channel_name(raw) == expected


def test_channel_map_is_the_identity_for_our_tree():
    """Our tree already stores the 19 in EEGPT's order, so the mapping
    should be a straight pass-through rather than a reshuffle."""
    indices = build_channel_map(IESSEEG_CHANNELS)
    assert list(indices) == list(range(19))


def test_channel_map_handles_a_reordered_montage():
    shuffled = list(reversed(IESSEEG_CHANNELS))
    indices = build_channel_map(shuffled)
    for target, index in zip(EEGPT_CHANNELS, indices):
        assert normalize_channel_name(shuffled[index]) == target


def test_missing_channel_fails_loudly():
    with pytest.raises(ValueError, match="missing EEGPT channels"):
        build_channel_map(IESSEEG_CHANNELS[:-1])


def test_recording_transform_selects_and_resamples():
    rng = np.random.default_rng(0)
    seconds = 8
    raw = rng.normal(0, 50, size=(19, SOURCE_SAMPLE_RATE * seconds))

    transform = make_recording_transform(standardize=False)
    out = transform(raw, IESSEEG_CHANNELS)

    assert out.shape == (19, EEGPT_SAMPLE_RATE * seconds)


def test_recording_transform_reorders_channels():
    """Give each channel a constant, distinct value and check the output
    row order follows EEGPT's montage rather than the source order."""
    shuffled = list(reversed(IESSEEG_CHANNELS))
    raw = np.stack([
        np.full(SOURCE_SAMPLE_RATE * 4, float(i)) for i in range(19)
    ])

    transform = make_recording_transform(standardize=False)
    out = transform(raw, shuffled)

    indices = build_channel_map(shuffled)
    middle = out[:, EEGPT_SAMPLE_RATE : EEGPT_SAMPLE_RATE * 3]
    # Polyphase resampling leaves a small ripple on a constant signal, so
    # the tolerance only has to be tight enough to tell neighbouring
    # channels apart -- their values differ by 1.0.
    for row, source_index in zip(middle, indices):
        assert np.allclose(row, float(source_index), atol=0.1)


def test_window_length_matches_pretraining_context():
    """EEGPT's pre-trained context is 4 s at 250 Hz."""
    assert EEGPT_SAMPLE_RATE * 4 == 1000


def test_resampling_ratio_is_exact():
    from math import gcd
    divisor = gcd(EEGPT_SAMPLE_RATE, SOURCE_SAMPLE_RATE)
    assert (EEGPT_SAMPLE_RATE // divisor, SOURCE_SAMPLE_RATE // divisor) == (5, 4)
