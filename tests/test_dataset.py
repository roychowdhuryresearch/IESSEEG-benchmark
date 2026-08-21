"""Windowing and per-model transform behaviour of the shared dataset.

Uses synthetic .npz recordings, so no real EEG is required. The transforms
are the one place the three former per-model dataset copies differed, so
they are pinned here.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iesseeg.data.raw_dataset import InMemoryRandomDataset  # noqa: E402

SAMPLE_RATE = 200
N_CHANNELS = 19


@pytest.fixture
def recordings(tmp_path):
    """Three 60-second recordings with deterministic content."""
    rng = np.random.default_rng(0)
    info = []
    for i in range(3):
        data = rng.normal(0, 50, size=(N_CHANNELS, 60 * SAMPLE_RATE))
        np.savez(
            tmp_path / f"rec{i}.npz",
            data=data,
            channel=np.array([f"ch{c}" for c in range(N_CHANNELS)]),
        )
        info.append((f"patient{i}", f"rec{i}", i % 2))
    return str(tmp_path), info


def build(data_dir, info, **kwargs):
    params = dict(
        data_dir=data_dir, info_list=info, mode="test", sample_rate=SAMPLE_RATE,
        window_sec=30, step_sec=30, n_channels=N_CHANNELS, scale_factor=1.0,
    )
    params.update(kwargs)
    return InMemoryRandomDataset(**params)


def test_tiles_recordings_into_non_overlapping_windows(recordings):
    data_dir, info = recordings
    dataset = build(data_dir, info)
    # 60 s per recording / 30 s windows = 2 windows each.
    assert len(dataset) == 3 * 2
    assert dataset[0]["waveform"].shape == (N_CHANNELS, 30 * SAMPLE_RATE)


def test_window_order_is_deterministic_in_test_mode(recordings):
    """Clip-level probability averaging depends on evaluation windows
    coming out in a fixed order."""
    data_dir, info = recordings
    first = [tuple(build(data_dir, info)[i]["waveform"].ravel()[:4]) for i in range(6)]
    second = [tuple(build(data_dir, info)[i]["waveform"].ravel()[:4]) for i in range(6)]
    assert first == second


def test_transform_none_returns_raw_window(recordings):
    data_dir, info = recordings
    dataset = build(data_dir, info, window_transform="none")
    window = dataset[0]["waveform"]
    assert window.shape == (N_CHANNELS, 30 * SAMPLE_RATE)
    # Amplitudes are untouched, so the raw microvolt scale survives.
    assert np.abs(window).max() > 1.0


def test_transform_quantile_normalises_per_channel(recordings):
    """BIOT's policy: each channel divided by its own 95th-percentile
    amplitude, so per-channel scale is ~1 regardless of input gain."""
    data_dir, info = recordings
    dataset = build(data_dir, info, window_transform="quantile")
    window = dataset[0]["waveform"]

    assert window.shape == (N_CHANNELS, 30 * SAMPLE_RATE)
    per_channel_q95 = np.quantile(np.abs(window), 0.95, axis=-1)
    assert np.allclose(per_channel_q95, 1.0, atol=1e-3)


def test_transform_quantile_is_invariant_to_input_gain(recordings):
    data_dir, info = recordings
    unscaled = build(data_dir, info, window_transform="quantile", scale_factor=1.0)[0]
    scaled = build(data_dir, info, window_transform="quantile", scale_factor=0.01)[0]
    assert np.allclose(unscaled["waveform"], scaled["waveform"])


def test_transform_patch_reshapes_into_one_second_patches(recordings):
    """CBraMod's policy: (C, T) -> (C, seconds, samples_per_second)."""
    data_dir, info = recordings
    dataset = build(data_dir, info, window_transform="patch")
    window = dataset[0]["waveform"]
    assert window.shape == (N_CHANNELS, 30, SAMPLE_RATE)

    flat = build(data_dir, info, window_transform="none")[0]["waveform"]
    assert np.array_equal(window.reshape(N_CHANNELS, -1), flat)


def test_unknown_transform_is_rejected(recordings):
    data_dir, info = recordings
    with pytest.raises(ValueError, match="Unknown window_transform"):
        build(data_dir, info, window_transform="not_a_transform")


def test_return_dict_false_yields_tuple(recordings):
    data_dir, info = recordings
    dataset = build(data_dir, info, return_dict=False)
    window, label = dataset[0]
    assert window.shape == (N_CHANNELS, 30 * SAMPLE_RATE)
    assert label in (0, 1)


def test_recordings_shorter_than_one_window_are_dropped(tmp_path):
    """A clip too short to fill a window contributes nothing rather than
    producing a truncated, silently misaligned window."""
    np.savez(
        tmp_path / "short.npz",
        data=np.zeros((N_CHANNELS, 10 * SAMPLE_RATE)),
        channel=np.array([f"ch{c}" for c in range(N_CHANNELS)]),
    )
    dataset = build(str(tmp_path), [("p", "short", 1)])
    assert len(dataset) == 0


def test_missing_recording_is_skipped(recordings):
    data_dir, info = recordings
    dataset = build(data_dir, info + [("pX", "does_not_exist", 1)])
    assert len(dataset) == 6


def test_collate_stacks_windows_and_keeps_identifiers(recordings):
    data_dir, info = recordings
    dataset = build(data_dir, info)
    batch = InMemoryRandomDataset.collate_fn([dataset[0], dataset[1], dataset[2]])

    assert batch["waveform"].shape == (3, N_CHANNELS, 30 * SAMPLE_RATE)
    assert batch["label"].shape == (3,)
    assert len(batch["recording_id"]) == 3
    assert len(batch["patient_id"]) == 3


def test_train_mode_length_is_the_iteration_budget(recordings):
    """Training draws random windows, so an epoch is a fixed number of
    draws rather than a pass over a fixed index."""
    data_dir, info = recordings
    dataset = build(data_dir, info, mode="train", train_iterations=128)
    assert len(dataset) == 128
    assert dataset[0]["waveform"].shape == (N_CHANNELS, 30 * SAMPLE_RATE)
