"""Each model's dataset module must bind that model's window transform.

The per-model `inmem_raw_dataset.py` modules are shims over the shared
dataset whose only job is to apply the right window policy by default. If
one silently bound the wrong transform, training would still run and the
numbers would just be wrong, so the bindings are pinned here.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

EXPECTED_TRANSFORM = {
    "biot": "quantile",
    "cbramod": "patch",
    "labram": "none",
}

SAMPLE_RATE = 200
N_CHANNELS = 19


def load_model_dataset_module(model):
    path = os.path.join(REPO_ROOT, "baselines", model, "inmem_raw_dataset.py")
    spec = importlib.util.spec_from_file_location(f"{model}_inmem", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recording(tmp_path):
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "rec0.npz",
        data=rng.normal(0, 50, size=(N_CHANNELS, 60 * SAMPLE_RATE)),
        channel=np.array([f"ch{c}" for c in range(N_CHANNELS)]),
    )
    return str(tmp_path), [("p0", "rec0", 1)]


@pytest.mark.parametrize("model,transform", sorted(EXPECTED_TRANSFORM.items()))
def test_module_declares_expected_transform(model, transform):
    assert load_model_dataset_module(model).WINDOW_TRANSFORM == transform


@pytest.mark.parametrize("model,transform", sorted(EXPECTED_TRANSFORM.items()))
def test_dataset_applies_transform_without_being_asked(model, transform, recording):
    """The model scripts construct the dataset without naming a transform,
    so the default binding is what actually shapes their input."""
    data_dir, info = recording
    module = load_model_dataset_module(model)
    dataset = module.InMemoryRandomDataset(
        data_dir=data_dir, info_list=info, mode="test", sample_rate=SAMPLE_RATE,
        window_sec=30, step_sec=30, n_channels=N_CHANNELS, scale_factor=1.0,
    )
    assert dataset.window_transform == transform

    window = dataset[0]["waveform"]
    if transform == "patch":
        assert window.shape == (N_CHANNELS, 30, SAMPLE_RATE)
    else:
        assert window.shape == (N_CHANNELS, 30 * SAMPLE_RATE)

    if transform == "quantile":
        q95 = np.quantile(np.abs(window), 0.95, axis=-1)
        assert np.allclose(q95, 1.0, atol=1e-3)


@pytest.mark.parametrize("model", sorted(EXPECTED_TRANSFORM))
def test_explicit_transform_still_wins(model, recording):
    """The default is a default, not an override."""
    data_dir, info = recording
    module = load_model_dataset_module(model)
    dataset = module.InMemoryRandomDataset(
        data_dir=data_dir, info_list=info, mode="test", sample_rate=SAMPLE_RATE,
        window_sec=30, step_sec=30, n_channels=N_CHANNELS, scale_factor=1.0,
        window_transform="none",
    )
    assert dataset.window_transform == "none"


@pytest.mark.parametrize("model", sorted(EXPECTED_TRANSFORM))
def test_legacy_loader_name_still_resolves(model):
    """Older scripts imported the loader as load_data_by_recording_id."""
    module = load_model_dataset_module(model)
    assert module.load_data_by_recording_id is module.load_recording


def test_every_model_binds_a_distinct_known_transform():
    from iesseeg.data.raw_dataset import WINDOW_TRANSFORMS

    assert set(EXPECTED_TRANSFORM.values()) <= set(WINDOW_TRANSFORMS)
    for model, transform in EXPECTED_TRANSFORM.items():
        assert load_model_dataset_module(model).WINDOW_TRANSFORM in WINDOW_TRANSFORMS
