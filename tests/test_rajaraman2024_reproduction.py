"""Checks for the published response-feature reproduction.

These tests use synthetic arrays and metadata only. They do not require EEG,
restricted response data, a model checkpoint, or a GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    path = ROOT / "analysis" / "response_features" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


extract = load_script("rajaraman_extract", "extract_rajaraman2024_raw_features.py")
reproduce = load_script("rajaraman_reproduce", "reproduce_rajaraman2024.py")


def synthetic_metadata() -> pd.DataFrame:
    rows = []
    recording = 0
    for patient in range(50):
        response = "Responder" if patient < 28 else "Non-responder"
        for condition in ("PRE", "POST"):
            for state in ("AWAKE", "SLEEP"):
                for _ in range(2):
                    recording += 1
                    rows.append(
                        {
                            "patient_id": patient,
                            "short_recording_id": f"SYN{recording:04d}",
                            "case_control_label": "CASE",
                            "pre_post_treatment_label": condition,
                            "sleep_awake_label": state,
                            "meaningful_responder": response,
                            "LeadtimeUKISS": patient % 3,
                        }
                    )
    return pd.DataFrame(rows)


def test_target_rows_match_the_three_required_clip_cells():
    metadata = synthetic_metadata()
    assert len(extract.target_rows(metadata, "beta")) == 300
    assert len(extract.target_rows(metadata, "pli")) == 100
    assert len(extract.target_rows(metadata, "all")) == 300


def test_matlab_hist_entropy_handles_constant_and_balanced_signals():
    signal = np.array([[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    observed = extract.matlab_hist_entropy(signal, bins=2)
    np.testing.assert_allclose(observed, [0.0, 1.0])


def test_fast_fluctuation_function_matches_direct_linear_detrending():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=100)
    widths = np.array([8, 13, 21])
    observed = extract.fluctuation_function(signal, widths)

    profile = np.cumsum(signal - signal.mean())
    expected = []
    for width in widths:
        deviations = []
        for start in range(0, len(profile) - width, max(1, round(width * 0.5))):
            segment = profile[start : start + width + 1]
            fitted = np.polyval(np.polyfit(np.arange(width + 1), segment, 1), np.arange(width + 1))
            deviations.append(np.sqrt(np.mean((segment - fitted) ** 2)))
        expected.append(np.median(deviations))
    np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-11)


def test_raw_feature_loader_averages_two_clips_per_patient(tmp_path):
    metadata = synthetic_metadata()
    metadata_path = tmp_path / "metadata.csv"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    metadata.to_csv(metadata_path, index=False)

    for row in metadata.itertuples(index=False):
        target = (row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE") or (
            row.pre_post_treatment_label == "POST"
        )
        if not target:
            continue
        payload = {"clean_fraction": 0.9}
        if row.sleep_awake_label == "AWAKE":
            payload["beta_dfa_intercept_mean"] = -0.2
        if row.pre_post_treatment_label == "POST" and row.sleep_awake_label == "SLEEP":
            payload["beta_entropy_mean"] = 6.0
        if row.pre_post_treatment_label == "PRE":
            payload["connectivity_percent_retained_pli"] = 5.0
            payload["connectivity_percent_significance_frequency"] = 7.0
        (feature_dir / f"{row.short_recording_id}.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )

    patients = reproduce.load_raw_patient_features(feature_dir, metadata_path)
    assert len(patients) == 50
    assert patients.label.value_counts().to_dict() == {1: 28, 0: 22}
    np.testing.assert_allclose(patients.dfa_intercept_beta_pre_awake, -0.2)
    np.testing.assert_allclose(patients.r0_raw, -2.361 * -0.2 - 0.051 * 5.0)
    np.testing.assert_allclose(patients.r1_raw, 4.765 * 6.0 - 7.786 * -0.2)
