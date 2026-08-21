"""Metrics and aggregation behaviour, exercised on synthetic runs.

No EEG data or trained checkpoints needed: these build small prediction
CSVs on disk and check what the scorer and the aggregator make of them.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iesseeg.evaluation import infer_task_from_path  # noqa: E402
from iesseeg.evaluation.aggregate import collect, summarize  # noqa: E402
from iesseeg.evaluation.metrics import evaluate_run  # noqa: E402


def write_run(directory, y_true, y_pred, y_prob=None, label_column="known_label"):
    os.makedirs(directory, exist_ok=True)
    frame = {
        "short_recording_id": [str(i) for i in range(len(y_true))],
        label_column: y_true,
        "pred_label": y_pred,
    }
    if y_prob is not None:
        frame["pred_prob"] = y_prob
    path = os.path.join(directory, "inference_results.csv")
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def test_infers_task_from_kfold_directory_name(tmp_path):
    assert infer_task_from_path(str(tmp_path / "case_control_fold0" / "x.csv")) == "case_control"
    assert infer_task_from_path(str(tmp_path / "immediate_responder_fold3" / "x.csv")) == "immediate_responder"
    assert infer_task_from_path(str(tmp_path / "meaningful_responder_fold4" / "x.csv")) == "meaningful_responder"


def test_infers_task_from_legacy_two_fold_directory_name(tmp_path):
    """The original release named runs "A_case_vs_control"; older result
    trees should still score."""
    assert infer_task_from_path(str(tmp_path / "A_case_vs_control" / "x.csv")) == "case_control"


def test_rejects_unrecognised_directory_name(tmp_path):
    with pytest.raises(ValueError, match="Could not determine task"):
        infer_task_from_path(str(tmp_path / "some_other_run" / "x.csv"))


def test_perfect_predictions_score_one(tmp_path):
    run_dir = tmp_path / "immediate_responder_fold0"
    y = [0, 0, 1, 1, 0, 1]
    path = write_run(str(run_dir), y, y, y_prob=[float(v) for v in y])
    metrics = evaluate_run(path, str(run_dir), human_label_meta=None)

    assert float(metrics["balanced_accuracy"]) == 1.0
    assert float(metrics["f1"]) == 1.0
    assert float(metrics["auroc"]) == 1.0
    assert os.path.isfile(run_dir / "metrics.json")
    assert os.path.isfile(run_dir / "confusion_matrix.png")


def test_balanced_accuracy_handles_class_imbalance(tmp_path):
    """Predicting the majority class everywhere scores 0.5, not the
    majority share -- the reason the benchmark reports balanced accuracy."""
    run_dir = tmp_path / "meaningful_responder_fold1"
    y_true = [1] * 8 + [0] * 2
    path = write_run(str(run_dir), y_true, [1] * 10, y_prob=[0.9] * 10)
    metrics = evaluate_run(path, str(run_dir), human_label_meta=None)
    assert float(metrics["balanced_accuracy"]) == 0.5


def test_single_class_fold_records_auroc_as_missing(tmp_path):
    """AUROC is undefined when a fold's held-out subjects share one label;
    it must be recorded as missing rather than crashing the run."""
    run_dir = tmp_path / "immediate_responder_fold2"
    path = write_run(str(run_dir), [1, 1, 1, 1], [1, 0, 1, 1], y_prob=[0.9, 0.2, 0.8, 0.7])
    metrics = evaluate_run(path, str(run_dir), human_label_meta=None)
    assert metrics["auroc"] is None
    assert metrics["balanced_accuracy"] is not None


def test_case_control_scores_against_expert_clip_labels(tmp_path):
    """Task 1 ground truth is the clip-level expert consensus, not the
    subject-level diagnosis carried in known_label."""
    run_dir = tmp_path / "case_control_fold0"
    os.makedirs(run_dir, exist_ok=True)

    # known_label disagrees with the expert labels on every clip, so
    # scoring against the right column is unambiguous.
    pd.DataFrame({
        "short_recording_id": ["1", "2", "3", "4"],
        "known_label": [1, 1, 1, 1],
        "pred_label": [1, 0, 1, 0],
        "pred_prob": [0.9, 0.1, 0.8, 0.2],
    }).to_csv(run_dir / "inference_results.csv", index=False)

    human_meta = tmp_path / "final_test.csv"
    pd.DataFrame({
        "short_recording_id": ["1", "2", "3", "4"],
        "human_label": [1, 0, 1, 0],
    }).to_csv(human_meta, index=False)

    metrics = evaluate_run(
        str(run_dir / "inference_results.csv"), str(run_dir), str(human_meta)
    )
    assert float(metrics["balanced_accuracy"]) == 1.0


def test_aggregate_reports_mean_and_std_across_folds(tmp_path):
    results_root = tmp_path / "biot" / "result" / "inference"
    accuracies = [0.6, 0.7, 0.8, 0.9, 1.0]
    for fold, accuracy in enumerate(accuracies):
        run_dir = results_root / f"case_control_fold{fold}"
        os.makedirs(run_dir, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as handle:
            json.dump({"balanced_accuracy": f"{accuracy:.3f}",
                       "f1": "0.500", "auroc": "0.500"}, handle)

    df = collect(str(tmp_path), ["biot"])
    assert len(df) == 5

    summary = summarize(df).iloc[0]
    assert summary["balanced_accuracy_count"] == 5
    assert summary["balanced_accuracy_mean"] == pytest.approx(np.mean(accuracies))
    assert summary["balanced_accuracy_std"] == pytest.approx(np.std(accuracies, ddof=1))


def test_aggregate_skips_runs_without_metrics(tmp_path):
    """A fold that was inferred but never scored must not be silently
    counted as if it had been."""
    run_dir = tmp_path / "biot" / "result" / "inference" / "case_control_fold0"
    os.makedirs(run_dir, exist_ok=True)
    write_run(str(run_dir), [0, 1], [0, 1])

    assert collect(str(tmp_path), ["biot"]).empty
