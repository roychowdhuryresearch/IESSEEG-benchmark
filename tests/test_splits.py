"""Protocol invariants for the released fold manifests.

These guard the properties that make the benchmark comparable across
groups. They read only the manifests shipped in splits/, so they run
without the EEG data present.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iesseeg import config  # noqa: E402

TASKS = list(config.TASKS)
FOLDS = list(range(config.N_FOLDS))

EXPECTED_SUBJECTS = {
    "case_control": 100,
    "immediate_responder": 50,
    "meaningful_responder": 50,
}


def manifest(task):
    return pd.read_csv(config.fold_manifest(task))


@pytest.mark.parametrize("task", TASKS)
def test_manifest_covers_every_subject_once(task):
    df = manifest(task)
    assert len(df) == EXPECTED_SUBJECTS[task]
    assert df["patient_id"].is_unique
    assert sorted(df["fold"].unique()) == FOLDS


@pytest.mark.parametrize("task", TASKS)
def test_folds_are_balanced_in_size(task):
    """Fold sizes may differ by at most one subject."""
    sizes = manifest(task).groupby("fold").size()
    assert sizes.max() - sizes.min() <= 1


@pytest.mark.parametrize("task", TASKS)
def test_folds_are_stratified(task):
    """Each fold's minority-class share stays close to the pooled share.

    Stratification cannot be exact when the minority count does not divide
    evenly across folds, so this allows one subject of slack per fold
    rather than demanding identical proportions.
    """
    df = manifest(task)
    minority = df["label"].value_counts().idxmin()
    pooled_share = (df["label"] == minority).mean()

    for fold in FOLDS:
        fold_df = df[df["fold"] == fold]
        share = (fold_df["label"] == minority).mean()
        slack = 1.0 / len(fold_df)
        assert abs(share - pooled_share) <= slack + 1e-9, (
            f"{task} fold {fold}: minority share {share:.3f} vs pooled "
            f"{pooled_share:.3f}"
        )


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fold", FOLDS)
def test_no_patient_leakage(task, fold):
    """The invariant a subject-wise benchmark exists to enforce."""
    train = pd.read_csv(config.fold_csv(task, fold, "train"))
    test = pd.read_csv(config.fold_csv(task, fold, "test"))
    assert not (set(train["patient_id"]) & set(test["patient_id"]))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fold", FOLDS)
def test_test_side_matches_manifest(task, fold):
    """A fold's test subjects are exactly the subjects assigned to it."""
    assigned = set(manifest(task).query("fold == @fold")["patient_id"])
    in_test = set(pd.read_csv(config.fold_csv(task, fold, "test"))["patient_id"])
    assert in_test == assigned


@pytest.mark.parametrize("task", TASKS)
def test_every_subject_is_tested_exactly_once(task):
    """Across the five folds each subject is held out exactly once, so the
    pooled test predictions cover the cohort without double-counting."""
    seen = []
    for fold in FOLDS:
        seen += list(pd.read_csv(config.fold_csv(task, fold, "test"))["patient_id"].unique())
    assert len(seen) == len(set(seen)) == EXPECTED_SUBJECTS[task]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fold", FOLDS)
def test_labels_are_encodable(task, fold):
    """Both splits carry the task's label column with only known values."""
    from iesseeg.data.splits import encode_labels

    column = config.TASK_LABEL_COLUMN[task]
    for split in ("train", "test"):
        df = pd.read_csv(config.fold_csv(task, fold, split))
        assert column in df.columns
        assert set(encode_labels(df, column)) <= {0, 1}
