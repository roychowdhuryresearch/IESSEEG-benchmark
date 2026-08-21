# BASED alignment analysis

Do models trained on IESSEEG's case/control labels encode the severity
that the field's reference scale, the BASED score, measures?

The short answer is no, and the way that answer is reached is the point.
Pooled across all expert-scored epochs, a model's case probability
correlates with BASED (LUNA rho=+0.42). Stratifying by treatment
condition removes almost all of it (LUNA rho=-0.07 within pre-treatment,
-0.08 within post-treatment). The pooled correlation was carried by the
separation between pre- and post-treatment recordings, not by severity.

## Inputs (site-specific, not part of the release)

These read clinical data that is not redistributed here:

```bash
export IESSEEG_LONG_EDF_ROOT=/path/to/long/edf        # source recordings
export IESSEEG_BASED_EPOCH_PAIRS=/path/to/pairs.csv   # rater epochs + BASED scores
export IESSEEG_SUBJECT_META=/path/to/subject_meta.csv # recording -> patient
export IESSEEG_BENCH_ROOT=/path/to/this/repo          # defaults to the repo root
```

`epoch_pairs.csv` needs one row per (recording, rater) with columns
`recording_id, rater, role, based, offset_sec`, where `offset_sec` is the
position in the recording of the 5-minute window that rater scored.

## Pipeline

```bash
# 1. Pull the exact epochs the raters scored, in each model's montage.
python extract_based_epochs.py --out_dir /path/to/epochs

# 2. Score them with the fold-appropriate model and save embeddings.
#    EEGPT needs its own interpreter (see baselines/eegpt/README.md).
python score_based_epochs.py --epoch_dir /path/to/epochs --models luna reve --out_dir results
$IESSEEG_PYTHON_EEGPT score_based_epochs.py --epoch_dir /path/to/epochs --models eegpt --out_dir results

# 3. Statistics and the paper figure.
python analyze_based.py --results_dir results
python based_figure.py
```

## Two things that matter for correctness

**Leakage.** Every recording belongs to a patient held out in exactly one
cross-validation fold, and each epoch is scored with that fold's model.
No epoch is scored by a model that trained on its patient.

**Stratification.** Any statistic relating model output to BASED is
reported pooled *and* within treatment condition. The pooled number alone
is misleading, for the reason above.

Outputs land in `results/` and are gitignored: they are embeddings and
scores derived from patient recordings, not release artifacts.
