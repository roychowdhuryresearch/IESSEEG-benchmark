# Published response-feature reproduction

This directory independently reimplements the PRE- and POST-treatment EEG
features used by Rajaraman et al. (2024) and compares the resulting
patient-level scores with the values reported in that article. It is a
same-cohort method reproduction, not an external validation study.

The analysis needs two local inputs that are not distributed here:

1. the source Clinical Clip EDF files; and
2. a governed clip-metadata CSV containing `patient_id`,
   `short_recording_id`, `case_control_label`, `pre_post_treatment_label`,
   `sleep_awake_label`, `meaningful_responder`, and `LeadtimeUKISS`.

The metadata-to-patient linkage must remain local. Both scripts therefore
require `--metadata-csv`, and generated patient-level files are written under
the Git-ignored `local_results/rajaraman2024/` directory.

## Run

From the repository root:

```bash
source /home/mingjian/miniforge3/etc/profile.d/mamba.sh
mamba activate seeg

# PLI extraction requires CUDA and exits if CUDA is unavailable.
CUDA_VISIBLE_DEVICES=3 python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features pli \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs

# DFA and entropy extraction run on CPU.
python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features beta \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs

python analysis/response_features/reproduce_rajaraman2024.py \
  --source raw \
  --metadata-csv /path/to/local_clip_metadata.csv
```

Extraction is resumable: each completed clip is cached as one local JSON file.
The final script averages the two matching clips for each patient and writes
the distribution and score comparisons. The reviewed, de-identified aggregate
tables from the completed run are in `reference_results/`.

For PRE connectivity, clean EEG is divided into non-overlapping eight-second
epochs without joining samples across artifact gaps. Raw delta-band PLI is
averaged across epochs for each of the 171 electrode pairs. The source
clip-level quantity `C0` is the percentage of pair means above 0.20. The
surrogate-thresholded outputs are retained only as diagnostics and are not used
in the published `R0` score.

To test the source quantities before averaging each patient's two clips:

```bash
python analysis/response_features/analyze_response_feature_associations.py \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --raw-features-dir local_results/rajaraman2024/raw_features
```

This analysis evaluates both immediate and sustained response. It first
computes an AUROC over the 100 individual clips in each relevant
condition/state cell. It then combines one POST awake clip and one POST sleep
clip in all four possible within-patient pairings, and shows the source
patient-averaged scores for reference. Its significance test shuffles labels
across the 50 patients while keeping every value from one patient together.
The committed table reports patient-clustered permutation P-values and Holm
correction across the eight inspected quantities within each endpoint.
The executed, reader-oriented version is
[`notebooks/clip_and_state_response_associations.ipynb`](notebooks/clip_and_state_response_associations.ipynb).

## Complete condition/state/feature grid

The source reproduction intentionally contains only the four qEEG cells used
by the published R0 and R1 scores. To examine the omitted combinations, extract
DFA, entropy, and raw PLI for every PRE/POST and awake/sleep clip:

```bash
CUDA_VISIBLE_DEVICES=3 python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features all --cell-scope all --n-surrogates 0 \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs

python analysis/response_features/analyze_full_qeeg_response_grid.py \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --raw-features-dir local_results/rajaraman2024/raw_features
```

This produces 24 tests per response endpoint: 12 condition/state/feature cells
evaluated once per clip and after averaging the two matching clips per patient.
The permutation test preserves patient membership, and Holm correction covers
all 24 tests within each endpoint. Surrogate diagnostics are unnecessary for
the raw-PLI grid, hence `--n-surrogates 0`. The source-selected cells remain
marked in the output so they are not confused with the newly examined cells.
The executed notebook is
[`notebooks/full_qeeg_response_grid.ipynb`](notebooks/full_qeeg_response_grid.ipynb).

See [`../../docs/rajaraman2024-reproduction.md`](../../docs/rajaraman2024-reproduction.md)
for the method correspondence, numerical findings, and limitations.

## Sources

- [Rajaraman et al. (2024)](https://doi.org/10.1016/j.clinph.2024.03.035)
- [Smith et al. (2021)](https://doi.org/10.1016/j.eplepsyres.2021.106704)
