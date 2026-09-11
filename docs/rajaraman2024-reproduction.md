# Reproduction of the published response-feature analysis

## Question

Can the response associations reported by Rajaraman et al. (2024) be recovered
from the raw Clinical Clips and the local sustained-response field?

This is a **same-cohort method reproduction**, not an independent clinical
validation. The 50 patients and their response labels are the patients analyzed
in the source article. A successful reproduction verifies the data mapping and
shows that these EEG measurements carry response-associated variation in this
cohort. It does not establish transport to a new hospital or predict benefit
from one treatment rather than another.

## Source method and local implementation

The source article analyzed 50 children, including 28 sustained responders and
22 non-responders. For each pre- or post-treatment study, it averaged two awake
clips for awake features and two sleep clips for sleep features. Its baseline
score uses pre-treatment awake beta-band DFA intercept and delta-band phase-lag
connectivity. Its post-treatment score uses post-treatment sleep beta-band
Shannon entropy and post-treatment awake beta-band DFA intercept.

The reproduction reads the original EDFs rather than the legacy Python feature
cache. MNE returns EDF signals in volts, so the signals are converted to
microvolts before applying the source artifact threshold. The 19 scalp channels
are linked-ear referenced for DFA and entropy and common-average referenced for
connectivity. The source automated artifact rule is applied at 7.5 standard
deviations with an absolute 200 microvolt floor and a 0.9-second buffer. Thirteen
of the 400 case EDFs have a native 2000 Hz rate and are polyphase-resampled to
the source analysis rate of 200 Hz. The remaining 387 are already 200 Hz.

The implementation follows the public Smith et al. MATLAB routines for beta and
delta filtering, 350-bin entropy, and DFA. SciPy requires an odd number of taps
for least-squares FIR design, so the nearest odd-tap design is used where MATLAB
accepts an even number. The source study also removed clinician-marked sleep
artifacts; those marks are not available here.

The PLI procedure is less completely specified. The primary implementation
retains an observed epoch-level PLI only when it exceeds the 95th percentile of
100 Fourier-phase-randomized surrogates, averages the resulting matrices across
clean eight-second epochs, and reports the percentage of channel pairs above
0.20. A declared sensitivity version records a significant edge as one before
the epoch average. Both were fixed before comparing the full-cohort results.

## Results

Patient-level values below are the mean of the two matching clips. Parentheses
contain the first and third quartiles. Local P-values are unadjusted two-sided
Mann-Whitney tests. The source feature P-values were reported in a seven-feature
sequential analysis with Benjamini-Hochberg correction, so the P-values are not
numerically identical estimands.

| Quantity | Local responders | Local non-responders | Local P | Published responders | Published non-responders | Published P |
|---|---:|---:|---:|---:|---:|---:|
| PRE awake beta DFA intercept | -0.208 (-0.417, 0.075) | -0.045 (-0.131, 0.139) | 0.0279 | -0.20 (-0.41, 0.09) | -0.05 (-0.14, 0.15) | 0.046 |
| PRE awake connectivity, primary PLI reading (%) | 0.00 (0.00, 0.00) | 0.00 (0.00, 1.17) | 0.102 | 6.7 (1.6, 13.5) | 12.7 (4.4, 23.7) | 0.039 |
| POST sleep beta entropy | 5.973 (5.789, 6.111) | 5.375 (5.084, 5.678) | 2.32e-5 | 8.06 (7.79, 8.19) | 7.50 (7.15, 7.81) | 0.006 |
| POST awake beta DFA intercept | -0.442 (-0.725, -0.279) | -0.087 (-0.283, 0.014) | 1.44e-4 | -0.43 (-0.70, -0.26) | -0.11 (-0.29, 0.04) | 0.006 |

The source equations were then applied without changing their coefficients:

\[
R_0=-2.361\,\mathrm{DFAI}_0-0.051\,C_0,
\]

\[
R_1=4.765\,H_1-7.786\,\mathrm{DFAI}_1.
\]

| Score | Local group separation | Local AUROC | Published AUROC | Local fixed-feature leave-one-patient-out AUROC | Published n-1 AUROC |
|---|---:|---:|---:|---:|---:|
| PRE score, R0 | 0.484 vs -0.213; P = 0.00720 | 0.724 | 0.75 (0.61-0.89) | 0.721 | 0.69 |
| POST score, R1 | 31.777 vs 26.936; P = 3.56e-7 | 0.924 | 0.93 (0.85-1.00) | 0.917 | 0.91 |

The local leave-one-patient-out analysis refits a logistic regression using the
source model's fixed features plus the published duration category. The article
does not state whether forward feature selection was repeated inside each n-1
iteration, so this is a fixed-feature reproduction of its validation step.

## What reproduced

The strongest numerical check is DFA. PRE and POST group medians and quartiles
agree with the paper to approximately one or two hundredths. This would be very
unlikely if the recording-to-patient mapping, response endpoint, signal unit,
montage, or DFA definition were materially wrong.

The direction and magnitude of the entropy group difference also reproduced:
the local responder-minus-non-responder median difference is 0.598 bits, versus
0.56 bits in the article. The absolute entropy values are lower by roughly two
bits. Because discrete Shannon entropy changes with histogram resolution, and
the 2024 article does not disclose its bin count, this absolute offset cannot be
resolved from the paper alone. A constant offset shifts R1 but does not change
its patient ranking or AUROC.

Most importantly, the source scores recover nearly the same discrimination.
R0 differs from the reported apparent AUROC by 0.026, while R1 differs by 0.006.
The fixed-feature leave-one-patient-out results differ from the reported n-1
AUROCs by 0.031 and 0.007, respectively.

## What did not reproduce exactly

The published PLI distribution did not reproduce. Under the primary reading,
most patient values are zero. Under the binary-significance sensitivity
reading, responders have a median of 1.75% and non-responders 3.95% (P = 0.228):
the direction agrees with the article, but the magnitude and P-value do not.
The target article cites the prior method rather than providing executable PLI
code. An earlier local Python cache contained a different quantity: mean raw
PLI across channel pairs without eight-second epochs, surrogate testing, or the
0.20 network threshold. The published PLI values therefore cannot be treated
as exactly reproduced.

The local automated artifact detector excludes a median 12.6% of extracted
awake data, compared with 8.9% in the article. For POST sleep, local exclusion
is 0.64% in responders and 2.79% in non-responders, compared with 0.7% and 1.7%
in the article. Unavailable clinician annotations, EDF conversion details, and
filter implementation can account for this remaining discrepancy.

## Scientific conclusion

This reproduction supports the statement that both PRE and POST EEG contain
information associated with the sustained-response endpoint in this cohort.
For PRE, the prespecified DFA feature differs between response groups and the
published two-feature score reaches AUROC 0.724. For POST, entropy and DFA each
differ strongly between groups and the published score reaches AUROC 0.924.

This result does **not** by itself establish a generalizable prognostic model.
It uses the same cohort in which the equations were derived, and POST EEG is
measured when immediate electrographic response is already observable. For the
dataset paper, this analysis is best used as a clinically grounded positive
control: it verifies that the response mapping and raw EEG support known
patient-level associations. The patient-disjoint five-fold foundation-model
experiments remain the appropriate evidence for performance under held-out
patients.

## Reproduction artifacts

- Raw extraction: `analysis/response_features/extract_rajaraman2024_raw_features.py`
- Patient aggregation and comparison: `analysis/response_features/reproduce_rajaraman2024.py`
- Committed aggregate result tables: `analysis/response_features/reference_results/`
- Row-level intermediate features and regenerated results: `local_results/rajaraman2024/` (ignored by Git)

The response column and its patient linkage are governed data and are not part
of the public EEG release. Provide the local metadata table explicitly rather
than copying it into this repository. Run from the repository root:

```bash
source /home/mingjian/miniforge3/etc/profile.d/mamba.sh
mamba activate seeg
CUDA_VISIBLE_DEVICES=3 python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features pli --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs
python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features beta --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs
python analysis/response_features/reproduce_rajaraman2024.py \
  --source raw --metadata-csv /path/to/local_clip_metadata.csv
```

## Sources

- Rajaraman et al. (2024), [Computational EEG attributes predict response to therapy for epileptic spasms](https://doi.org/10.1016/j.clinph.2024.03.035).
- Smith et al. (2021), [Computational characteristics of interictal EEG as objective markers of epileptic spasms](https://doi.org/10.1016/j.eplepsyres.2021.106704).
