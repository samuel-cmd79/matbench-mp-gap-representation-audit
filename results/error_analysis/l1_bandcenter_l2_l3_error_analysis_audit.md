# Four-model error-tail analysis audit

## Source and alignment

- Comparison directory: `${HOME}/material_projects_structure/matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0`
- Aligned clipped predictions: `${HOME}/material_projects_structure/matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0/l1_bandcenter_l2_l3_aligned_predictions.csv`
- Input SHA-256: `4d7d4da07239250935ab88a1b88e3bb757ba04fa819b1be2932bc8a4bc7001cb`
- Official fold-assignment SHA-256: `34ee55098e821ad0c3fba1392a79fc839efc8f7690c121be15a80fd92881e2b5`
- The comparison config was loaded and its task/fold hash was verified.
- Every sample ID was unique and appeared exactly once.
- Official folds were exactly `[0, 1, 2, 3, 4]`.
- True labels and all four frozen clipped predictions were finite and nonnegative.
- No model was trained and no prediction was modified.

## Frozen manuscript definitions

```python
true_zero = true_label == 0
true_nonzero = true_label > 0

predicted_near_zero = clipped_prediction < 0.1
predicted_intermediate = (
    (clipped_prediction >= 0.1)
    & (clipped_prediction <= 0.5)
)
predicted_high = clipped_prediction > 0.5

false_near_zero = true_nonzero & predicted_near_zero
zero_gap_miss = true_zero & predicted_high
```

No `true_label >= 0.5` condition was used. The 0.1 and 0.5 eV boundaries
remain asymmetric and the intermediate dead band is retained.

## Absolute-error tails and SSE concentration

- P50/P90/P99/max use each model's full-sample absolute errors.
- Quantiles use NumPy's default linear interpolation.
- Top 1% and 5% are selected separately for each model.
- Selected counts are `ceil(N × fraction)`.
- Ordering is absolute error descending, then sample ID ascending; stable
  mergesort makes boundary ties reproducible.
- SSE share is selected-subset `sum(error²)` divided by full-model `sum(error²)`.

| Model | P50 | P90 | P99 | Max | Top 1% SSE | Top 5% SSE | FNZ SSE |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Level 1 | 0.162311 | 0.902824 | 2.214533 | 7.516181 | 27.196342% | 57.027642% | 0.867591% |
| Level 1 + BandCenter | 0.162433 | 0.905616 | 2.211449 | 7.526256 | 27.176512% | 56.999183% | 0.829196% |
| Level 2 | 0.127378 | 0.785213 | 1.870060 | 6.029141 | 25.559179% | 56.138917% | 0.963457% |
| Level 3 | 0.037540 | 0.437560 | 2.304335 | 7.067976 | 55.999623% | 85.862551% | 8.555127% |

| Model | Top fraction | Selected N | Selected SSE | Total SSE | SSE share |
|:---|---:|---:|---:|---:|---:|
| Level 1 | 1% | 1062 | 9951.861268 | 36592.646260 | 27.196342% |
| Level 1 | 5% | 5306 | 20867.923252 | 36592.646260 | 57.027642% |
| Level 1 + BandCenter | 1% | 1062 | 9974.339581 | 36702.058913 | 27.176512% |
| Level 1 + BandCenter | 5% | 5306 | 20919.873603 | 36702.058913 | 56.999183% |
| Level 2 | 1% | 1062 | 6746.819592 | 26396.855734 | 25.559179% |
| Level 2 | 5% | 5306 | 14818.908865 | 26396.855734 | 56.138917% |
| Level 3 | 1% | 1062 | 13118.530148 | 23426.104261 | 55.999623% |
| Level 3 | 5% | 5306 | 20114.250721 | 23426.104261 | 85.862551% |

## Main 2×3 exact count tables

The upper-right cell is `zero_gap_miss`. The lower-left cell is
`false_near_zero`.

### Level 1

| True-label class | pred < 0.1 | 0.1 ≤ pred ≤ 0.5 | pred > 0.5 | Row total |
|:---|---:|---:|---:|---:|
| true zero (y == 0) | 33720 | 7469 | 4962 | 46151 |
| true nonzero (y > 0) | 2388 | 6686 | 50888 | 59962 |

### Level 1 + BandCenter

| True-label class | pred < 0.1 | 0.1 ≤ pred ≤ 0.5 | pred > 0.5 | Row total |
|:---|---:|---:|---:|---:|
| true zero (y == 0) | 33670 | 7535 | 4946 | 46151 |
| true nonzero (y > 0) | 2396 | 6674 | 50892 | 59962 |

### Level 2

| True-label class | pred < 0.1 | 0.1 ≤ pred ≤ 0.5 | pred > 0.5 | Row total |
|:---|---:|---:|---:|---:|
| true zero (y == 0) | 35370 | 6563 | 4218 | 46151 |
| true nonzero (y > 0) | 2582 | 5997 | 51383 | 59962 |

### Level 3

| True-label class | pred < 0.1 | 0.1 ≤ pred ≤ 0.5 | pred > 0.5 | Row total |
|:---|---:|---:|---:|---:|
| true zero (y == 0) | 42435 | 1858 | 1858 | 46151 |
| true nonzero (y > 0) | 5028 | 5855 | 49079 | 59962 |

## Directional error rates

| Model | Zero-gap miss count / true-zero | Miss / N | False-near-zero count / true-nonzero | FNZ / N | Combined / N |
|:---|---:|---:|---:|---:|---:|
| Level 1 | 4962 / 46151 (10.751663%) | 4.676147% | 2388 / 59962 (3.982522%) | 2.250431% | 6.926578% |
| Level 1 + BandCenter | 4946 / 46151 (10.716994%) | 4.661069% | 2396 / 59962 (3.995864%) | 2.257970% | 6.919039% |
| Level 2 | 4218 / 46151 (9.139564%) | 3.975008% | 2582 / 59962 (4.306061%) | 2.433255% | 6.408263% |
| Level 3 | 1858 / 46151 (4.025915%) | 1.750964% | 5028 / 59962 (8.385311%) | 4.738345% | 6.489309% |

## False-near-zero SSE

| Model | Count | True-label median | FNZ SSE | Total SSE | FNZ SSE share |
|:---|---:|---:|---:|---:|---:|
| Level 1 | 2388 | 0.096550 | 317.474428 | 36592.646260 | 0.867591% |
| Level 1 + BandCenter | 2396 | 0.095050 | 304.332061 | 36702.058913 | 0.829196% |
| Level 2 | 2582 | 0.077750 | 254.322424 | 26396.855734 | 0.963457% |
| Level 3 | 5028 | 0.110200 | 2004.132957 | 23426.104261 | 8.555127% |

## Optional 2×2 table warning

`l1_bandcenter_l2_l3_confusion_2x2_near_zero_counts.csv` uses the single prediction split `pred < 0.1` versus
`pred >= 0.1`. Its `true zero & pred >= 0.1` cell is **not** the manuscript's
`zero_gap_miss`; the manuscript miss requires `true_label == 0` and
`prediction > 0.5`.

## Output inventory

- Overall summary: `l1_bandcenter_l2_l3_error_summary.csv`
- Quantiles: `l1_bandcenter_l2_l3_absolute_error_quantiles.csv`
- Top SSE shares: `l1_bandcenter_l2_l3_top_sse_shares.csv`
- Selected top-5% sample rows: `l1_bandcenter_l2_l3_top5pct_samples.csv`
- False-near-zero SSE: `l1_bandcenter_l2_l3_false_near_zero_sse.csv`
- Directional rates/counts: `l1_bandcenter_l2_l3_directional_errors.csv`
- Main 2×3 counts: `l1_bandcenter_l2_l3_confusion_2x3_counts.csv`
- Optional 2×2 counts: `l1_bandcenter_l2_l3_confusion_2x2_near_zero_counts.csv`
- False-near-zero samples: `l1_bandcenter_l2_l3_false_near_zero_samples.csv`
- Zero-gap-miss samples: `l1_bandcenter_l2_l3_zero_gap_miss_samples.csv`
