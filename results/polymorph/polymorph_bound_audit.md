# Fold-conditioned repeated-composition oracle audit

- Run time: `2026-08-15T16:14:36.306983-04:00`
- Python: `3.9.23`
- NumPy: `2.0.2`
- pandas: `2.3.1`
- MatBench: `0.6`
- Task: `matbench_mp_gap`
- Target column: `gap pbe`
- Official OOF samples: `106,113`
- Official folds: `[0, 1, 2, 3, 4]`

## Inputs and sample connection

L1/L2 NPY files contain predictions only. Their IDs were reconstructed from the official test-fold order established by the reviewed generation pipelines. L3 was joined using IDs stored in each NPZ. After ingestion, every merge and subset operation used stable `mbid`, never the current DataFrame row order.

| Level | Fold | Input file | ID mapping | SHA-256 prefix |
| --- | --- | --- | --- | --- |
| L1 | 0 | ${HOME}/material_projects_structure/outputs_v1_run0709/predictions_xgb/pred_fold_0.npy | official MatBench test-fold position reconstructed from reviewed generator | b6b18988c3b1adaa… |
| L1 | 1 | ${HOME}/material_projects_structure/outputs_v1_run0709/predictions_xgb/pred_fold_1.npy | official MatBench test-fold position reconstructed from reviewed generator | b8e844983ad9e4a8… |
| L1 | 2 | ${HOME}/material_projects_structure/outputs_v1_run0709/predictions_xgb/pred_fold_2.npy | official MatBench test-fold position reconstructed from reviewed generator | 0830e8bf36472c01… |
| L1 | 3 | ${HOME}/material_projects_structure/outputs_v1_run0709/predictions_xgb/pred_fold_3.npy | official MatBench test-fold position reconstructed from reviewed generator | 297ef6cddff778fe… |
| L1 | 4 | ${HOME}/material_projects_structure/outputs_v1_run0709/predictions_xgb/pred_fold_4.npy | official MatBench test-fold position reconstructed from reviewed generator | e5539f2caaaaf13a… |
| L2 | 0 | ${HOME}/material_projects_structure/matbench_outputs_v2_run0709/predictions_xgb/pred_fold_0.npy | official MatBench test-fold position reconstructed from reviewed generator | 4ee57cb69d9927e4… |
| L2 | 1 | ${HOME}/material_projects_structure/matbench_outputs_v2_run0709/predictions_xgb/pred_fold_1.npy | official MatBench test-fold position reconstructed from reviewed generator | cb65c1fb8e3ae208… |
| L2 | 2 | ${HOME}/material_projects_structure/matbench_outputs_v2_run0709/predictions_xgb/pred_fold_2.npy | official MatBench test-fold position reconstructed from reviewed generator | 75605a1468fd61b1… |
| L2 | 3 | ${HOME}/material_projects_structure/matbench_outputs_v2_run0709/predictions_xgb/pred_fold_3.npy | official MatBench test-fold position reconstructed from reviewed generator | b2e8bbb520347c68… |
| L2 | 4 | ${HOME}/material_projects_structure/matbench_outputs_v2_run0709/predictions_xgb/pred_fold_4.npy | official MatBench test-fold position reconstructed from reviewed generator | 3d02b1c0f63978ee… |
| L3 | 0 | ${HOME}/material_projects_structure/results_v4/fold_0/test_preds_clipped.npz | direct ID join from NPZ | a4cfe70ce0a062fb… |
| L3 | 1 | ${HOME}/material_projects_structure/results_v4/fold_1/test_preds_clipped.npz | direct ID join from NPZ | 9c58c60c70f8571d… |
| L3 | 2 | ${HOME}/material_projects_structure/results_v4/fold_2/test_preds_clipped.npz | direct ID join from NPZ | 557c78b09d1aa206… |
| L3 | 3 | ${HOME}/material_projects_structure/results_v4/fold_3/test_preds_clipped.npz | direct ID join from NPZ | 2524f288aab2a1a7… |
| L3 | 4 | ${HOME}/material_projects_structure/results_v4/fold_4/test_preds_clipped.npz | direct ID join from NPZ | 93f88a812ee34018… |

## Definitions

- `global fixed-function oracle bound`: formula-level median absolute deviation for one fixed pooled function f(c). It is not a strict lower bound for pooled OOF predictions from fold-specific models.
- `fold-conditioned OOF oracle bound`: median absolute deviation within each `(reduced_formula, official_test_fold)` group.
- `global_repeated_formula_subset`: all entries whose formula occurs at least twice globally.
- `same_fold_repeated_subset`: only complete `(formula, fold)` groups of size at least two.
- Group and population bounds are sample-weighted.
- Formula names alone are described as repeated-composition entries/same-formula groups; no claim of structurally distinct polymorphism is made.

## Legacy reproduction

- Repeated-composition entries: `39,737`
- Same-formula groups: `11,788`
- Global fixed-function oracle bound: `0.15394493 eV` (`0.1539 eV` at four decimals)

| Global formula spread bin | n samples | n formulas | Old global bound (eV) |
| --- | --- | --- | --- |
| <0.1 eV | 14,218 | 5,897 | 0.006179 |
| 0.1–0.5 eV | 9,535 | 3,239 | 0.102407 |
| 0.5–1 eV | 7,116 | 1,546 | 0.226977 |
| >1 eV | 8,868 | 1,106 | 0.387669 |

The old bound is retained only under the name **global fixed-function oracle bound**. It must not be presented as a strict pooled-OOF composition-only bound.

## Main populations

| Analysis | Population | n samples | n formulas | n (formula, fold) groups | Oracle bound (eV) | L1 MAE | L2 MAE | L3 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_reproduction | global_repeated_formula_subset | 39,737 | 11,788 | 26,119 | 0.153945 | 0.298926 | 0.276516 | 0.172346 |
| fold_conditioned | global_repeated_formula_subset | 39,737 | 11,788 | 26,119 | 0.084195 | 0.298926 | 0.276516 | 0.172346 |
| fold_conditioned | same_fold_repeated_subset | 20,299 | 4,489 | 6,681 | 0.164818 | 0.298274 | 0.268308 | 0.173656 |

For the global repeated-formula population:

- size=1 groups: `19,438` groups / `19,438` samples; their oracle contribution is zero.
- size>=2 groups: `6,681` groups / `20,299` samples.

Per-fold fold-conditioned bounds for the global repeated-formula population:

| Fold | n samples | n groups | Bound (eV) |
| --- | --- | --- | --- |
| 0 | 7,933 | 5,198 | 0.084896 |
| 1 | 7,981 | 5,137 | 0.089575 |
| 2 | 8,016 | 5,298 | 0.084231 |
| 3 | 7,915 | 5,320 | 0.077605 |
| 4 | 7,892 | 5,166 | 0.084621 |

The complete exact group-size distribution is in `polymorph_bound_group_size_distribution.csv`.

## Level-1 within-group constancy

- Groups checked (size>=2): `6,681`
- Maximum within-group range: `0 eV`
- Maximum within-group population standard deviation: `0 eV`
- Strictly nonzero groups: `0`
- Groups exceeding tolerance: `0`
- Numerical tolerance: `1e-09 eV`
- No nonzero Level-1 within-group ranges were found.

## Formula-cluster bootstrap

- Bootstrap replicates: `2,000`
- Random seed: `20260730`
- Interval method: percentile bootstrap (2.5th and 97.5th percentiles).
- Cluster unit: `reduced_formula`; every formula's entries across all folds remain together in a resample.

| Population | Metric | Estimate | 95% lower | 95% upper |
| --- | --- | --- | --- | --- |
| global_repeated_formula_subset | fold_conditioned_bound_eV | 0.084195 | 0.075964 | 0.093855 |
| global_repeated_formula_subset | L2_MAE_minus_bound_eV | 0.192321 | 0.176907 | 0.205167 |
| global_repeated_formula_subset | L3_MAE_minus_bound_eV | 0.088151 | 0.074136 | 0.100139 |
| same_fold_repeated_subset | fold_conditioned_bound_eV | 0.164818 | 0.153940 | 0.175511 |
| same_fold_repeated_subset | L2_MAE_minus_bound_eV | 0.103490 | 0.084153 | 0.121097 |
| same_fold_repeated_subset | L3_MAE_minus_bound_eV | 0.008838 | -0.008263 | 0.025042 |

## Global-spread-bin formula-cluster bootstrap

- Spread definition: `global within-formula label spread`.
- Bootstrap replicates per population/bin: `2,000`.
- Random seed: `20260730`.
- Interval method: percentile bootstrap (2.5th and 97.5th percentiles).
- Cluster unit: `reduced_formula`; all folds and samples of a formula within the named analysis population remain together.
- Resampling multiplicity is preserved by repeated integer-array indexing; there is no post-draw groupby, unique operation, or deduplication.
- Decision rule: CI upper < 0 = robust undercut; CI lower > 0 = robust above bound; otherwise the CI includes zero and only the point estimate is reported as directional evidence.

Main table for `same_fold_repeated_subset`:

| Global spread bin | Oracle bound | L2 - bound | L2 95% CI | L2 decision | L3 - bound | L3 95% CI | L3 decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <0.1 eV | 0.005032 | 0.157999 | [0.144103, 0.172952] | robust_above_bound_ci_lower_above_zero | 0.081953 | [0.067706, 0.099333] | robust_above_bound_ci_lower_above_zero |
| 0.1–0.5 eV | 0.081495 | 0.198662 | [0.181492, 0.216264] | robust_above_bound_ci_lower_above_zero | 0.093959 | [0.078241, 0.111207] | robust_above_bound_ci_lower_above_zero |
| 0.5–1 eV | 0.174737 | 0.097801 | [0.082994, 0.113338] | robust_above_bound_ci_lower_above_zero | 0.000661 | [-0.011844, 0.014705] | inconclusive_ci_includes_zero |
| >1 eV | 0.308829 | 0.016485 | [-0.014811, 0.053171] | inconclusive_ci_includes_zero | -0.082437 | [-0.110574, -0.050507] | robust_undercut_ci_upper_below_zero |

Both populations are available in `polymorph_bound_bootstrap_by_spread.csv`.
The main same-fold CI evidence is visualized in `polymorph_bound_delta_ci_by_spread.png`.

## Assertions

| Assertion | Status | Detail |
| --- | --- | --- |
| official fold labels match configured folds | PASS | official=(0, 1, 2, 3, 4); configured=(0, 1, 2, 3, 4) |
| fold 0 has a unique official index | PASS | n_rows=21223; n_unique_ids=21223 |
| fold 0 contains structure and target columns | PASS | required=['gap pbe', 'structure']; actual=['structure', 'gap pbe'] |
| fold 1 has a unique official index | PASS | n_rows=21223; n_unique_ids=21223 |
| fold 1 contains structure and target columns | PASS | required=['gap pbe', 'structure']; actual=['structure', 'gap pbe'] |
| fold 2 has a unique official index | PASS | n_rows=21223; n_unique_ids=21223 |
| fold 2 contains structure and target columns | PASS | required=['gap pbe', 'structure']; actual=['structure', 'gap pbe'] |
| fold 3 has a unique official index | PASS | n_rows=21222; n_unique_ids=21222 |
| fold 3 contains structure and target columns | PASS | required=['gap pbe', 'structure']; actual=['structure', 'gap pbe'] |
| fold 4 has a unique official index | PASS | n_rows=21222; n_unique_ids=21222 |
| fold 4 contains structure and target columns | PASS | required=['gap pbe', 'structure']; actual=['structure', 'gap pbe'] |
| every official OOF sample belongs to exactly one test fold | PASS | n_rows=106113; n_unique_mbid=106113 |
| official labels are finite | PASS | n_nonfinite=0 |
| official fold-position pairs are unique | PASS | n_duplicates=0 |
| legacy repeated-composition sample count | PASS | computed=39737; reference=39737 |
| legacy same-formula group count | PASS | computed=11788; reference=11788 |
| legacy global fixed-function bound at four decimals | PASS | computed=0.1539; reference=0.1539 |
| all 15 frozen prediction files exist | PASS | 5 folds each for L1, L2, and L3 |
| L1 reconstructed IDs are unique | PASS | n_rows=106113; n_unique_mbid=106113 |
| L2 reconstructed IDs are unique | PASS | n_rows=106113; n_unique_mbid=106113 |
| L3 fold 0 ID set matches the official test fold | PASS | n_ids=21223; official=21223; missing=0; extra=0 |
| L3 fold 1 ID set matches the official test fold | PASS | n_ids=21223; official=21223; missing=0; extra=0 |
| L3 fold 2 ID set matches the official test fold | PASS | n_ids=21223; official=21223; missing=0; extra=0 |
| L3 fold 3 ID set matches the official test fold | PASS | n_ids=21222; official=21222; missing=0; extra=0 |
| L3 fold 4 ID set matches the official test fold | PASS | n_ids=21222; official=21222; missing=0; extra=0 |
| L3 IDs are unique across all official folds | PASS | n_rows=106113; n_unique_mbid=106113 |
| L1 covers exactly the official OOF sample IDs | PASS | official=106113; predictions=106113; missing=0; extra=0 |
| L1 fold labels agree with the official mapping | PASS | n_mismatches=0 |
| L1 has no missing values after stable-ID merge | PASS | n_missing=0 |
| L2 covers exactly the official OOF sample IDs | PASS | official=106113; predictions=106113; missing=0; extra=0 |
| L2 fold labels agree with the official mapping | PASS | n_mismatches=0 |
| L2 has no missing values after stable-ID merge | PASS | n_missing=0 |
| L3 covers exactly the official OOF sample IDs | PASS | official=106113; predictions=106113; missing=0; extra=0 |
| L3 fold labels agree with the official mapping | PASS | n_mismatches=0 |
| L3 has no missing values after stable-ID merge | PASS | n_missing=0 |
| labels and all three prediction levels cover the same samples | PASS | prediction_columns=['pred_L1', 'pred_L2', 'pred_L3'] |
| L1 predictions are finite and respect nonnegative clipping | PASS | n=106113; n_nonfinite=0; minimum=0 |
| L2 predictions are finite and respect nonnegative clipping | PASS | n=106113; n_nonfinite=0; minimum=0 |
| L3 predictions are finite and respect nonnegative clipping | PASS | n=106113; n_nonfinite=0; minimum=0 |
| size=1 (formula, fold) groups contribute exactly zero oracle error | PASS | n_singleton_samples=19438; max_contribution=0 |
| fold-conditioned oracle is no larger than the global fixed-function oracle | PASS | fold_conditioned=0.0841945692931; global_fixed=0.153944925384 |
| same-fold repeated subset contains only complete size>=2 groups | PASS | n_samples=20299; min_group_size=2 |
| same-fold repeated subset is a subset of the global repeated-formula subset | PASS | A=39737; B=20299 |
| L1 is constant within every size>=2 (formula, fold) group at tolerance | PASS | checked=6681; max_range=0; max_std=0; tolerance=1e-09; n_exceeding=0 |
| L1 is exactly constant within every size>=2 (formula, fold) group | PASS | n_strictly_nonzero=0 |
| L1 MAE is not below the fold-conditioned oracle in global_repeated_formula_subset | PASS | L1_MAE=0.298926125122; bound=0.0841945692931 |
| L1 MAE is not below the fold-conditioned oracle in same_fold_repeated_subset | PASS | L1_MAE=0.298273561602; bound=0.164817951623 |
| formula-cluster bootstrap completed for both populations | PASS | rows=6; replicates=2000; seed=20260730 |
| global-spread-bin formula-cluster bootstrap completed for both populations | PASS | rows=8; expected=8; replicates=2000; seed=20260730 |
| spread-bin bootstrap uses 2,000 replicates and seed 20260730 | PASS | replicates=2000; seed=20260730; required_replicates=2000; required_seed=20260730 |
| spread-bin bootstrap explicitly uses global within-formula label spread | PASS | definitions=['global within-formula label spread'] |
| spread-bin bootstrap preserves repeated formula sampling multiplicity | PASS | implementation uses repeated integer indices with no post-draw groupby or deduplication |
| spread-bin bootstrap point estimates equal the existing spread-table estimates | PASS | all rows match at 12 decimals |
| same-fold repeated subset has all four required global-spread bins | PASS | bins=['<0.1 eV', '0.1–0.5 eV', '0.5–1 eV', '>1 eV']; sample_counts={'<0.1 eV': 4493, '0.1–0.5 eV': 4181, '0.5–1 eV': 4533, '>1 eV': 7092} |
| acceptance point estimate same_fold_repeated_subset / <0.1 eV / L2_MAE_minus_bound_eV | PASS | computed=+0.157999; reference=+0.157999 |
| acceptance point estimate same_fold_repeated_subset / <0.1 eV / L3_MAE_minus_bound_eV | PASS | computed=+0.081953; reference=+0.081953 |
| acceptance point estimate same_fold_repeated_subset / 0.1–0.5 eV / L2_MAE_minus_bound_eV | PASS | computed=+0.198662; reference=+0.198662 |
| acceptance point estimate same_fold_repeated_subset / 0.1–0.5 eV / L3_MAE_minus_bound_eV | PASS | computed=+0.093959; reference=+0.093959 |
| acceptance point estimate same_fold_repeated_subset / 0.5–1 eV / L2_MAE_minus_bound_eV | PASS | computed=+0.097801; reference=+0.097801 |
| acceptance point estimate same_fold_repeated_subset / 0.5–1 eV / L3_MAE_minus_bound_eV | PASS | computed=+0.000661; reference=+0.000661 |
| acceptance point estimate same_fold_repeated_subset / >1 eV / L2_MAE_minus_bound_eV | PASS | computed=+0.016485; reference=+0.016485 |
| acceptance point estimate same_fold_repeated_subset / >1 eV / L3_MAE_minus_bound_eV | PASS | computed=-0.082437; reference=-0.082437 |

## Interpretation

### global_repeated_formula_subset

- L2 MAE 0.276516 eV is above the bound 0.084195 eV by 0.192321 eV, 228.43% above.
- L3 MAE 0.172346 eV is above the bound 0.084195 eV by 0.088151 eV, 104.70% above.

### same_fold_repeated_subset

- L2 MAE 0.268308 eV is above the bound 0.164818 eV by 0.103490 eV, 62.79% above.
- L3 MAE 0.173656 eV is above the bound 0.164818 eV by 0.008838 eV, 5.36% above.

The manuscript's old `0.1539 eV` value may be retained only with the renamed global fixed-function interpretation above. The old “32% undercut” statement is not carried forward automatically: it must be withdrawn or replaced by the computed fold-conditioned percentage in the corresponding population/spread row. An undercut percentage is populated in the CSV only when model MAE is truly below the relevant bound; otherwise the separate above-bound percentage is reported.

## Output files

- `polymorph_bound_prediction_manifest.csv`
- `polymorph_bound_summary.csv`
- `polymorph_bound_by_spread.csv`
- `polymorph_bound_group_size_distribution.csv`
- `polymorph_bound_bootstrap.csv`
- `polymorph_bound_bootstrap_by_spread.csv`
- `repeated_composition_groups_top.csv`
- `polymorph_bound_by_spread.png`
- `polymorph_bound_delta_ci_by_spread.png`
- `polymorph_bound_audit.md`
