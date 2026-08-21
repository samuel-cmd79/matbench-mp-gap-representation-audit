# Level 1 + BandCenter training audit

## Scope

This audit covers only the independent 133-feature XGBoost training run.
The Level-1 / Level-2 frozen-result comparison is intentionally deferred to a
separate comparison script.

## Frozen feature source

- Cache directory: `${HOME}/material_projects_structure/matbench_cache`
- Level-1 source: `fold_{fold}_{train|test}_ElementProperty.pkl`
- BandCenter source: `fold_{fold}_{train|test}_BandCenter.pkl`
- BandCenter was read directly from the same pre-imputation, per-featurizer
  cache convention used by the uploaded frozen Level-2 pipeline.
- No matminer featurizer was instantiated and no missing cache was regenerated.
- Exact file paths and SHA-256 hashes are stored in `l1_plus_bandcenter_cache_manifest.csv` and
  `l1_plus_bandcenter_config.json`.

## Feature assertions

- ElementProperty columns: **132**
- BandCenter columns: **1**
- Combined columns: **133**
- Structural cache files were never loaded.
- No features were sourced from: DensityFeatures, GlobalSymmetryFeatures, StructuralHeterogeneity, ChemicalOrdering, Dimensionality, SiteStatsFingerprint.
- Train/test feature names and order were identical across all five folds.
- Feature provenance and order are stored in `l1_plus_bandcenter_feature_list.csv`.

## Sample and fold alignment

- Official MatBench folds were exactly `[0, 1, 2, 3, 4]`.
- Within every fold, official train/test sample IDs were unique and disjoint.
- The five official test folds covered the full sample universe exactly once.
- Cache row counts matched the corresponding ordered official split.
- Range-indexed frozen caches were bound positionally to the ordered sample IDs,
  matching the uploaded cache-generation pipeline.
- Official fold-assignment SHA-256:
  `34ee55098e821ad0c3fba1392a79fc839efc8f7690c121be15a80fd92881e2b5`
- Assignments are stored in `l1_plus_bandcenter_fold_assignments.csv`.
- The realized internal 80/20 sample roles are stored in `l1_plus_bandcenter_internal_split.csv`.

## Imputation policy

This run deliberately reproduced the uploaded baseline/ablation scripts:

1. Replace infinity with missing values.
2. Compute column means on the **full official outer train-and-validation fold**.
3. Fill the full outer train-and-validation fold and official test fold using
   those means.
4. Apply the uploaded script's final zero fallback for a still-all-missing column.
5. Only then split the imputed outer train-and-validation fold 80/20 with
   `random_state=42`.

The internal early-stopping validation subset therefore participates in the
outer-fold mean, exactly as in the uploaded scripts. No missingness indicator
was added.

## BandCenter missingness

| Fold | Split | Rows | BC missing | BC missing rate | Cache index mode |
|---:|:---|---:|---:|---:|:---|
| 0 | train | 84890 | 3013 | 3.549299% | positional_range_index |
| 0 | test | 21223 | 682 | 3.213495% | positional_range_index |
| 1 | train | 84890 | 2924 | 3.444458% | positional_range_index |
| 1 | test | 21223 | 771 | 3.632851% | positional_range_index |
| 2 | train | 84890 | 2943 | 3.466839% | positional_range_index |
| 2 | test | 21223 | 752 | 3.543326% | positional_range_index |
| 3 | train | 84891 | 2977 | 3.506850% | positional_range_index |
| 3 | test | 21222 | 718 | 3.383282% | positional_range_index |
| 4 | train | 84891 | 2923 | 3.443239% | positional_range_index |
| 4 | test | 21222 | 772 | 3.637734% | positional_range_index |

## Fold results

All reported official-test metrics use non-negativity-clipped predictions.
Raw and clipped predictions are both stored.

| Fold | Features | BC missing outer train | BC missing official test | BC mean | Best iter (0-based) | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 133 | 3013 (3.549299%) | 682 (3.213495%) | 5.245816618 | 7999 | 0.339254 | 0.586613 | 0.863251 |
| 1 | 133 | 2924 (3.444458%) | 771 (3.632851%) | 5.246122754 | 7996 | 0.331338 | 0.581504 | 0.866042 |
| 2 | 133 | 2943 (3.466839%) | 752 (3.543326%) | 5.2440659 | 7999 | 0.336683 | 0.581594 | 0.867269 |
| 3 | 133 | 2977 (3.506850%) | 718 (3.383282%) | 5.244889863 | 7999 | 0.339414 | 0.585620 | 0.867896 |
| 4 | 133 | 2923 (3.443239%) | 772 (3.637734%) | 5.244978497 | 7999 | 0.345698 | 0.604914 | 0.859142 |

## Five-fold summary

- MAE: **0.338477 ± 0.005192 eV**
- RMSE: **0.588049 ± 0.009707 eV**
- R²: **0.864720 ± 0.003592**
- Fold SD uses `ddof=1`.

## Interpretation boundary

This training-only script does not determine how much of the frozen Level-1 to
Level-2 improvement is already present after adding BandCenter. That question,
including the paired fold differences and the “sequential gain under this
feature-addition order,” must be answered by the separate comparison script.
No result from this run should be interpreted as a strict causal contribution
of BandCenter.
