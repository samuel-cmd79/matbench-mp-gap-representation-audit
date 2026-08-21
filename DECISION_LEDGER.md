# Decision Ledger

**Status:** Governing ledger for release `v1.0.0`. Earlier ledger drafts are superseded and are not the authoritative interpretation of the accompanying manuscript.

## Purpose and provenance

This document records how the study developed, why major analytical choices were made, and how several claims were narrowed after audit. It is intended to make exploratory decisions visible rather than to present the project as if every analysis had been specified in advance.

The initial motivation below is a **retrospective reconstruction supplied by the author on 17 August 2026**, based on the author's recollection of the earliest project discussions. It is not a preregistration. Later entries are reconstructed from the manuscript, Supporting Information, frozen outputs, scripts, and audit files in this repository. Exact calendar dates are not available for every decision, so the ledger uses study stages rather than invented timestamps.

## How the research question evolved

The project began from a chemistry-motivated question: could machine-learning models be used as screening tools to identify which chemical and structural factors might be worth investigating as contributors to band-gap variation? The aim was not to accept a model's importance score as a physical law. Instead, the intended workflow was to change the information supplied to a model, observe which additions improved prediction, compare the model signals with the literature, and use the resulting patterns to prioritize hypotheses for later physical validation.

The first model used composition alone. Its behavior was unexpected in two ways: XGBoost did not outperform the random-forest baseline as anticipated, and the largest practical problems were concentrated toward the low-gap/zero-gap region. Extending XGBoost training from an initial 2,000-round diagnostic to an 8,000-round cap did not remove the discrepancy, but every run reached that cap, so the exercise did not establish convergence. It weakened a simple gross-cap explanation without ruling out residual optimization sensitivity and motivated a change in representation rather than an open-ended hyperparameter search.

The next step added engineered structural descriptors. The planned graph step followed from a different representational idea: a graph network models relations among neighboring atoms rather than relying only on a fixed table of global or hand-crafted descriptors. From a coordination-chemistry perspective, improved graph performance could therefore motivate closer examination of local environments, coordination, and bond geometry.

That initial interpretation was subsequently made more conservative. Better graph performance alone cannot establish that coordination chemistry is the cause, because the graph model also differs in architecture, capacity, optimization, and direct model-fitting fraction. Likewise, attention or feature attribution is not direct evidence of a physical mechanism. The final study therefore asks a narrower question: **what predictive changes occur as representation–learner pipelines introduce structural information in stages, and which apparent explanatory signals survive deletion, provenance, confounding, and uncertainty checks?** The resulting physical interpretations are hypothesis-generating rather than causal.

---

## Major study decisions

### D01 — Use a fixed public benchmark rather than a bespoke split

- **Trigger:** A staged representation comparison requires the same evaluation cases at every level.
- **Decision:** Use MatBench v0.1 `mp_gap`, its 106,113 PBE targets, and the five official test folds. Record Levels 1 and 2 through the MatBench API; export the same fold identities for the separate ALIGNN environment and import its predictions for official scoring.
- **Reason:** This prevents each representation from benefiting from a different test split and makes the reported scores comparable to the benchmark protocol.
- **Boundary:** The official random folds are not formula-grouped or structure-deduplicated. Identical formulas and near-duplicate structures can cross train/test boundaries.
- **Repository evidence:** `notebooks/`, `scripts/export_labels_by_fold.py`, `gnn_export/`, and `results/frozen_scores/`.

### D02 — Begin with composition and retain two tree algorithms

- **Trigger:** The original question was whether composition alone was sufficient and which chemical patterns the model would use.
- **Decision:** Construct Level 1 from 132 Magpie/ElementProperty columns and evaluate both random forest and XGBoost.
- **Reason:** Composition is available without a crystal structure and provides a chemically interpretable baseline. The second tree algorithm serves as an algorithm–representation control rather than as a second rung in the main ladder.
- **Consequence:** RF and XGBoost are close at Level 1, but only XGBoost realizes a strong gain after descriptors are added. The descriptor benefit is therefore described as learner-dependent, not universal to all tree ensembles.
- **Repository evidence:** `notebooks/mp_gap_baseline_v1_run0709.ipynb`, `results/frozen_scores/level1/`, and the earlier exploratory notebooks in `archive/legacy_diagnostic/`.

### D03 — Increase the XGBoost cap as a practical cap-sensitivity check, not as test-directed tuning

- **Trigger:** The early XGBoost/RF ordering was contrary to expectation, raising the possibility that the boosted model had simply not trained long enough.
- **Decision:** Increase the maximum from an initial 2,000-round diagnostic to 8,000 rounds, retain validation monitoring and 200-round patience, and use the same XGBoost settings at Levels 1 and 2.
- **Reason:** The purpose was to test whether the discrepancy was dominated by a grossly inadequate initial cap before adding new information. No systematic hyperparameter search was performed, and official test labels were not used to select the cap.
- **Consequence:** Every run reached the 8,000-round ceiling and is therefore treated as capped rather than converged. Late validation improvement was small relative to the frozen Level-1→2 and Level-2→3 MAE gaps. The paper compares those scales; it does not equate validation optimization gain with fold-to-fold test uncertainty.
- **Repository evidence:** `scripts/mp_gap_replay_fold0_curve.py`, `scripts/mp_gap_replay_v2_allfold_training_info.py`, `results/training_replay/`, and `figure/xgb_mae_curve_fold_0.png`.

### D04 — Add engineered structure before moving to a graph model

- **Trigger:** Longer tree training did not resolve the main error pattern, especially near the low-gap boundary.
- **Decision:** Define Level 2 by augmenting Level 1 with density, symmetry, structural heterogeneity, chemical ordering, dimensionality, and CrystalNN-based site statistics, while holding the Level-1/Level-2 XGBoost hyperparameters fixed.
- **Reason:** This tests whether explicitly supplied structural summaries improve the same learner before changing to a different model family.
- **Boundary:** Level 2 contains 283 columns, including one composition-derived Level-2-only feature, BandCenter. At least one structure featurizer fails for 10.1% of entries, and that missingness is target-dependent.
- **Repository evidence:** `notebooks/mp_gap_baseline_v2_run0709.ipynb`, `scripts/check_features_v3.py`, `scripts/subset_robustness_check.py`, and the Level-2 outputs in `matbench_outputs_v2_run0709/`.

### D05 — Add ALIGNN to test learned local relational structure

- **Trigger:** Engineered descriptors improved XGBoost, but the original scientific interest included local atomic relationships that fixed global descriptors may compress or miss.
- **Decision:** Use ALIGNN as Level 3, with atom–bond message passing and a line graph containing angular relations.
- **Initial expectation:** If a graph model improved strongly, local-environment information would become a priority for chemical interpretation and later physical validation.
- **Final interpretation:** The complete graph representation–learner–training pipeline achieves lower MAE in this comparison. That result does **not** uniquely attribute the gain to coordination, bond angles, attention, or any single physical mechanism.
- **Repository evidence:** `scripts/colab/v4_alignn_matbench_mp_gap_fixed_final_fi.py`, `gnn_export/`, `results_v4/`, and `results/frozen_scores/level3/`.

### D06 — Define the comparison as fold-controlled, not budget-matched

- **Trigger:** The models share official test folds but do not use identical learners, direct model-fitting fractions, capacities, optimizers, configuration sources, or optimization budgets.
- **Decision — revised during pre-release audit:** Describe the design as a **fold-controlled representation–learner audit**. “Fold-controlled” denotes common targets, official test folds, scoring metrics, and non-negativity post-processing. Learner, model family and capacity, optimizer, recommended-configuration source, direct model-fitting fraction, and optimization budget are not controlled across levels. The approximate direct model-fitting fractions are 64% for XGBoost, 72% for ALIGNN, and 80% for random forest. At Level 2, preprocessing means use the complete official outer-training fold, so 64% refers only to XGBoost parameter fitting.
- **Reason:** Calling the budgets “matched,” or describing the hierarchy as controlled without defining the boundary, would overstate the design.
- **Consequence:** The Level-2-to-Level-3 result is an observed representation–learner–training contrast and does not quantify a pure representation effect.
- **Repository evidence:** `notebooks/mp_gap_baseline_v1_run0709.ipynb`, `notebooks/mp_gap_baseline_v2_run0709.ipynb`, `scripts/colab/v4_alignn_matbench_mp_gap_fixed_final_fi.py`, and the frozen prediction directories documented in `README.md`.

### D07 — Apply a non-negativity constraint and retain raw outputs separately

- **Trigger:** A band-gap prediction below zero is outside the intended target domain, while raw values remain diagnostically useful near the zero boundary.
- **Decision:** Clip headline predictions to `[0, infinity)` before benchmark scoring and preserve raw arrays where provenance permits.
- **Reason:** This gives physically admissible submitted predictions without erasing information needed to determine whether boundary concentration was learned or created by clipping.
- **Consequence:** All headline metrics use frozen clipped predictions. Raw claims are governed by the provenance decision below.
- **Repository evidence:** frozen arrays in `outputs_v1_run0709/`, `matbench_outputs_v2_run0709/`, and `results_v4/`; submission records in `results/frozen_scores/`.

### D08 — Stratify raw-prediction claims by provenance

- **Trigger:** The frozen Level-2 run did not export its pre-clipping predictions, while archived Level-2 raw values came from an older imputation policy.
- **Decision:** Use three evidentiary labels:
  - **Level 1:** frozen-equivalent reconstruction; directly verified on the 89,276 positive raw values and supported by deterministic model identity on the negative subset.
  - **Level 2:** old-imputation diagnostic only; excluded from formal raw pairwise and frozen clipping-gain claims.
  - **Level 3:** same-run frozen raw and clipped arrays; reconciled exactly.
- **Reason:** Clipped equality alone is many-to-one on negative predictions and cannot, by itself, establish raw equality everywhere.
- **Consequence:** The formal zero-gap raw comparison is Level 1→Level 3. Level 2 may appear only as a clearly labeled diagnostic interpolation.
- **Repository evidence:** `scripts/raw_provenance_audit.py`, `results/audits/raw_provenance_audit.md`, and the provenance-labeled raw arrays.

### D09 — Treat attribution, attribution density, and deletion response as different fitted-pipeline quantities

- **Trigger:** The 122-column coordination group collected the largest Level-2 group-total SHAP share, which could be mistaken for evidence that the group was uniquely indispensable.
- **Decision — revised during pre-release audit:** Normalize SHAP within each fold, average folds equally, report both group totals and mean share per column, and retrain after deleting the two tested groups, coordination and symmetry.
- **Reason:** Group totals accumulate attribution over all member columns and are sensitive to group size and correlation. Deletion retraining measures a different fixed-protocol retraining response.
- **Consequence:** Coordination has 26.2% total attribution. Under the fixed single-seed, 8,000-round capped protocol, deleting it gives an equal-weight five-fold mean-MAE point estimate of ΔMAE = −0.005315 eV; that point estimate does not indicate worsening. No uncertainty interval was computed, and its magnitude is similar to the 0.0048 eV validation improvement observed over the final 2,000 rounds of the fold-0 replay, so the negative sign is treated as protocol-specific rather than convergence-stable. Those two quantities concern different datasets and estimands and are neither subtracted nor equated. Symmetry has 11.8% total attribution, while its deletion response is ΔMAE = +0.028701 eV. In these two tested groups, gross coordinatewise SHAP share and retrain-after-removal response rank differently. Neither quantity is interpreted as causal or representation-intrinsic, and the cases do not validate per-column density as a general estimator of deletion response.
- **Repository evidence:** `scripts/shap_crossfold_aggregate_audited_v3.py`, `scripts/verify_deletion_per_fold.py`, `results/shap/`, `matbench_outputs_v2_ablation_coordination/`, `matbench_outputs_v2_ablation_symmetry/`, and the frozen Level-1/Level-2 SHAP arrays and feature-name files.

### D10 — Separate BandCenter from structural information and test it directly

- **Trigger:** BandCenter appears only at Level 2 but is composition-derived, so including it in the structural subtotal would confound the Level-1→2 interpretation.
- **Decision:** Give BandCenter its own SHAP group, exclude it from the structural total, and train a Level-1+BandCenter control.
- **Consequence:** Under `Δ_improvement = MAE_reference − MAE_candidate`, the BandCenter gain is −0.000511 eV, whereas +0.052980 eV remains between Level 1+BandCenter and Level 2. BandCenter therefore does not materially account for the descriptor-stage gain.
- **Repository evidence:** `scripts/train_l1_plus_bandcenter.py`, `scripts/compare_l1_bandcenter_l2_l3.py`, `results/bandcenter/`, and `results/bandcenter_comparison/`.

### D11 — Convert the repeated-composition limitation into an empirical formula-fold test-set oracle

- **Trigger:** Composition-only models necessarily assign one prediction within each formula–fold group, but the first pooled calculation and the within-fold constancy test covered different populations.
- **Decision — revised during pre-release audit:** Keep three quantities distinct. The legacy unconditioned whole-formula oracle is 0.153945 eV on all 39,737 globally repeated-formula entries. Applying fold conditioning to that same descriptive population gives 0.084195 eV, but 19,438 entries (48.9%) are singleton formula–fold cells and contribute zero by construction. The primary post hoc fold-conditioned repeated-composition empirical test-set oracle is therefore defined on reduced-formula–official-test-fold groups containing at least two official-test entries: 20,299 entries in 6,681 groups and 4,489 reduced formulas, with an oracle MAE of 0.164818 eV.
- **Reason:** The primary definition domain must match the domain on which a fold-specific composition model is verified to be constant. The quantity is the observed-label MAE minimum for predictors constrained to be constant within each observed formula–fold group; it is neither a prospective benchmark nor a universal irreducible-error estimate.
- **Consequence:** On the aggregate primary domain, Level-3 MAE minus the oracle is +0.008838 eV with a 95% formula-cluster bootstrap interval of [−0.008263, +0.025042] eV. The interval spans zero, so the aggregate comparison is inconclusive and does not establish equivalence. Only in the post hoc `>1 eV` **global** within-formula-spread stratum is the Level-3 difference negative with its entire unadjusted pointwise interval below zero: −0.082437 eV [−0.110574, −0.050507]. The four spread-bin intervals are not multiplicity-adjusted and condition on frozen predictions; they exclude refitting, random-seed, hyperparameter, checkpoint-selection, model-selection, and bin-definition uncertainty.
- **Boundary:** “Repeated composition” does not establish that every group contains experimentally distinct polymorphs, and no structural deduplication was performed.
- **Repository evidence:** `scripts/polymorph_bound_fold_conditioned_v3.py`, `results/polymorph/`, and the frozen prediction inputs listed in its source manifest.

### D12 — Separate the clipping atom, error body, boundary placements, and extreme tail

- **Trigger:** Level 2→3 reduced MAE by 37.1% but RMSE by only 5.8%.
- **Decision — revised during pre-release audit:** Do not use the pooled median as the primary error-body comparison because non-negativity clipping creates a growing exact-zero-error atom on zero-gap targets. Report the atom explicitly, use the atom-free positive-gap quantiles for the error-body comparison, and retain full-dataset SSE concentration for the tail. Describe the directional event as a **positive-gap near-zero placement**, because a near-zero prediction can be accurate when the true positive gap is itself small.
- **Reason:** A pooled quantile or average can hide both boundary pinning and concentration in a small severe-error tail. The directional threshold events are descriptive placements, not a balanced classification-error metric.
- **Consequence:** The full-dataset exact-zero-error atom grows from 14.96% to 16.85% to 18.46% across Levels 1–3. On the positive-gap subset, the Level-1-to-Level-3 improvement factor is 2.47× at P50, 1.89× at P90, and 1.63× at P95, then reverses between P95 and P99. Across the full dataset, the worst 1% of Level-3 errors carries 56.0% of its SSE (the corresponding positive-gap value is 48.7%). The body contracts, but the extreme tail does not improve uniformly.
- **Repository evidence:** `scripts/analyze_l1_bandcenter_l2_l3_error_tails.py`, `results/error_analysis/`, and `figure/absolute_error_quantile_curves.png`.

### D13 — Replace a depth-changing graph ablation with a configuration-matched angle-value mask

- **Trigger:** The initial idea of setting `alignn_layers=0` would change model depth, capacity, and message-passing structure at the same time, so it would not isolate bond-angle values.
- **Decision — revised during pre-release audit:** Retain four ALIGNN layers, four GCN layers, the same architecture and nominal parameter count, and the line-graph topology, but multiply the angle-embedding output by zero. Verify the intervention with a same-model, same-input smoke test before training.
- **Reason:** This more closely isolates the numerical angle information while preserving the architecture and three-body connectivity.
- **Why not use “attention” for this question:** The original ALIGNN architecture alternates **edge-gated graph convolutions** on the bond graph and line graph; it is not a standard graph-attention network exposing a single set of GAT-style attention coefficients. Its learned edge gates are message-passing quantities that vary by layer and context. Even if they are described informally as attention-like weights, they are not validated estimates of causal physical importance and do not provide a unique decomposition of the prediction into coordination, angle, and other structural mechanisms.
- **Why use an ablation instead:** The mask defines a direct intervention on one information channel—numerical angle embeddings—while retaining model depth, nominal parameter count, line-graph topology, training schedule, and input structures. The resulting held-out-MAE change measures the response of this recorded fold-0 run when explicit angle values are unavailable. It does not claim to identify which individual angle or chemical mechanism is causal.
- **Consequence:** On fold 0, masking raises clipped MAE by 0.01453 eV (8.2%), computed from the unrounded official-test MAEs. This one-fold, one-seed result shows that the recorded pipeline is sensitive to suppression of the explicit numerical angle-embedding channel in this configuration. It does not decompose the Level-2-to-Level-3 contrast or assign the remaining difference among topology, capacity, atom–bond message passing, learned local geometry, optimizer, or direct model-fitting fraction. Recorded GPU kernels were nondeterministic, and run-to-run variance was not quantified.
- **Repository evidence:** `scripts/colab/v4_alignn_matbench_mp_gap_angle_zero_fold0.py`, `results/angle_mask/`, and `result_v4_angle_zero_fold0/`.

### D14 — Use model interpretation as hypothesis screening, not causal identification

- **Trigger:** The original project sought physically meaningful structure–property relationships, but the features, materials, and learned responses are correlated.
- **Decision:** Require literature consistency and robustness checks before giving directional model patterns any physical discussion, and explicitly withhold independent interpretation when a check fails.
- **Examples:**
  - The apparent octahedral-coordination direction changes sign after transition-metal stratification and is not given an independent physical interpretation.
  - Space-group number may help tree partitions but is not interpreted as a continuous coordinate within a crystal system.
  - GNN attention was part of the initial interpretability motivation, but no validated attention analysis is used as evidence in the final study.
- **Consequence:** The paper identifies candidate relationships, predictive patterns, and representation–learner contrasts. It does not claim that SHAP, attention, or predictive improvement proves a real-world mechanism.
- **Repository evidence:** `scripts/check_octahedral_tm_confound_2.py`, the SHAP audits in `results/shap/`, and the manuscript limitations.

### D15 — Freeze headline outputs and preserve diagnostics without promoting them

- **Trigger:** Several useful exploratory analyses were produced before the final imputation and provenance policies were fixed.
- **Decision:** Use only frozen, clipped, official-fold predictions for headline benchmark metrics; preserve older raw or subset results only when explicitly labeled diagnostic; archive superseded notebooks separately.
- **Reason:** Replacing historical artifacts with rerun outputs would erase useful provenance, while silently mixing versions would make the results unauditable.
- **Consequence:** “Diagnostic” does not mean erroneous; it means the result is not substituted for the formal frozen comparison.
- **Repository evidence:** `results/frozen_scores/`, `results/audits/`, `archive/legacy_diagnostic/`, `ARCHIVE_ONLY.txt`, and `MANIFEST.csv`.

### D16 — Record the frozen Level-2 preprocessing scope

- **Trigger:** Audit of the frozen Level-2 pipeline established that missing-value means were estimated before the internal fit/validation split rather than on the later 80% fit subset alone.
- **Decision:** Retain the frozen policy: feature means were estimated from the complete official outer-training fold before the internal 80%/20% fit/validation split.
- **Boundary:** For each outer fold, that fold's official-test entries and targets were excluded. Covariates later assigned to the internal validation subset contributed to preprocessing statistics, but their targets contributed only to validation monitoring. This is within-fold preprocessing sharing, not official-test leakage. Its influence on validation MAE, validation curves, or best-iteration selection was not quantified. The reported 64% fraction refers only to XGBoost parameter fitting; preprocessing statistics used the complete outer-training fold.
- **Repository evidence:** `notebooks/mp_gap_baseline_v2_run0709.ipynb`, the frozen Level-2 configuration and training records, `README.md`, and the manuscript Methods and Limitations.

### D17 — Preserve frozen manuscript artifacts as immutable release evidence

- **Trigger:** Several historical scripts retain output defaults that overlap populated manuscript result directories, while the released predictions, scores, audits, and figures serve as provenance evidence.
- **Decision:** Treat frozen artifacts in a tagged release as immutable evidence. Direct every reproduction run to a new, initially empty output directory. For legacy scripts with fixed output constants, redirect only the output-path constant in a private copy or run the script in a disposable checkout, and record that operational change. Reproduced outputs must not silently replace frozen manuscript artifacts.
- **Reason:** Separating release evidence from regenerated output prevents accidental overwrite, preserves checksum meaning, and makes disagreements between frozen and reproduced results visible rather than destructive.
- **Repository evidence:** the release-preservation and reproduction-route instructions in `README.md`, script output-directory guards, `MANIFEST.csv`, and `SHA256SUMS.txt`.

---

## Claims deliberately not made

The final study does not claim that:

1. the three levels isolate representation while holding model family, capacity, optimization, and training fraction identical;
2. the PBE targets are experimental truth or that their zero values prove metallicity;
3. a high SHAP value, attention weight, or deletion response establishes a causal physical mechanism;
4. every repeated reduced formula represents a distinct experimentally realizable polymorph;
5. the Level-2 archived raw predictions belong to the frozen Level-2 run;
6. five-fold dispersion measures random-seed uncertainty;
7. the single-fold angle mask estimates a population-wide fraction of graph-model performance;
8. the sparse 8+ eV bin (`n=23`) supports substantive conclusions;
9. an aggregate interval spanning zero establishes equivalence between Level 3 and the repeated-composition oracle;
10. the negative coordination deletion response is convergence-stable or generally beneficial;
11. per-column SHAP density is a representation-invariant correction for group dimensionality or an estimator of marginal deletion response;
12. the pooled median reduction represents a uniform contraction of the error distribution despite the clipping-induced point mass at zero.

## Open follow-up decisions

The original physical goal remains a future program rather than a completed causal claim. The most direct extensions would be:

- formula-grouped and structure-deduplicated evaluation;
- multiple random seeds for all primary models;
- multi-fold factorial graph ablations separating angle values, line-graph topology, depth, and training fraction;
- targeted analysis of the severe-error tail;
- validation against higher-fidelity calculations or experimental gaps;
- prospective calculations or experiments on structures selected from model-generated hypotheses.

## Update policy

Before the first public release, entries may be revised in place to reconcile the ledger with the frozen evidence and final manuscript; material changes are labeled as pre-release revisions where they narrow an earlier interpretation. After a tagged public release, a changed claim, input, or provenance label should be recorded through a dated amendment or new entry rather than by silently rewriting the released rationale. Numerical truth remains in the frozen result files and audit outputs; this document records why those analyses were performed and how they were interpreted.
