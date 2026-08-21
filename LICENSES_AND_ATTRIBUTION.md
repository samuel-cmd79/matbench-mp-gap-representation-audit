# Licenses and attribution

This repository combines original project material with data and software
interfaces from third parties. The licenses apply by component; no single
license overrides the terms attached to an upstream source.

## Project-authored material

- Original project code is released under the MIT License in `LICENSE`.
- Project-authored documentation, figures, and frozen outputs are released
  under the Creative Commons Attribution 4.0 International license
  ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), to the extent
  that the author holds the relevant rights.
- Third-party data, code, names, and notices embedded in or underlying those
  materials remain subject to their own terms. The two source-informed ALIGNN
  Colab workflows are specifically identified in `THIRD_PARTY_NOTICES.md`.

## Materials Project source data

The MatBench `mp_gap` task is adapted from Materials Project PBE band-gap and
crystal-structure data. Materials Project data are made available under
CC BY 4.0; users must preserve appropriate attribution when redistributing or
reusing them. See the [Materials Project terms of
use](https://next-gen.materialsproject.org/about/terms).

Primary citation:

> Jain, A. et al. Commentary: The Materials Project: A materials genome
> approach to accelerating materials innovation. *APL Materials* **1**,
> 011002 (2013). https://doi.org/10.1063/1.4812323

## MatBench and the `mp_gap` task

The upstream MatBench software repository and the published `mp_gap` task
record are distributed under the MIT License. The complete upstream MatBench
MIT notice is reproduced in `THIRD_PARTY_NOTICES.md`.

- MatBench repository: https://github.com/materialsproject/matbench
- Upstream `mp_gap` record: https://doi.org/10.6084/m9.figshare.9461444

Primary citation:

> Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking materials
> property prediction methods: the Matbench test set and Automatminer
> reference algorithm. *npj Computational Materials* **6**, 138 (2020).
> https://doi.org/10.1038/s41524-020-00406-3

## ALIGNN

The repository does not redistribute the ALIGNN package. Two project-authored
Colab workflows import ALIGNN and are source-informed by its public training,
data-loading, configuration, and model interfaces. ALIGNN is a NIST work made
available under the terms in its official `LICENSE.rst`; those terms are
reproduced in full in `THIRD_PARTY_NOTICES.md`.

- ALIGNN repository and terms: https://github.com/usnistgov/alignn

Primary citation:

> Choudhary, K. & DeCost, B. Atomistic Line Graph Neural Network for improved
> materials property predictions. *npj Computational Materials* **7**, 185
> (2021). https://doi.org/10.1038/s41524-021-00650-1

## Feature-generation software

The frozen engineered-feature cache was generated with MatBench, matminer,
and their dependencies. Those packages are dependencies rather than bundled
source distributions; their licenses continue to govern their software.

Primary matminer citation:

> Ward, L. et al. Matminer: An open source toolkit for materials data mining.
> *Computational Materials Science* **152**, 60–69 (2018).
> https://doi.org/10.1016/j.commatsci.2018.05.018

## Transformations and changes

Relative to the upstream data and software interfaces, this release records
the following project-specific work:

- use of the five official MatBench `mp_gap` outer folds;
- serialization of fold-specific structures, identifiers, and training labels
  for external ALIGNN retraining, without test-label files in `gnn_export`;
- generation and freezing of eight engineered-feature families for each
  fold and train/test split in `matbench_cache`;
- project-specific RF, XGBoost, ALIGNN, clipping, error, SHAP, deletion,
  repeated-composition, and provenance analyses; and
- deterministic packaging, extracted-file manifests, and archive-level
  SHA-256 verification for the external bundles.

These statements describe provenance and license scope; they do not imply
endorsement by Materials Project, Hacking Materials Research Group, NIST, or
the authors of the cited software and datasets.
