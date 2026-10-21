#!/usr/bin/env python3
"""Prepare the v1.1.0 Zenodo data update for the mp_gap release.

Version 1.1.0 adds one archive, the frozen Level-1 composition-only (Magpie)
feature cache ``outputs_v1_run0709/features_cache.pkl``.  The two v1.0.0
archives are carried over byte-for-byte and are NOT rebuilt by this script.

Run from the root of the public repository::

    python scripts/release/prepare_zenodo_data_update_v1_1_0.py \
        --doi 10.5281/zenodo.NNNNNNNN \
        --previous-dir ../mp_gap_zenodo_data_v1.0.0

``--previous-dir`` must contain the three small v1.0.0 record files
``DATA_FILE_MANIFEST.csv``, ``LARGE_DATA_SHA256SUMS.txt`` and ``DATA_README.md``
(downloaded from the v1.0.0 record or kept from the original packaging run).

The script:

1. validates the Level-1 cache file without loading the pickle;
2. checks the v1.0.0 record files against the approved v1.0.0 archive hashes;
3. packages the cache with the same deterministic archive routine as
   ``prepare_zenodo_data_release.py`` and verifies the archived member;
4. writes the combined v1.1.0 manifest, archive checksums, DATA_README.md and
   LICENSES_AND_ATTRIBUTION.md to a new directory outside the repository.

Source files are never modified and existing output directories are never
overwritten.  Python 3.9+ standard library only.
"""

import argparse
import csv
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

VERSION = "v1.1.0"
PREVIOUS_VERSION = "v1.0.0"
PREVIOUS_DOI = "10.5281/zenodo.22038572"
CACHE_RELATIVE = "outputs_v1_run0709/features_cache.pkl"
CACHE_BUNDLE = "outputs_v1_run0709"
CACHE_ARCHIVE = "mp_gap_v1_features_cache_{}.tar.gz".format(VERSION)
PREVIOUS_ARCHIVES = {
    "mp_gap_gnn_export_v1.0.0.tar.gz":
        "4f83bebb4bb9d14c2a6708806bc7620cd87f531a8202fc0b3c54a7b051b4201c",
    "mp_gap_matbench_cache_v1.0.0.tar.gz":
        "1dea7d2dc99529e75046e6a402783bb935ad889ac8df8715afb318589a238d9b",
}
PREVIOUS_FILE_COUNTS = {"gnn_export": 25, "matbench_cache": 80}
MANIFEST_HEADER = ["archive_name", "bundle", "relative_path", "size_bytes", "sha256"]


class ReleaseError(RuntimeError):
    pass


def load_v1_tool():
    path = Path(__file__).resolve().parent / "prepare_zenodo_data_release.py"
    spec = importlib.util.spec_from_file_location("prepare_zenodo_data_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description="Prepare the v1.1.0 Zenodo data update.")
    p.add_argument("--doi", required=True, help="Reserved DOI of the v1.1.0 record.")
    p.add_argument("--previous-dir", type=Path, required=True,
                   help="Directory with the v1.0.0 DATA_FILE_MANIFEST.csv, "
                        "LARGE_DATA_SHA256SUMS.txt and DATA_README.md.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: sibling directory mp_gap_zenodo_data_v1.1.0.")
    p.add_argument("--compression-level", type=int, choices=range(1, 10), default=6,
                   metavar="1-9")
    return p.parse_args()


def read_previous(previous_dir: Path) -> Tuple[List[List[str]], str, Dict[str, int]]:
    """Return (v1.0.0 manifest rows, v1.0.0 checksum text, v1.0.0 archive byte sizes)."""
    sums_path = previous_dir / "LARGE_DATA_SHA256SUMS.txt"
    expected_sums = "".join("{}  {}\n".format(h, n) for n, h in PREVIOUS_ARCHIVES.items())
    sums = sums_path.read_text(encoding="ascii")
    if sums != expected_sums:
        raise ReleaseError("{} does not match the approved v1.0.0 archive hashes".format(sums_path))

    with (previous_dir / "DATA_FILE_MANIFEST.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    if header != MANIFEST_HEADER:
        raise ReleaseError("unexpected v1.0.0 manifest header: {}".format(header))
    counts: Dict[str, int] = {}
    for row in rows:
        if len(row) != 5 or row[0] not in PREVIOUS_ARCHIVES:
            raise ReleaseError("unexpected v1.0.0 manifest row: {}".format(row))
        counts[row[1]] = counts.get(row[1], 0) + 1
    if counts != PREVIOUS_FILE_COUNTS:
        raise ReleaseError("v1.0.0 manifest file counts {} != {}".format(counts, PREVIOUS_FILE_COUNTS))

    readme = (previous_dir / "DATA_README.md").read_text(encoding="utf-8")
    archive_bytes: Dict[str, int] = {}
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6:
            name = cells[0].strip("`")
            if name in PREVIOUS_ARCHIVES:
                m = re.match(r"([\d,]+)", cells[5])
                if m:
                    archive_bytes[name] = int(m.group(1).replace(",", ""))
    if set(archive_bytes) != set(PREVIOUS_ARCHIVES):
        raise ReleaseError("could not read v1.0.0 archive sizes from {}".format(previous_dir / "DATA_README.md"))
    return rows, sums, archive_bytes


def validate_cache(repo_root: Path):
    path = repo_root / CACHE_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise ReleaseError("missing or non-regular cache file: {}".format(path))
    st = path.stat()
    if st.st_size <= 0:
        raise ReleaseError("empty cache file: {}".format(path))
    with path.open("rb") as fh:
        fh.read(1)
    return path, st


def data_readme_text(v1, doi, prev_rows, prev_bytes, cache, cache_archive_bytes, sums) -> str:
    def bundle_bytes(bundle):
        return sum(int(r[3]) for r in prev_rows if r[1] == bundle)
    hs = v1.human_size
    gnn_a, cache_a = list(PREVIOUS_ARCHIVES)
    gnn_b, mb_b = bundle_bytes("gnn_export"), bundle_bytes("matbench_cache")
    sums_block = sums.rstrip("\n")
    return """# External data bundles for the MatBench `mp_gap` study

## Record information

- Resource type: Dataset
- Reserved version-specific DOI: https://doi.org/{doi}
- Dataset version: `{version}`
- Previous version: `{prev_version}` (https://doi.org/{prev_doi})
- Benchmark task: MatBench v0.1 `mp_gap` (API task key: `matbench_mp_gap`)
- Associated study: *How Much Structure Does Band-Gap Prediction Need? A
  Fold-Controlled Representation–Learner Audit from Composition to Graphs on
  MatBench mp_gap*

The DOI is reserved for this version-specific Zenodo record and may not resolve
until the record is published. Creator, publication date, linked code release,
and formatted citation are maintained in the Zenodo record metadata so that
this file does not become a conflicting second copy.

## Changes in {version}

Version {version} adds `{cache_archive}`, the frozen Level-1 composition-only
(Magpie) feature cache used by the Level-1 training notebook and its
deterministic replay. This file was not included in {prev_version}. The two
{prev_version} archives are carried over byte-for-byte; their SHA-256 values are
unchanged.

## Purpose

This record supplies the large, version-matched data files excluded from the
lightweight Git repository. They support cache-dependent analyses and the full
retraining routes. They are not required for analyses that operate only on the
frozen predictions and compact result artifacts distributed in Git.

## Distributed bundles

| Archive | Restored path | Contents | Extracted files | Exact uncompressed bytes | Exact archive bytes | Primary use |
|---|---|---|---:|---:|---:|---|
| `{gnn_a}` | `gnn_export/` | Train/test crystal structures and identifiers plus training labels for the five official MatBench folds | 25 | {gnn_b} ({gnn_bh}) | {gnn_ab} ({gnn_abh}) | Five-fold ALIGNN retraining and fold-0 angle-mask retraining |
| `{cache_a}` | `matbench_cache/` | Frozen engineered-feature outputs for 5 folds × train/test × 8 featurizers | 80 | {mb_b} ({mb_bh}) | {mb_ab} ({mb_abh}) | Level-2/BandCenter training and replay, deletion retraining, and cache-dependent Supporting Information audits |
| `{cache_archive}` | `{cache_rel}` | Frozen Level-1 composition-only (Magpie) feature cache for the five official folds (train and test) | 1 | {v1_b} ({v1_bh}) | {v1_ab} ({v1_abh}) | Level-1 training and deterministic replay, and Level-1 grouped cross-validation |

The `gnn_export/` bundle does not contain test-label files. Before packaging,
the source files were checked for their expected file counts, non-empty
regular `.pkl` files, and absence of symbolic links. macOS metadata files
(`.DS_Store`, `._*`, and `__MACOSX`) are not included.

## Integrity files

This record includes:

- `DATA_README.md` — this document;
- `LICENSES_AND_ATTRIBUTION.md` — license scope, upstream notices, and citations;
- `DATA_FILE_MANIFEST.csv` — one row per extracted data file;
- `LARGE_DATA_SHA256SUMS.txt` — SHA-256 values for the three final archives.

`DATA_FILE_MANIFEST.csv` uses these columns:

```text
archive_name,bundle,relative_path,size_bytes,sha256
```

Verify the downloaded archives before extraction:

```bash
shasum -a 256 -c LARGE_DATA_SHA256SUMS.txt
```

The expected archive SHA-256 values are:

```text
{sums_block}
```

After extraction, member-level sizes and SHA-256 values can be checked against
`DATA_FILE_MANIFEST.csv`.

## Installation

Download the archives and the integrity files into one temporary directory.
After the archive-level check succeeds, extract the archives at the root of the
associated Git checkout:

```bash
tar -xzf {gnn_a}
tar -xzf {cache_a}
tar -xzf {cache_archive}
```

The resulting paths must be exactly:

```text
<repository-root>/gnn_export/
<repository-root>/matbench_cache/
<repository-root>/{cache_rel}
```

The Level-1 cache archive restores a single file into the existing
`{cache_bundle}/` directory of the Git checkout. Do not rename or reorganize
these paths unless the corresponding path constants or command-line arguments
in the reproduction scripts are updated.

## Security warning: Python pickle files

All bundles contain Python pickle files. Loading a pickle can execute arbitrary
code. Only load files obtained from this DOI record after verifying both the
archive-level and extracted-file SHA-256 values. Do not load renamed copies,
files from third-party mirrors, or files whose checksums do not match. Use the
documented project environments, preferably in an isolated environment.

## Provenance and transformations

The bundles are frozen intermediate artifacts derived from the official
MatBench `mp_gap` dataset, fold definitions, and associated Materials Project
structures and PBE band-gap values.

- `gnn_export/` preserves train/test structures and identifiers plus training
  labels for each official fold as serialized inputs to the ALIGNN workflow.
- `matbench_cache/` preserves version-sensitive engineered-feature outputs used
  by the tree-model and cache-dependent workflows.
- `{cache_rel}` preserves the Level-1 composition-only (Magpie) feature
  matrices used by the frozen Level-1 run, its deterministic replay, and the
  Level-1 grouped cross-validation.
- The release process repackaged these frozen files into deterministic archives
  and added member-level and archive-level integrity metadata. It did not
  regenerate the scientific contents.

These files preserve the computational provenance of the reported frozen runs
and avoid silent changes caused by regenerating version-sensitive intermediate
representations with a different software stack. See the linked code repository,
environment files, manuscript, and Supporting Information for model definitions,
reproduction routes, and interpretation.

## License and attribution

This is a mixed-origin data record distributed under both CC BY 4.0 and MIT
because different upstream components carry different terms. See
`LICENSES_AND_ATTRIBUTION.md` for the applicable scope, source links, changes,
software-dependency notices, and citation guidance. No license in this record
replaces an applicable upstream license.

## Citation

After publication, use the citation exported from the Zenodo record for DOI
`{doi}`. Please also cite the associated study and the
applicable upstream Materials Project, MatBench, and matminer sources listed in
`LICENSES_AND_ATTRIBUTION.md`.
""".format(
        doi=doi, version=VERSION, prev_version=PREVIOUS_VERSION, prev_doi=PREVIOUS_DOI,
        cache_archive=CACHE_ARCHIVE, cache_rel=CACHE_RELATIVE, cache_bundle=CACHE_BUNDLE,
        gnn_a=gnn_a, cache_a=cache_a, sums_block=sums_block,
        gnn_b=format(gnn_b, ","), gnn_bh=hs(gnn_b),
        mb_b=format(mb_b, ","), mb_bh=hs(mb_b),
        gnn_ab=format(prev_bytes[gnn_a], ","), gnn_abh=hs(prev_bytes[gnn_a]),
        mb_ab=format(prev_bytes[cache_a], ","), mb_abh=hs(prev_bytes[cache_a]),
        v1_b=format(cache.size_bytes, ","), v1_bh=hs(cache.size_bytes),
        v1_ab=format(cache_archive_bytes, ","), v1_abh=hs(cache_archive_bytes),
    )


def licenses_text(v1, doi) -> str:
    v1.DOI = doi
    text = v1.licenses_and_attribution_text(VERSION)
    edits = [
        ("- calculation and caching of eight engineered feature families for each fold\n"
         "  and train/test split;\n",
         "- calculation and caching of eight engineered feature families for each fold\n"
         "  and train/test split;\n"
         "- calculation and caching of Level-1 composition-only (Magpie) feature\n"
         "  matrices for each fold and train/test split;\n"),
        ("- organization into the `gnn_export/` and `matbench_cache/` directory trees;\n",
         "- organization into the `gnn_export/` and `matbench_cache/` directory trees\n"
         "  and the Level-1 cache file `{}`;\n".format(CACHE_RELATIVE)),
    ]
    for old, new in edits:
        if text.count(old) != 1:
            raise ReleaseError("license template changed; cannot apply v1.1.0 edit")
        text = text.replace(old, new)
    return text


def run() -> int:
    args = parse_args()
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", args.doi) or args.doi == PREVIOUS_DOI:
        raise ReleaseError("--doi must be the new reserved Zenodo DOI, e.g. 10.5281/zenodo.12345678")
    v1 = load_v1_tool()
    repo_root = args.repo_root.expanduser().resolve()
    out = (args.output_dir.expanduser() if args.output_dir else repo_root.parent / "mp_gap_zenodo_data_{}".format(VERSION))
    if not out.is_absolute():
        out = repo_root / out
    out = out.resolve()
    if v1.is_relative_to(out, repo_root):
        raise ReleaseError("output directory must be outside the Git repository: {}".format(out))
    if out.exists():
        raise ReleaseError("output directory already exists; nothing was overwritten: {}".format(out))

    print("[1/5] Validating inputs...", flush=True)
    prev_rows, prev_sums, prev_bytes = read_previous(args.previous_dir.expanduser().resolve())
    cache_path, st = validate_cache(repo_root)
    print("  Level-1 cache: {} bytes ({})".format(st.st_size, v1.human_size(st.st_size)))

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".{}.tmp-".format(out.name), dir=str(out.parent)))
    try:
        print("[2/5] Hashing the Level-1 cache...", flush=True)
        cache = v1.SourceFile(bundle=CACHE_BUNDLE, source_path=cache_path,
                              relative_path=CACHE_RELATIVE, size_bytes=st.st_size,
                              mtime_ns=st.st_mtime_ns, sha256=v1.sha256_file(cache_path))

        print("[3/5] Creating deterministic archive {}...".format(CACHE_ARCHIVE), flush=True)
        archive = staging / CACHE_ARCHIVE
        v1.create_archive(archive, bundle=CACHE_BUNDLE, files=[cache],
                          compression_level=args.compression_level)

        print("[4/5] Verifying the archived file...", flush=True)
        v1.verify_archive(archive, bundle=CACHE_BUNDLE, files=[cache])
        v1.check_sources_unchanged([cache])

        print("[5/5] Writing v1.1.0 record files...", flush=True)
        archive_sha = v1.sha256_file(archive)
        sums = prev_sums + "{}  {}\n".format(archive_sha, CACHE_ARCHIVE)
        (staging / "LARGE_DATA_SHA256SUMS.txt").write_text(sums, encoding="ascii")
        with (staging / "DATA_FILE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(MANIFEST_HEADER)
            w.writerows(prev_rows)
            w.writerow([CACHE_ARCHIVE, CACHE_BUNDLE, CACHE_RELATIVE, cache.size_bytes, cache.sha256])
        (staging / "DATA_README.md").write_text(
            data_readme_text(v1, args.doi, prev_rows, prev_bytes, cache, archive.stat().st_size, sums),
            encoding="utf-8")
        (staging / "LICENSES_AND_ATTRIBUTION.md").write_text(licenses_text(v1, args.doi), encoding="utf-8")

        os.replace(str(staging), str(out))
        print("\nRelease update prepared: {}".format(out))
        print("Upload to the v1.1.0 Zenodo draft (the two v1.0.0 archives are imported, not re-uploaded):")
        for name in (CACHE_ARCHIVE, "DATA_README.md", "LICENSES_AND_ATTRIBUTION.md",
                     "DATA_FILE_MANIFEST.csv", "LARGE_DATA_SHA256SUMS.txt"):
            print("  {}".format(out / name))
        print("\nRepository LARGE_DATA_SHA256SUMS.txt must equal:\n" + sums)
        print("Level-1 cache file sha256: {}".format(cache.sha256))
        return 0
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))


def main():
    try:
        raise SystemExit(run())
    except ReleaseError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
    except OSError as error:
        print("ERROR: filesystem operation failed: {}".format(error), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
