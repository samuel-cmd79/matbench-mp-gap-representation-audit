#!/usr/bin/env python3
"""Prepare the two external Zenodo data bundles for the mp_gap release.

Run this script from the root of the public repository, where the source
directories ``gnn_export/`` and ``matbench_cache/`` are present.  The script:

1. validates the two source trees without loading any pickle files;
2. computes a SHA-256 manifest for every extracted data file;
3. creates deterministic ``tar.gz`` archives outside the repository;
4. verifies every archived member against the source manifest;
5. writes archive-level SHA-256 checksums and final release documentation.

The source directories are never modified.  Existing output directories are
never overwritten.  The script uses only the Python standard library and is
compatible with Python 3.9 or newer.
"""

import argparse
import csv
import gzip
import hashlib
import os
import re
import shutil
import sys
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


DOI = "10.5281/zenodo.22038572"
DEFAULT_VERSION = "v1.0.0"
CHUNK_SIZE = 8 * 1024 * 1024
SYSTEM_NAMES = {".DS_Store", "__MACOSX"}
EXPECTED_FILE_COUNTS = {
    "gnn_export": 25,
    "matbench_cache": 80,
}


class ReleaseError(RuntimeError):
    """Raised when a release-safety check fails."""


@dataclass(frozen=True)
class SourceFile:
    bundle: str
    source_path: Path
    relative_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and package gnn_export/ and matbench_cache/ for Zenodo."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Public repository root (default: current directory).",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Release version used in archive names (default: v1.0.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Relative paths are resolved from the repository "
            "root. Default: a sibling directory named mp_gap_zenodo_data_<version>."
        ),
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(1, 10),
        default=6,
        metavar="1-9",
        help="gzip compression level (default: 6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validation and show the plan without hashing or writing files.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    """Python 3.9-compatible Path.is_relative_to()."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return "{} {}".format(int(value), unit)
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    raise AssertionError("unreachable")


def forbidden_system_path(relative: Path) -> bool:
    for part in relative.parts:
        if part in SYSTEM_NAMES or part.startswith("._"):
            return True
    return False


def collect_source_files(repo_root: Path) -> List[SourceFile]:
    files: List[SourceFile] = []
    problems: List[str] = []
    normalized_paths: Dict[str, str] = {}

    for bundle, expected_count in EXPECTED_FILE_COUNTS.items():
        root = repo_root / bundle
        if not root.exists():
            problems.append("missing required directory: {}".format(root))
            continue
        if root.is_symlink():
            problems.append("source directory must not be a symlink: {}".format(root))
            continue
        if not root.is_dir():
            problems.append("required path is not a directory: {}".format(root))
            continue

        bundle_files: List[SourceFile] = []
        for candidate in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            relative_to_root = candidate.relative_to(root)
            release_relative = Path(bundle) / relative_to_root
            release_name = release_relative.as_posix()

            if "\n" in release_name or "\r" in release_name:
                problems.append(
                    "newlines are not allowed in release paths: {!r}".format(
                        release_name
                    )
                )
                continue
            collision_key = unicodedata.normalize("NFC", release_name).casefold()
            prior_name = normalized_paths.get(collision_key)
            if prior_name is not None and prior_name != release_name:
                problems.append(
                    "case/Unicode-normalized path collision: {!r} and {!r}".format(
                        prior_name, release_name
                    )
                )
                continue
            normalized_paths[collision_key] = release_name

            if forbidden_system_path(relative_to_root):
                problems.append(
                    "forbidden macOS metadata remains in source tree: {}".format(
                        release_relative.as_posix()
                    )
                )
                continue
            if candidate.is_symlink():
                problems.append(
                    "symbolic links are not allowed: {}".format(
                        release_relative.as_posix()
                    )
                )
                continue
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                problems.append(
                    "unsupported filesystem entry: {}".format(
                        release_relative.as_posix()
                    )
                )
                continue
            if candidate.suffix.lower() != ".pkl":
                problems.append(
                    "unexpected non-.pkl file: {}".format(
                        release_relative.as_posix()
                    )
                )
                continue

            stat_result = candidate.stat()
            if stat_result.st_size <= 0:
                problems.append(
                    "empty data file is not allowed: {}".format(
                        release_relative.as_posix()
                    )
                )
                continue
            try:
                with candidate.open("rb") as handle:
                    handle.read(1)
            except OSError as error:
                problems.append(
                    "data file is not readable: {} ({})".format(
                        release_relative.as_posix(), error
                    )
                )
                continue
            bundle_files.append(
                SourceFile(
                    bundle=bundle,
                    source_path=candidate,
                    relative_path=release_relative.as_posix(),
                    size_bytes=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                )
            )

        if len(bundle_files) != expected_count:
            problems.append(
                "{} contains {} valid .pkl files; expected {}".format(
                    bundle, len(bundle_files), expected_count
                )
            )
        files.extend(bundle_files)

    if problems:
        raise ReleaseError("Preflight failed:\n- " + "\n- ".join(problems))

    return sorted(files, key=lambda item: item.relative_path)


def hash_source_files(files: Sequence[SourceFile]) -> List[SourceFile]:
    hashed: List[SourceFile] = []
    total = len(files)
    for index, item in enumerate(files, start=1):
        hashed.append(
            SourceFile(
                bundle=item.bundle,
                source_path=item.source_path,
                relative_path=item.relative_path,
                size_bytes=item.size_bytes,
                mtime_ns=item.mtime_ns,
                sha256=sha256_file(item.source_path),
            )
        )
        if index == 1 or index % 10 == 0 or index == total:
            print("  hashed {}/{} files".format(index, total), flush=True)
    return hashed


def archive_name(bundle: str, version: str) -> str:
    if bundle == "gnn_export":
        return "mp_gap_gnn_export_{}.tar.gz".format(version)
    if bundle == "matbench_cache":
        return "mp_gap_matbench_cache_{}.tar.gz".format(version)
    raise ReleaseError("Unknown bundle: {}".format(bundle))


def normalized_tar_info(name: str, size: int, is_directory: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    if is_directory:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o644
        info.size = size
    return info


def directory_names_for(files: Sequence[SourceFile], bundle: str) -> List[str]:
    directories = {bundle}
    for item in files:
        if item.bundle != bundle:
            continue
        parent = Path(item.relative_path).parent
        while parent.as_posix() not in (".", ""):
            directories.add(parent.as_posix())
            parent = parent.parent
    return sorted(directories, key=lambda value: (value.count("/"), value))


def create_archive(
    destination: Path,
    bundle: str,
    files: Sequence[SourceFile],
    compression_level: int,
) -> None:
    members = [item for item in files if item.bundle == bundle]
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=compression_level,
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                mode="w",
                fileobj=gzip_handle,
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for directory in directory_names_for(members, bundle):
                    archive.addfile(
                        normalized_tar_info(directory, size=0, is_directory=True)
                    )
                for item in members:
                    info = normalized_tar_info(
                        item.relative_path,
                        size=item.size_bytes,
                        is_directory=False,
                    )
                    with item.source_path.open("rb") as source_handle:
                        archive.addfile(info, fileobj=source_handle)


def expected_manifest(files: Sequence[SourceFile]) -> Dict[str, SourceFile]:
    return {item.relative_path: item for item in files}


def safe_member_name(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def verify_archive(
    archive_path: Path,
    bundle: str,
    files: Sequence[SourceFile],
) -> None:
    expected = {
        path: item
        for path, item in expected_manifest(files).items()
        if item.bundle == bundle
    }
    seen: Dict[str, bool] = {}

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not safe_member_name(member.name):
                raise ReleaseError(
                    "Unsafe archive member path in {}: {}".format(
                        archive_path.name, member.name
                    )
                )
            if forbidden_system_path(Path(member.name)):
                raise ReleaseError(
                    "Forbidden system file in {}: {}".format(
                        archive_path.name, member.name
                    )
                )
            if member.isdir():
                continue
            if not member.isfile():
                raise ReleaseError(
                    "Unsupported archive member type in {}: {}".format(
                        archive_path.name, member.name
                    )
                )
            if member.name not in expected:
                raise ReleaseError(
                    "Unexpected archive member in {}: {}".format(
                        archive_path.name, member.name
                    )
                )
            if member.name in seen:
                raise ReleaseError(
                    "Duplicate archive member in {}: {}".format(
                        archive_path.name, member.name
                    )
                )
            reference = expected[member.name]
            if member.size != reference.size_bytes:
                raise ReleaseError(
                    "Size mismatch for {} in {}".format(
                        member.name, archive_path.name
                    )
                )

            extracted = archive.extractfile(member)
            if extracted is None:
                raise ReleaseError(
                    "Could not read {} from {}".format(
                        member.name, archive_path.name
                    )
                )
            digest = hashlib.sha256()
            while True:
                chunk = extracted.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != reference.sha256:
                raise ReleaseError(
                    "SHA-256 mismatch for {} in {}".format(
                        member.name, archive_path.name
                    )
                )
            seen[member.name] = True

    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ReleaseError(
            "Archive {} is missing members:\n- {}".format(
                archive_path.name, "\n- ".join(missing)
            )
        )


def check_sources_unchanged(files: Sequence[SourceFile]) -> None:
    changed: List[str] = []
    for item in files:
        stat_result = item.source_path.stat()
        if (
            stat_result.st_size != item.size_bytes
            or stat_result.st_mtime_ns != item.mtime_ns
        ):
            changed.append(item.relative_path)
    if changed:
        raise ReleaseError(
            "Source files changed while packaging:\n- " + "\n- ".join(changed)
        )


def write_manifest(
    destination: Path,
    files: Sequence[SourceFile],
    version: str,
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["archive_name", "bundle", "relative_path", "size_bytes", "sha256"]
        )
        for item in files:
            writer.writerow(
                [
                    archive_name(item.bundle, version),
                    item.bundle,
                    item.relative_path,
                    item.size_bytes,
                    item.sha256,
                ]
            )


def write_archive_checksums(
    destination: Path,
    archives: Sequence[Path],
) -> Mapping[str, str]:
    checksums: Dict[str, str] = {}
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for archive in archives:
            digest = sha256_file(archive)
            checksums[archive.name] = digest
            handle.write("{}  {}\n".format(digest, archive.name))
    return checksums


def bundle_stats(files: Sequence[SourceFile], bundle: str) -> Tuple[int, int]:
    selected = [item for item in files if item.bundle == bundle]
    return len(selected), sum(item.size_bytes for item in selected)


def ensure_free_space(output_parent: Path, files: Sequence[SourceFile]) -> None:
    source_bytes = sum(item.size_bytes for item in files)
    safety_margin = max(256 * 1024 * 1024, int(source_bytes * 0.05))
    required_bytes = source_bytes + safety_margin
    free_bytes = shutil.disk_usage(str(output_parent)).free
    if free_bytes < required_bytes:
        raise ReleaseError(
            "Insufficient free space in {}: {} available, at least {} required".format(
                output_parent,
                human_size(free_bytes),
                human_size(required_bytes),
            )
        )


def data_readme_text(
    files: Sequence[SourceFile],
    version: str,
    archives: Sequence[Path],
    archive_checksums: Mapping[str, str],
) -> str:
    gnn_count, gnn_bytes = bundle_stats(files, "gnn_export")
    cache_count, cache_bytes = bundle_stats(files, "matbench_cache")
    archive_sizes = {archive.name: archive.stat().st_size for archive in archives}
    gnn_archive = archive_name("gnn_export", version)
    cache_archive = archive_name("matbench_cache", version)

    return """# External data bundles for the MatBench `mp_gap` study

## Record information

- Resource type: Dataset
- Reserved version-specific DOI: https://doi.org/{doi}
- Dataset version: `{version}`
- Benchmark task: MatBench v0.1 `mp_gap` (API task key: `matbench_mp_gap`)
- Associated study: *How Much Structure Does Band-Gap Prediction Need? A
  Fold-Controlled Representation–Learner Audit from Composition to Graphs on
  MatBench mp_gap*

The DOI is reserved for this version-specific Zenodo record and may not resolve
until the record is published. Creator, publication date, linked code release,
and formatted citation are maintained in the Zenodo record metadata so that
this file does not become a conflicting second copy.

## Purpose

This record supplies the two large, version-matched data directories excluded
from the lightweight Git repository. They support cache-dependent analyses and
the full retraining routes. They are not required for analyses that operate
only on the frozen predictions and compact result artifacts distributed in Git.

## Distributed bundles

| Archive | Restored directory | Contents | Extracted files | Exact uncompressed bytes | Exact archive bytes | Primary use |
|---|---|---|---:|---:|---:|---|
| `{gnn_archive}` | `gnn_export/` | Train/test crystal structures and identifiers plus training labels for the five official MatBench folds | {gnn_count} | {gnn_bytes} ({gnn_human}) | {gnn_archive_bytes} ({gnn_archive_human}) | Five-fold ALIGNN retraining and fold-0 angle-mask retraining |
| `{cache_archive}` | `matbench_cache/` | Frozen engineered-feature outputs for 5 folds × train/test × 8 featurizers | {cache_count} | {cache_bytes} ({cache_human}) | {cache_archive_bytes} ({cache_archive_human}) | Level-2/BandCenter training and replay, deletion retraining, and cache-dependent Supporting Information audits |

The `gnn_export/` bundle does not contain test-label files. Before packaging,
both source directories were checked for their expected file counts, non-empty
regular `.pkl` files, and absence of symbolic links. macOS metadata files
(`.DS_Store`, `._*`, and `__MACOSX`) are not included.

## Integrity files

This record includes:

- `DATA_README.md` — this document;
- `LICENSES_AND_ATTRIBUTION.md` — license scope, upstream notices, and citations;
- `DATA_FILE_MANIFEST.csv` — one row per extracted data file;
- `LARGE_DATA_SHA256SUMS.txt` — SHA-256 values for the two final archives.

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
{gnn_archive_sha}  {gnn_archive}
{cache_archive_sha}  {cache_archive}
```

After extraction, member-level sizes and SHA-256 values can be checked against
`DATA_FILE_MANIFEST.csv`.

## Installation

Download both archives and the integrity files into one temporary directory.
After the archive-level check succeeds, extract both archives at the root of
the associated Git checkout:

```bash
tar -xzf {gnn_archive}
tar -xzf {cache_archive}
```

The resulting paths must be exactly:

```text
<repository-root>/gnn_export/
<repository-root>/matbench_cache/
```

Do not rename or reorganize these directories unless the corresponding path
constants or command-line arguments in the reproduction scripts are updated.

## Security warning: Python pickle files

Both bundles contain Python pickle files. Loading a pickle can execute arbitrary
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
        doi=DOI,
        version=version,
        gnn_archive=gnn_archive,
        cache_archive=cache_archive,
        gnn_count=gnn_count,
        cache_count=cache_count,
        gnn_bytes=format(gnn_bytes, ","),
        cache_bytes=format(cache_bytes, ","),
        gnn_human=human_size(gnn_bytes),
        cache_human=human_size(cache_bytes),
        gnn_archive_bytes=format(archive_sizes[gnn_archive], ","),
        cache_archive_bytes=format(archive_sizes[cache_archive], ","),
        gnn_archive_human=human_size(archive_sizes[gnn_archive]),
        cache_archive_human=human_size(archive_sizes[cache_archive]),
        gnn_archive_sha=archive_checksums[gnn_archive],
        cache_archive_sha=archive_checksums[cache_archive],
    )


def licenses_and_attribution_text(version: str) -> str:
    return """# Licenses and attribution

## Record

- Dataset version: `{version}`
- Reserved version-specific DOI: https://doi.org/{doi}

This record combines derived data and organizational elements from sources
with different licensing terms. The deposited files are distributed under both
**Creative Commons Attribution 4.0 International (CC BY 4.0)** and the
**MIT License**. The scope of each is described below; neither license displaces
terms that apply to upstream material. Zenodo record metadata is separately
made available under CC0 under Zenodo's general policies.

## License scope

### Materials Project-derived data — CC BY 4.0

The crystal structures and PBE band-gap values originate from the Materials
Project and are redistributed here through the MatBench `mp_gap` benchmark and
derived project artifacts. Materials Project data are made available under
CC BY 4.0:

- Materials Project terms: https://next-gen.materialsproject.org/about/terms
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Reuse must provide appropriate credit, link to the license, and indicate that
changes were made. This record does not claim ownership of the upstream
Materials Project records and does not imply endorsement by Materials Project.

### MatBench `mp_gap` organization — MIT

The upstream `mp_gap` dataset record is attributed to Hacking Materials, is
adapted from the Materials Project database, and is marked as MIT-licensed:

- Upstream dataset record: https://doi.org/10.6084/m9.figshare.9461444
- MatBench source and MIT notice:
  https://github.com/materialsproject/matbench/blob/main/LICENSE

The complete upstream MatBench MIT notice is reproduced here:

```text
MIT License

Copyright (c) 2021 Hacking Materials Research Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Deposited documentation, integrity files, and derived outputs — CC BY 4.0

To the extent that rights are held by the depositor, the new documentation,
archive arrangement, manifest, checksums, and derived numerical feature-cache
outputs contributed in this record are made available under CC BY 4.0. This
statement does not relicense third-party material or remove upstream conditions.

## Changes and processing applied

Relative to the upstream data sources, this record includes the following
processing and organizational changes:

- use of the official five MatBench folds;
- export of train/test crystal structures and identifiers plus training labels;
- serialization of the exported objects as Python pickle files;
- calculation and caching of eight engineered feature families for each fold
  and train/test split;
- organization into the `gnn_export/` and `matbench_cache/` directory trees;
- deterministic `tar.gz` packaging; and
- addition of member-level and archive-level size and SHA-256 metadata.

No test-label files are included in `gnn_export/`. The release packaging did
not regenerate the frozen scientific contents.

## Software dependencies and their licenses

The archives contain data objects, not copies of the MatBench, matminer,
pymatgen, or ALIGNN source distributions. Those packages and other runtime
dependencies retain their own upstream licenses.

- MatBench software: MIT —
  https://github.com/materialsproject/matbench/blob/main/LICENSE
- matminer software: BSD-style LBNL license —
  https://github.com/hackingmaterials/matminer/blob/main/LICENSE
- pymatgen software: MIT —
  https://github.com/materialsproject/pymatgen/blob/master/LICENSE

matminer was used to generate the engineered-feature cache. Its software
license is therefore a dependency notice, not a third record-wide data license.
If any future version embeds third-party source code or packaged model resources,
that version must add the corresponding notices before release.

## Scholarly citation and attribution

Please cite the Zenodo record and the associated study. Please also cite the
following upstream sources as applicable when using these data:

1. A. Jain et al., “Commentary: The Materials Project: A materials genome
   approach to accelerating materials innovation,” *APL Materials* **1**,
   011002 (2013). https://doi.org/10.1063/1.4812323
2. A. Dunn, Q. Wang, A. Ganose, D. Dopp, and A. Jain, “Benchmarking Materials
   Property Prediction Methods: The Matbench Test Set and Automatminer
   Reference Algorithm,” *npj Computational Materials* **6**, 138 (2020).
   https://doi.org/10.1038/s41524-020-00406-3
3. L. Ward et al., “Matminer: An open source toolkit for materials data mining,”
   *Computational Materials Science* **152**, 60–69 (2018).
   https://doi.org/10.1016/j.commatsci.2018.05.018
4. Hacking Materials, “mp_gap,” Figshare dataset (2019).
   https://doi.org/10.6084/m9.figshare.9461444

Featurizer-specific method citations documented by matminer and in the
associated manuscript remain applicable in addition to the core citations
above.

## No endorsement or warranty

Attribution does not imply endorsement by Materials Project, Hacking Materials,
Lawrence Berkeley National Laboratory, or any software author. The materials
are supplied without warranties; users are responsible for verifying integrity
and assessing fitness for their intended use.
""".format(doi=DOI, version=version)


def validate_version(version: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ReleaseError(
            "Invalid version {!r}; use only letters, numbers, '.', '_' and '-'.".format(
                version
            )
        )


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        raise ReleaseError("Repository root does not exist: {}".format(repo_root))

    if args.output_dir is None:
        output_dir = repo_root.parent / "mp_gap_zenodo_data_{}".format(args.version)
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        output_dir = output_dir.resolve()

    if is_relative_to(output_dir, repo_root):
        raise ReleaseError(
            "Output directory must be outside the Git repository: {}".format(
                output_dir
            )
        )
    if output_dir.exists():
        raise ReleaseError(
            "Output directory already exists; nothing was overwritten: {}".format(
                output_dir
            )
        )
    return repo_root, output_dir


def print_plan(
    repo_root: Path,
    output_dir: Path,
    version: str,
    files: Sequence[SourceFile],
) -> None:
    print("Repository root: {}".format(repo_root))
    print("Output directory: {}".format(output_dir))
    print("Reserved DOI: {}".format(DOI))
    print("Release version: {}".format(version))
    for bundle in EXPECTED_FILE_COUNTS:
        count, size_bytes = bundle_stats(files, bundle)
        print(
            "{}: {} files, {} bytes ({})".format(
                bundle, count, size_bytes, human_size(size_bytes)
            )
        )


def run() -> int:
    args = parse_args()
    validate_version(args.version)
    repo_root, output_dir = resolve_paths(args)

    print("[1/5] Validating source directories...", flush=True)
    files = collect_source_files(repo_root)
    print_plan(repo_root, output_dir, args.version, files)
    if args.dry_run:
        print("Dry run passed. No files were written.")
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ensure_free_space(output_dir.parent, files)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".{}.tmp-".format(output_dir.name),
            dir=str(output_dir.parent),
        )
    )

    try:
        print("[2/5] Hashing source files...", flush=True)
        files = hash_source_files(files)
        write_manifest(
            staging_dir / "DATA_FILE_MANIFEST.csv", files, args.version
        )

        print("[3/5] Creating deterministic archives...", flush=True)
        archive_paths: List[Path] = []
        for bundle in EXPECTED_FILE_COUNTS:
            destination = staging_dir / archive_name(bundle, args.version)
            print("  creating {}".format(destination.name), flush=True)
            create_archive(
                destination,
                bundle=bundle,
                files=files,
                compression_level=args.compression_level,
            )
            archive_paths.append(destination)

        print("[4/5] Verifying every archived file...", flush=True)
        for bundle, archive_path in zip(EXPECTED_FILE_COUNTS, archive_paths):
            print("  verifying {}".format(archive_path.name), flush=True)
            verify_archive(archive_path, bundle=bundle, files=files)
        check_sources_unchanged(files)

        print("[5/5] Writing release metadata...", flush=True)
        archive_checksums = write_archive_checksums(
            staging_dir / "LARGE_DATA_SHA256SUMS.txt", archive_paths
        )
        with (staging_dir / "DATA_README.md").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(
                data_readme_text(
                    files,
                    args.version,
                    archive_paths,
                    archive_checksums,
                )
            )
        with (staging_dir / "LICENSES_AND_ATTRIBUTION.md").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(licenses_and_attribution_text(args.version))

        os.replace(str(staging_dir), str(output_dir))
        print("Release package prepared successfully: {}".format(output_dir))
        print("Zenodo draft files:")
        for name in (
            archive_name("gnn_export", args.version),
            archive_name("matbench_cache", args.version),
            "DATA_README.md",
            "LICENSES_AND_ATTRIBUTION.md",
            "DATA_FILE_MANIFEST.csv",
            "LARGE_DATA_SHA256SUMS.txt",
        ):
            print("  {}".format(output_dir / name))
        return 0
    finally:
        if staging_dir.exists():
            shutil.rmtree(str(staging_dir))


def main() -> None:
    try:
        raise SystemExit(run())
    except ReleaseError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
    except OSError as error:
        print("ERROR: filesystem operation failed: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nInterrupted. Source data were not modified.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
