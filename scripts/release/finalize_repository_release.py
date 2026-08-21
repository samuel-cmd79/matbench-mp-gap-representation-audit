#!/usr/bin/env python3
"""Validate and finalize the lightweight public repository release.

The public file set is obtained from Git, including tracked files and
non-ignored untracked files.  ``--write`` atomically regenerates
``MANIFEST.csv`` and ``SHA256SUMS.txt`` after all release gates pass.
``--check`` performs the same validation and compares both generated files
byte-for-byte without writing anything.

Run from anywhere inside the public Git repository, for example::

    python scripts/release/finalize_repository_release.py --write
    python scripts/release/finalize_repository_release.py --check

Python 3.9+ and the standard library are sufficient.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Sequence, Set, Tuple


MAX_FILE_BYTES = 100 * 1024 * 1024
MANIFEST_PATH = "MANIFEST.csv"
SHA256_PATH = "SHA256SUMS.txt"
SCRIPT_INDEX_PATH = "SCRIPT_INDEX.csv"
LARGE_SHA256_PATH = "LARGE_DATA_SHA256SUMS.txt"
FINALIZER_PATH = "scripts/release/finalize_repository_release.py"
DATA_DOI = "10.5281/zenodo.22038572"

REQUIRED_GITIGNORE_LINES = {
    "/_release_work/",
    "/gnn_export/",
    "/matbench_cache/",
    "/mp_gap_zenodo_data_*/",
    "/reproduction_runs/",
    "/reviewer_packages/",
    "/mp_gap_gnn_export_*.tar.gz",
    "/mp_gap_matbench_cache_*.tar.gz",
    "/mp_gap_repo_text_review_*.tar.gz",
}

FORBIDDEN_PUBLIC_TOP_LEVEL = {
    "_release_work",
    "gnn_export",
    "matbench_cache",
    "reproduction_runs",
    "reviewer_packages",
}

FORBIDDEN_PUBLIC_TOP_LEVEL_PREFIXES = (
    "mp_gap_reviewer_code_",
    "mp_gap_zenodo_data_",
)

FORBIDDEN_PUBLIC_ROOT_PATTERNS = (
    "mp_gap_gnn_export_*.tar.gz",
    "mp_gap_matbench_cache_*.tar.gz",
    "mp_gap_repo_text_review_*.tar.gz",
)

REQUIRED_RELEASE_FILES = (
    ".gitignore",
    "README.md",
    "LICENSE",
    "LICENSES_AND_ATTRIBUTION.md",
    "THIRD_PARTY_NOTICES.md",
    "DECISION_LEDGER.md",
    "ARCHIVE_ONLY.txt",
    SCRIPT_INDEX_PATH,
    LARGE_SHA256_PATH,
    "environments/README.md",
    FINALIZER_PATH,
    "scripts/release/prepare_zenodo_data_release.py",
)

EXPECTED_LARGE_SHA256 = (
    "4f83bebb4bb9d14c2a6708806bc7620cd87f531a8202fc0b3c54a7b051b4201c"
    "  mp_gap_gnn_export_v1.0.0.tar.gz\n"
    "1dea7d2dc99529e75046e6a402783bb935ad889ac8df8715afb318589a238d9b"
    "  mp_gap_matbench_cache_v1.0.0.tar.gz\n"
)

README_PLACEHOLDERS = (
    re.compile(r"\b(?:TBD|TODO|FIXME|XXX)\b", re.IGNORECASE),
    re.compile(r"\bto be inserted\b", re.IGNORECASE),
    re.compile(r"\bbefore public release\b", re.IGNORECASE),
    re.compile(r"\bupdate this block\b", re.IGNORECASE),
    re.compile(r"\badd an explicit\s+`?LICENSE`?", re.IGNORECASE),
    re.compile(
        r"\[[^\]\n]*(?:insert|placeholder|author|final study title)[^\]\n]*\]",
        re.IGNORECASE,
    ),
)

# The finalizer is omitted from its own content scan because it necessarily
# contains literal detection signatures.  It remains subject to every other
# gate and is included in both release metadata products.
CONTENT_SCAN_EXCLUSIONS = {FINALIZER_PATH}

SENSITIVE_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    (
        "private macOS/Linux absolute path",
        re.compile(
            r"(?:file:(?://)?|(?<![A-Za-z0-9:/]))"
            r"/(?:Users|home|root)/[^\s'\"<>`]+",
            re.IGNORECASE,
        ),
    ),
    (
        "private Windows user path",
        re.compile(
            r"(?<![A-Za-z0-9])[A-Z]:(?:\\+|/+)Users(?:\\+|/+)[^\s'\"<>`]+",
            re.IGNORECASE,
        ),
    ),
    ("AWS access-key identifier", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    (
        "OpenAI-style secret key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "private-key block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
    ),
    (
        "credential assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
            r"password|passwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "ChatGPT/OpenAI signed-content URL",
        re.compile(
            r"(?:oaiusercontent\.com|"
            r"oai[a-z0-9-]*(?:\.blob\.core\.windows\.net|\.s3\.[^/\s]+)|"
            r"chatgpt\.com/backend-api/(?:files|estuary))",
            re.IGNORECASE,
        ),
    ),
)

SENSITIVE_SENTINELS = tuple(
    token.lower()
    for token in (
        b"/Users/",
        b"/home/",
        b"/root/",
        b"Users\\",
        b"AKIA",
        b"ASIA",
        b"ghp_",
        b"gho_",
        b"ghu_",
        b"ghs_",
        b"ghr_",
        b"github_pat_",
        b"sk-",
        b"AIza",
        b"xox",
        b"hf_",
        b"PRIVATE KEY",
        b"api_key",
        b"api-key",
        b"apikey",
        b"access_token",
        b"access-token",
        b"accesstoken",
        b"auth_token",
        b"auth-token",
        b"authtoken",
        b"client_secret",
        b"client-secret",
        b"clientsecret",
        b"password",
        b"passwd",
        b"oaiusercontent.com",
        b"blob.core.windows.net",
        b".s3.",
        b"chatgpt.com/backend-api/",
    )
)


class ReleaseValidationError(RuntimeError):
    """Raised when Git cannot provide a safe public file set."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="validate, then atomically write MANIFEST.csv and SHA256SUMS.txt",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate and compare release metadata without writing",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="path inside the public Git repository (default: current directory)",
    )
    return parser.parse_args()


def run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", str(root)] + list(arguments)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def find_repository_root(start: Path) -> Path:
    result = run_git(start.resolve(), ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseValidationError(
            "not inside a Git repository{}".format(": " + detail if detail else "")
        )
    try:
        return Path(result.stdout.decode("utf-8", errors="strict").strip()).resolve()
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError("repository root is not valid UTF-8") from exc


def git_public_files(root: Path) -> List[str]:
    result = run_git(
        root,
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"],
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseValidationError("git ls-files failed: " + detail)
    try:
        names = [
            item.decode("utf-8", errors="strict")
            for item in result.stdout.split(b"\0")
            if item
        ]
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(
            "a public repository path is not valid UTF-8"
        ) from exc
    if len(names) != len(set(names)):
        raise ReleaseValidationError("git returned duplicate public paths")
    return sorted(names)


def generated_paths_are_public(root: Path) -> List[str]:
    issues: List[str] = []
    for rel in (MANIFEST_PATH, SHA256_PATH):
        result = run_git(root, ["check-ignore", "--quiet", "--no-index", "--", rel])
        if result.returncode == 0:
            issues.append("{} is ignored by Git".format(rel))
        elif result.returncode not in (0, 1):
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            issues.append("cannot check ignore status for {}: {}".format(rel, detail))
    return issues


def path_has_symlink_component(root: Path, rel: str) -> bool:
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def validate_public_paths(root: Path, paths: Sequence[str]) -> List[str]:
    issues: List[str] = []
    for rel in paths:
        pure = PurePosixPath(rel)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or "\n" in rel
            or "\r" in rel
        ):
            issues.append("unsafe Git path: {!r}".format(rel))
            continue
        if pure.parts[0] in FORBIDDEN_PUBLIC_TOP_LEVEL or any(
            pure.parts[0].startswith(prefix)
            for prefix in FORBIDDEN_PUBLIC_TOP_LEVEL_PREFIXES
        ):
            issues.append(
                "external or local-run data must not be Git-distributed: {}".format(
                    rel
                )
            )
            continue
        if len(pure.parts) == 1 and any(
            pure.match(pattern) for pattern in FORBIDDEN_PUBLIC_ROOT_PATTERNS
        ):
            issues.append("release/review archive must remain outside Git: " + rel)
            continue
        path = root / rel
        if path_has_symlink_component(root, rel):
            issues.append("symlink is not allowed in the public release: {}".format(rel))
            continue
        if not path.exists():
            issues.append("tracked path is missing from the working tree: {}".format(rel))
            continue
        if not path.is_file():
            issues.append("non-file Git entry is not allowed: {}".format(rel))
            continue
        size = path.stat().st_size
        if size >= MAX_FILE_BYTES:
            issues.append(
                "file is at least 100 MiB and must be externally distributed: {} "
                "({} bytes)".format(rel, size)
            )
    return issues


def validate_gitignore(root: Path) -> List[str]:
    path = root / ".gitignore"
    if not path.is_file():
        return [".gitignore is unavailable"]
    try:
        lines = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except (OSError, UnicodeError) as exc:
        return ["cannot read .gitignore: {}".format(exc)]
    missing = sorted(REQUIRED_GITIGNORE_LINES - lines)
    if missing:
        return [".gitignore is missing required rule(s): " + ", ".join(missing)]
    return []


def validate_required_files(public_paths: Set[str]) -> List[str]:
    return [
        "required release file is missing or ignored: {}".format(rel)
        for rel in REQUIRED_RELEASE_FILES
        if rel not in public_paths
    ]


def validate_readme(root: Path) -> List[str]:
    issues: List[str] = []
    path = root / "README.md"
    if not path.is_file():
        return ["README.md is unavailable for DOI and placeholder validation"]
    text = path.read_text(encoding="utf-8")
    if DATA_DOI not in text:
        issues.append("README.md does not contain the version-specific data DOI " + DATA_DOI)
    for pattern in README_PLACEHOLDERS:
        match = pattern.search(text)
        if match is not None:
            line = text.count("\n", 0, match.start()) + 1
            issues.append(
                "README.md:{} contains a release placeholder ({!r})".format(
                    line, match.group(0)[:80]
                )
            )
    return issues


def validate_script_index(
    root: Path, public_paths: Set[str]
) -> List[str]:
    issues: List[str] = []
    index_path = root / SCRIPT_INDEX_PATH
    if not index_path.is_file():
        return [SCRIPT_INDEX_PATH + " is unavailable"]
    try:
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
    except (OSError, UnicodeError, csv.Error) as exc:
        return ["cannot parse {}: {}".format(SCRIPT_INDEX_PATH, exc)]

    expected_fields = [
        "filename",
        "publication_status",
        "category",
        "role",
        "source_path",
    ]
    if fieldnames != expected_fields:
        issues.append(
            "{} header must be exactly {}".format(
                SCRIPT_INDEX_PATH, ",".join(expected_fields)
            )
        )
        return issues

    indexed_paths: List[str] = []
    indexed_names: List[str] = []
    row_by_path: Dict[str, Dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        empty = [name for name in expected_fields if not (row.get(name) or "").strip()]
        if empty:
            issues.append(
                "{}:{} has empty field(s): {}".format(
                    SCRIPT_INDEX_PATH, number, ", ".join(empty)
                )
            )
            continue
        source_path = row["source_path"].strip()
        filename = row["filename"].strip()
        indexed_paths.append(source_path)
        indexed_names.append(filename)
        row_by_path[source_path] = row
        if filename != PurePosixPath(source_path).name:
            issues.append(
                "{}:{} filename does not match source_path basename".format(
                    SCRIPT_INDEX_PATH, number
                )
            )
        if source_path not in public_paths:
            issues.append(
                "{}:{} path is missing or ignored: {}".format(
                    SCRIPT_INDEX_PATH, number, source_path
                )
            )
        if PurePosixPath(source_path).suffix.lower() not in (".py", ".ipynb"):
            issues.append(
                "{}:{} source_path is not a Python script or notebook".format(
                    SCRIPT_INDEX_PATH, number
                )
            )

    if len(indexed_paths) != len(set(indexed_paths)):
        duplicates = sorted(
            path for path in set(indexed_paths) if indexed_paths.count(path) > 1
        )
        issues.append("duplicate source_path values: " + ", ".join(duplicates))
    if len(indexed_names) != len(set(indexed_names)):
        duplicates = sorted(
            name for name in set(indexed_names) if indexed_names.count(name) > 1
        )
        issues.append("duplicate filename values: " + ", ".join(duplicates))

    public_code = {
        rel
        for rel in public_paths
        if PurePosixPath(rel).suffix.lower() in (".py", ".ipynb")
    }
    indexed = set(indexed_paths)
    missing = sorted(public_code - indexed)
    extra = sorted(indexed - public_code)
    if missing:
        issues.append("scripts/notebooks missing from SCRIPT_INDEX.csv: " + ", ".join(missing))
    if extra:
        issues.append("SCRIPT_INDEX.csv contains non-public paths: " + ", ".join(extra))

    angle_path = "scripts/colab/v4_alignn_matbench_mp_gap_angle_zero_fold0.py"
    angle_row = row_by_path.get(angle_path)
    required_wording = "configuration-matched numerical angle-mask control"
    if angle_row is not None and required_wording not in angle_row["role"].lower():
        issues.append(
            "{} must use the wording {!r} for {}".format(
                SCRIPT_INDEX_PATH, required_wording, angle_path
            )
        )
    return issues


def validate_large_checksums(root: Path) -> List[str]:
    path = root / LARGE_SHA256_PATH
    if not path.is_file():
        return [LARGE_SHA256_PATH + " is unavailable"]
    try:
        actual = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        return ["cannot read {}: {}".format(LARGE_SHA256_PATH, exc)]
    if actual != EXPECTED_LARGE_SHA256:
        return [
            LARGE_SHA256_PATH
            + " must contain exactly the two approved v1.0.0 archive hashes"
        ]
    return []


def scan_sensitive_content(root: Path, paths: Sequence[str]) -> List[str]:
    issues: List[str] = []
    runtime_patterns: List[Tuple[str, re.Pattern[str]]] = []
    runtime_sentinels: List[bytes] = []
    node = platform.node().strip()
    private_host_markers = {node, node.split(".", 1)[0]} if node else set()
    for marker in sorted(private_host_markers):
        if len(marker) < 4 or marker.lower() in {"localhost", "localhost.localdomain"}:
            continue
        runtime_patterns.append(
            (
                "local host name",
                re.compile(
                    r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])".format(
                        re.escape(marker)
                    ),
                    re.IGNORECASE,
                ),
            )
        )
        runtime_sentinels.append(marker.encode("utf-8").lower())

    scan_patterns = tuple(SENSITIVE_PATTERNS) + tuple(runtime_patterns)
    scan_sentinels = tuple(SENSITIVE_SENTINELS) + tuple(runtime_sentinels)
    for rel in paths:
        if rel in CONTENT_SCAN_EXCLUSIONS:
            continue
        path = root / rel
        # Never follow a path that already fails the structural release gate.
        # This also prevents reading a misplaced multi-gigabyte artifact merely
        # to report the same size error a second time.
        if path_has_symlink_component(root, rel):
            continue
        try:
            if not path.is_file() or path.stat().st_size >= MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            issues.append("cannot scan {}: {}".format(rel, exc))
            continue
        lowered = raw.lower()
        if not any(sentinel in lowered for sentinel in scan_sentinels):
            continue
        text = raw.decode("utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines() or [text], start=1):
            # Public Google Colab paths under /content are intentionally allowed.
            # They do not match the private home-directory indicators below.
            for label, pattern in scan_patterns:
                if pattern.search(line) is not None:
                    issues.append(
                        "{}:{}: {}; matched content redacted".format(
                            rel, line_number, label
                        )
                    )
                    if len(issues) >= 100:
                        issues.append("sensitive-content diagnostics truncated at 100 matches")
                        return issues
    return issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_role(path: str) -> str:
    pure = PurePosixPath(path)
    suffix = pure.suffix.lower()
    first = pure.parts[0] if pure.parts else ""
    if path in {"LICENSE", "LICENSES_AND_ATTRIBUTION.md", "THIRD_PARTY_NOTICES.md"}:
        return "license-and-attribution"
    if path in {SCRIPT_INDEX_PATH, LARGE_SHA256_PATH}:
        return "release-metadata"
    if path == ".gitignore":
        return "repository-configuration"
    if path in {"DECISION_LEDGER.md", "ARCHIVE_ONLY.txt"}:
        return "provenance"
    if path == "README.md" or first == "environments":
        return "documentation-and-environment"
    if first == "archive":
        return "archived-diagnostic"
    if path.startswith("scripts/release/"):
        return "release-tooling"
    if suffix in {".py", ".ipynb", ".sh"}:
        return "analysis-code"
    if first == "figure" or suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure"
    if first.startswith("result") or first.startswith("output") or first.startswith(
        "matbench_output"
    ):
        return "frozen-output"
    if suffix in {".npz", ".npy", ".pkl", ".csv", ".json"}:
        return "data-or-result"
    return "repository-file"


def build_manifest(root: Path, public_paths: Sequence[str]) -> bytes:
    included = sorted(
        rel for rel in public_paths if rel not in {MANIFEST_PATH, SHA256_PATH}
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["path", "role", "size_bytes", "sha256"])
    for rel in included:
        path = root / rel
        writer.writerow(
            [rel, release_role(rel), path.stat().st_size, sha256_file(path)]
        )
    return buffer.getvalue().encode("utf-8")


def build_sha256s(
    root: Path, public_paths: Sequence[str], manifest_bytes: bytes
) -> bytes:
    included = sorted(
        set(rel for rel in public_paths if rel not in {MANIFEST_PATH, SHA256_PATH})
        | {MANIFEST_PATH}
    )
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    lines: List[str] = []
    for rel in included:
        digest = manifest_hash if rel == MANIFEST_PATH else sha256_file(root / rel)
        lines.append("{}  {}\n".format(digest, rel))
    return "".join(lines).encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".release-metadata-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o644)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def compare_generated(path: Path, expected: bytes) -> List[str]:
    if not path.is_file():
        return ["generated release file is missing: {}".format(path.name)]
    actual = path.read_bytes()
    if actual != expected:
        return [
            "{} is stale or non-deterministic; run --write after fixing all gates".format(
                path.name
            )
        ]
    return []


def print_issues(issues: Iterable[str]) -> None:
    print("Release validation failed:", file=sys.stderr)
    for issue in issues:
        print("  - " + issue, file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        root = find_repository_root(args.root)
        public_paths = git_public_files(root)
    except ReleaseValidationError as exc:
        print("Release validation failed: {}".format(exc), file=sys.stderr)
        return 2

    public_set = set(public_paths)
    issues: List[str] = []
    issues.extend(generated_paths_are_public(root))
    issues.extend(validate_public_paths(root, public_paths))
    issues.extend(validate_gitignore(root))
    issues.extend(validate_required_files(public_set))
    issues.extend(validate_readme(root))
    issues.extend(validate_script_index(root, public_set))
    issues.extend(validate_large_checksums(root))
    issues.extend(scan_sensitive_content(root, public_paths))
    if issues:
        print_issues(issues)
        return 1

    manifest_bytes = build_manifest(root, public_paths)
    sha256_bytes = build_sha256s(root, public_paths, manifest_bytes)

    if args.write:
        atomic_write(root / MANIFEST_PATH, manifest_bytes)
        atomic_write(root / SHA256_PATH, sha256_bytes)
        print(
            "Release metadata written successfully: {} public files validated; "
            "{} manifest entries; {} checksum entries.".format(
                len(public_paths),
                manifest_bytes.count(b"\n") - 1,
                sha256_bytes.count(b"\n"),
            )
        )
        return 0

    compare_issues: List[str] = []
    compare_issues.extend(compare_generated(root / MANIFEST_PATH, manifest_bytes))
    compare_issues.extend(compare_generated(root / SHA256_PATH, sha256_bytes))
    if compare_issues:
        print_issues(compare_issues)
        return 1
    print(
        "Release check passed: {} public files; {} manifest entries; "
        "{} checksum entries; no files were written.".format(
            len(public_paths),
            manifest_bytes.count(b"\n") - 1,
            sha256_bytes.count(b"\n"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
