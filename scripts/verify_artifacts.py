"""Verify the frozen experiment bytes; explicitly regenerate its inventory after a new run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts.sha256"
ARTIFACT_DIRECTORIES = ("data", "models", "reports")


def artifact_files(root: Path = ROOT) -> list[Path]:
    return sorted(
        path
        for directory in ARTIFACT_DIRECTORIES
        for path in (root / directory).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "logs" not in path.relative_to(root).parts
        and path.name != "validation.json"
    )


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_manifest(root: Path = ROOT, manifest: Path = MANIFEST) -> int:
    files = artifact_files(root)
    if not files:
        raise ValueError("No experiment artifacts found")
    text = (
        "\n".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files) + "\n"
    )
    manifest.write_text(text, encoding="utf-8", newline="\n")
    return len(files)


def verify(root: Path = ROOT, manifest: Path = MANIFEST) -> int:
    expected = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest path leaves project: {name}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or name in expected:
            raise ValueError(f"Invalid or repeated manifest path: {name}")
        expected.add(name)
        if not path.is_file() or sha256(path) != checksum:
            raise ValueError(f"Missing or modified recorded artifact: {name}")
    actual = {path.relative_to(root).as_posix() for path in artifact_files(root)}
    if expected != actual:
        raise ValueError(
            f"Artifact inventory mismatch; added={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )
    if not expected:
        raise ValueError("Artifact inventory is empty")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Intentionally replace the inventory after validating a new experiment",
    )
    args = parser.parse_args()
    try:
        count = write_manifest() if args.write else verify()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"artifacts": count, "status": "written" if args.write else "verified"}))


if __name__ == "__main__":
    main()
