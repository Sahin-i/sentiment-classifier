from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re
import unicodedata
from pathlib import Path

ROOT = (
    Path(os.environ.get("REVIEW_SENTIMENT_HOME", Path(__file__).resolve().parents[1]))
    .expanduser()
    .resolve()
)
LABELS = ["negative", "neutral", "positive"]
BINARY_LABELS = ["negative", "positive"]
SEED = 42


def clean_text(text: str) -> str:
    """Minimal common cleaning. Preserve negation, punctuation and word order."""
    if not isinstance(text, str):
        raise ValueError("Review must be a string")
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def digest_ids(ids: list[str]) -> str:
    return digest_bytes("\n".join(ids).encode("utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def environment() -> dict:
    versions = {}
    for name in [
        "torch",
        "transformers",
        "scikit-learn",
        "numpy",
        "scipy",
        "joblib",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
        "threadpoolctl",
        "matplotlib",
        "psutil",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
