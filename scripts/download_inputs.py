"""Optional input restoration. Bundled artifacts normally work fully offline."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))
ood = json.loads((ROOT / "data/ood_provenance.json").read_text(encoding="utf-8"))


def restore(path, url, expected, zipped_csv=False):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        print("Verified", path.relative_to(ROOT))
        return
    with urllib.request.urlopen(url, timeout=180) as response:
        content = response.read()
    if zipped_csv:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            content = archive.read(next(n for n in archive.namelist() if n.endswith(".csv")))
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(
            f"Upstream file changed: {url}. Keep the bundled snapshot for this experiment."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print("Restored", path.relative_to(ROOT))


restore(
    ROOT / "data/raw/clothing_reviews.csv",
    manifest["download_url"],
    manifest["raw_sha256"],
    zipped_csv=True,
)
restore(
    ROOT / "data/raw/uci_sentiment.zip",
    "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
    ood["archive_sha256"],
)
for name, expected in manifest["pretrained_files"].items():
    restore(
        ROOT / "models/pretrained" / name,
        f"https://huggingface.co/{manifest['pretrained_model']}/resolve/{manifest['pretrained_revision']}/{name}",
        expected,
    )
