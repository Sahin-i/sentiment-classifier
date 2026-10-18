"""Create the split once; every model must load and verify that artifact."""

from __future__ import annotations

import csv
import zipfile
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from .common import (
    ROOT,
    SEED,
    clean_text,
    digest_bytes,
    digest_file,
    digest_ids,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

MODEL_REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"


def rating_label(rating: int) -> str:
    if rating not in range(1, 6):
        raise ValueError(f"Invalid rating: {rating}")
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


def prepare(sample_size: int = 10000) -> dict:
    """Stratified sample with natural prevalence, then 70/15/15 stratification."""
    if (ROOT / "data/splits.json").exists():
        manifest, _ = load_data()
        if manifest["sample_size"] != sample_size:
            raise ValueError(
                "Saved split exists with a different size. Use a new project copy for a new experiment."
            )
        return manifest
    path = ROOT / "data/raw/clothing_reviews.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    groups = defaultdict(list)
    empty = invalid = 0
    for row in raw:
        text = clean_text(row.get("Review Text", ""))
        if not text:
            empty += 1
            continue
        try:
            rating = int(row["Rating"])
            label = rating_label(rating)
        except (ValueError, KeyError):
            invalid += 1
            continue
        text_hash = digest_bytes(text.casefold().encode("utf-8"))
        groups[text_hash].append(
            {
                "id": f"clothing:{int(row[''])}",
                "text": text,
                "rating": rating,
                "label": label,
                "text_sha256": text_hash,
            }
        )
    rows = []
    conflicting_rows = duplicate_rows = 0
    for group in groups.values():
        if len({r["label"] for r in group}) != 1:
            conflicting_rows += len(group)
            continue
        rows.append(group[0])
        duplicate_rows += len(group) - 1
    rows.sort(key=lambda r: int(r["id"].split(":")[1]))
    if sample_size < 1000 or sample_size > len(rows):
        raise ValueError(f"sample_size must be between 1000 and {len(rows)}")
    labels = np.array([r["label"] for r in rows])
    indices = np.arange(len(rows))
    if sample_size < len(rows):
        indices, _ = train_test_split(
            indices, train_size=sample_size, stratify=labels, random_state=SEED
        )
    selected = sorted([rows[int(i)] for i in indices], key=lambda r: int(r["id"].split(":")[1]))
    ids = [r["id"] for r in selected]
    lookup = {r["id"]: r for r in selected}
    train_val, test = train_test_split(
        ids, test_size=0.15, stratify=[lookup[i]["label"] for i in ids], random_state=SEED
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=[lookup[i]["label"] for i in train_val],
        random_state=SEED,
    )
    splits = {"train": sorted(train), "validation": sorted(val), "test": sorted(test)}
    snapshot = ROOT / "data/reviews.jsonl"
    write_jsonl(snapshot, selected)
    manifest = {
        "seed": SEED,
        "sample_size": sample_size,
        "source_rows": len(raw),
        "empty_text_dropped": empty,
        "invalid_rating_dropped": invalid,
        "duplicate_rows_dropped": duplicate_rows,
        "conflicting_duplicate_rows_dropped": conflicting_rows,
        "eligible_unique_rows": len(rows),
        "eligible_counts": dict(Counter(r["label"] for r in rows)),
        "sample_counts": dict(Counter(r["label"] for r in selected)),
        "label_mapping": {
            "1-2": "negative",
            "3": "neutral (rating proxy; includes mixed sentiment)",
            "4-5": "positive",
        },
        "source": "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
        "source_version": 1,
        "download_url": "https://www.kaggle.com/api/v1/datasets/download/nicapotato/womens-ecommerce-clothing-reviews",
        "raw_sha256": digest_file(path),
        "snapshot_sha256": digest_file(snapshot),
        "splits": splits,
        "id_hashes": {key: digest_ids(value) for key, value in splits.items()},
        "split_counts": {
            key: dict(Counter(lookup[i]["label"] for i in value)) for key, value in splits.items()
        },
        "pretrained_model": "prajjwal1/bert-tiny",
        "pretrained_revision": MODEL_REVISION,
        "pretrained_files": {
            p.name: digest_file(p)
            for p in sorted((ROOT / "models/pretrained").glob("*"))
            if p.is_file()
        },
        "notes": "Body text only. Ratings create targets; rating/recommendation/title/age/product metadata never enter either model. No stopword removal or stemming. Exact normalized casefolded text deduplicated before sampling and splitting. Near duplicates and shared products may remain. Validation is used for selection and calibration only. Test is never used for fitting or selection.",
    }
    write_json(ROOT / "data/splits.json", manifest)
    prepare_ood()
    load_data()
    return manifest


def prepare_ood() -> None:
    with zipfile.ZipFile(ROOT / "data/raw/uci_sentiment.zip") as archive:
        name = next(
            n
            for n in archive.namelist()
            if n.endswith("/yelp_labelled.txt") and "__MACOSX" not in n
        )
        lines = archive.read(name).decode("utf-8-sig").splitlines()
    rows = []
    for i, line in enumerate(lines):
        text, label = line.rsplit("\t", 1)
        rows.append(
            {
                "id": f"uci-yelp:{i}",
                "text": clean_text(text),
                "label": "positive" if int(label) else "negative",
            }
        )
    write_jsonl(ROOT / "data/ood_yelp.jsonl", rows)
    write_json(
        ROOT / "data/ood_provenance.json",
        {
            "source": "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
            "citation": "Kotzias, D. (2015). Sentiment Labelled Sentences. UCI. https://doi.org/10.24432/C57604",
            "license": "CC BY 4.0",
            "rows": len(rows),
            "archive_sha256": digest_file(ROOT / "data/raw/uci_sentiment.zip"),
            "snapshot_sha256": digest_file(ROOT / "data/ood_yelp.jsonl"),
            "notes": "Yelp restaurant sentences; no neutral gold class; never used for fitting, calibration or selection. Text cleaned with the same common function; no other adaptation.",
        },
    )


def load_data(binary: bool = False):
    manifest = read_json(ROOT / "data/splits.json")
    snapshot = ROOT / "data/reviews.jsonl"
    if digest_file(snapshot) != manifest["snapshot_sha256"]:
        raise ValueError("Dataset snapshot hash mismatch")
    rows = read_jsonl(snapshot)
    lookup = {r["id"]: r for r in rows}
    if len(lookup) != len(rows) or len({r["text_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate review ID or normalized text")
    for row in rows:
        if (
            digest_bytes(row["text"].casefold().encode("utf-8")) != row["text_sha256"]
            or rating_label(row["rating"]) != row["label"]
        ):
            raise ValueError("Dataset row is inconsistent")
    assigned = []
    for key, ids in manifest["splits"].items():
        if digest_ids(ids) != manifest["id_hashes"][key]:
            raise ValueError(f"Saved {key} split hash mismatch")
        assigned.extend(ids)
    if len(set(assigned)) != len(assigned) or set(assigned) != set(lookup):
        raise ValueError("Splits overlap, omit rows or include unknown IDs")
    partitions = {
        key: [lookup[i] for i in ids if not binary or lookup[i]["label"] != "neutral"]
        for key, ids in manifest["splits"].items()
    }
    return manifest, partitions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10000)
    args = parser.parse_args()
    result = prepare(args.sample_size)
    print(
        {k: result[k] for k in ["sample_size", "sample_counts", "split_counts", "id_hashes"]},
        flush=True,
    )
