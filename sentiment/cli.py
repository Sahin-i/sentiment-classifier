"""JSON sentiment predictions from bundled local models; no network required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import predict


def main():
    parser = argparse.ArgumentParser(
        description="Classify English product reviews as negative, neutral or positive."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One review")
    source.add_argument("--file", type=Path, help="UTF-8 file: one review per line")
    source.add_argument("--stdin", action="store_true", help="Read one review per line from stdin")
    parser.add_argument("--model", choices=["best", "baseline", "transformer"], default="best")
    args = parser.parse_args()
    try:
        if args.text is not None:
            texts = [args.text]
        elif args.file is not None:
            texts = args.file.read_text(encoding="utf-8-sig").splitlines()
        else:
            texts = sys.stdin.read().splitlines()
        for row in predict(texts, args.model):
            print(json.dumps(row, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
