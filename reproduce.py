"""Run from this directory after installing requirements.txt. Inputs are bundled."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(arguments):
    log_dir = ROOT / "reports/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(arguments).replace("--", "").replace(".", "_")
    with (log_dir / f"{name}.log").open("w", encoding="utf-8") as log:
        log.write(f"UTC start: {datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"Command: python {' '.join(arguments)}\n")
        log.flush()
        process = subprocess.Popen(
            [sys.executable, *arguments],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        log.write(f"UTC end: {datetime.now(timezone.utc).isoformat()}\nExit code: {code}\n")
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse bundled fitted models; regenerate predictions, timings and report",
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    run(["-m", "sentiment.data"])
    for task in ["three_class", "binary"]:
        for model in ["baseline", "transformer"]:
            if not args.evaluate_only:
                run(
                    [
                        "-u",
                        "-m",
                        "sentiment.experiment",
                        "--task",
                        task,
                        "--model",
                        model,
                        "--epochs",
                        str(args.epochs),
                    ]
                )
            run(["-u", "-m", "sentiment.evaluate", "--task", task, "--model", model])
    run(["-m", "sentiment.report"])
    run(["-m", "unittest", "discover", "-s", "tests", "-v"])


if __name__ == "__main__":
    main()
