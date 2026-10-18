"""Smoke-test the built wheel in a fresh venv without editable/source imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = list((ROOT / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Build exactly one wheel with python -m build first")
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-check-", dir=work) as temporary:
        directory = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(directory / "venv")
        executable = (
            directory / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["REVIEW_SENTIMENT_HOME"] = str(ROOT)
        # Reuse installed third-party dependencies without reprocessing the
        # parent's editable-install .pth hooks. The application itself must be
        # loaded from this venv's wheel, which is asserted below.
        child_purelib = subprocess.check_output(
            [str(executable), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            cwd=directory,
            env=env,
            text=True,
        ).strip()
        dependency_paths = {Path(sysconfig.get_paths()["purelib"])}
        dependency_paths.update(
            Path(p) for p in sys.path if Path(p).name in {"site-packages", "dist-packages"}
        )
        (Path(child_purelib) / "runtime_dependencies.pth").write_text(
            "\n".join(str(path) for path in sorted(dependency_paths) if path.is_dir()) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [str(executable), "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
            cwd=directory,
            env=env,
            check=True,
        )
        installed = subprocess.check_output(
            [str(executable), "-c", "import sentiment; print(sentiment.__file__)"],
            cwd=directory,
            env=env,
            text=True,
        ).strip()
        if not Path(installed).resolve().is_relative_to((directory / "venv").resolve()):
            raise RuntimeError(
                f"Wheel check imported code outside its own environment: {installed}"
            )
        for model in ["baseline", "transformer"]:
            result = subprocess.run(
                [
                    str(executable),
                    "-m",
                    "sentiment.cli",
                    "--model",
                    model,
                    "--text",
                    "The fit is perfect and the fabric feels wonderful.",
                ],
                cwd=directory,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            prediction = json.loads(result.stdout)
            if prediction["model"] != model or prediction["sentiment"] not in {
                "negative",
                "neutral",
                "positive",
            }:
                raise RuntimeError("Installed wheel returned invalid prediction")
            print(
                json.dumps(
                    {
                        "wheel_model": model,
                        "prediction": prediction["sentiment"],
                        "status": "passed",
                    }
                )
            )


if __name__ == "__main__":
    main()
