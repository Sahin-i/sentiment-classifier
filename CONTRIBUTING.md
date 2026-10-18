# Contributing

Use Python 3.12 and create a virtual environment. For CPU development, install PyTorch from the CPU index first, then install the development dependencies and editable package:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

Before opening a pull request:

```bash
ruff check .
ruff format --check .
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
python -m build
python scripts/check_wheel.py
```

Keep code changes separate from scientific experiment changes. The recorded scores belong to the exact included split and fitted weights. If changing labels, input cleaning, training or membership, run a new experiment and update the report, predictions, error reviews and artifact manifest together. Never report models tested on different IDs as a valid comparison.

`python reproduce.py` overwrites derived experiment files. Back up results before retraining, and review any changed error-analysis examples. Use `python scripts/verify_artifacts.py --write` only after intentionally changing and validating recorded artifacts; `--write` is not a way to make a failed integrity check disappear.

Runtime dependencies are pinned to the recorded environment because serialized scikit-learn models depend on compatible versions. Upgrade pins with testing and, when necessary, retrain the baseline. New large checkpoints should be distributed through GitHub Releases or Git LFS rather than repeatedly committed to ordinary Git history.
