## Change

Describe the concrete behavior being changed and why.

## Validation

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `python scripts/verify_artifacts.py`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python -m build` and `python scripts/check_wheel.py`

## Experiment impact

State whether training data, split membership, predictions or model artifacts changed. For a new experiment, include its configuration and regenerated results; do not combine metrics from different splits or silently replace the recorded benchmark.
