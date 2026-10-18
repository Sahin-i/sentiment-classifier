# Product Review Sentiment Classifier

A completed, executed comparison of **TF-IDF + Logistic Regression** against an **actually fine-tuned Hugging Face BERT-tiny** model on real, imbalanced, three-class product reviews.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

**Start with [the full report](reports/report.md)**, which renders directly on GitHub. Open `reports/report.html` locally for the illustrated HTML version. The report covers every required feature and all four bonus features, including a separately trained binary ablation. Trained models, inputs and row-level predictions are included; inference does not need a network connection.

## Recorded main results

| Model | Test accuracy | Test macro F1 | Warm CPU p50 / review |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.7953 | 0.6087 | 0.559 ms |
| Fine-tuned BERT-tiny | 0.7707 | 0.5879 | 2.007 ms |

Both use **exactly the same 7,000 train / 1,500 validation / 1,500 test reviews**. All 10,000 reviews are real examples sampled with natural class prevalence from the 23,486-row clothing-review dataset. Ratings map to negative (1–2), neutral (3), positive (4–5).

![Per-class F1 comparison](reports/charts/per_class_f1.png)

The baseline is recommended for a stated CPU-only weekly apparel-reporting use case, based on validation macro F1 and measured operating cost. The paired test macro-F1 difference has a 95% interval crossing zero; do not claim a definitive population-level advantage. BERT performs better on the separate Yelp test. Minority-class F1 is modest for both, so the recommendation is for aggregate reporting with manual audits. See the full report for all per-class metrics and limitations.

## Install and classify

Python **3.12** is recommended. From this project directory on Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\review-sentiment.exe --text "Nice color, but the fit is just okay."
.venv\Scripts\python.exe -m sentiment.cli --text "Nice color, but the fit is just okay."
.venv\Scripts\python.exe -m sentiment.cli --model transformer --file examples/reviews.txt
```

On Linux/macOS, use `python3.12 -m venv .venv`, then `.venv/bin/python` in place of `.venv\Scripts\python.exe`, and `.venv/bin/review-sentiment` for the installed command. Installing PyTorch from the CPU index first avoids downloading an unnecessary CUDA runtime.

Run these commands from the full repository checkout. Editable installation keeps the package connected to its bundled model/data/report files. A standalone wheel contains code only: set `REVIEW_SENTIMENT_HOME` to a trusted checkout when using it outside this directory.

The CLI outputs one JSON object per review with `sentiment`, calibrated `confidence`, all three class `probabilities`, selected `model` and a transformer `truncated` flag. The default `--model best` reads the saved validation-based deployment choice. Both models remain available with `--model baseline` or `--model transformer`. Input may come from `--text`, a UTF-8 `--file` containing one review per line, or `--stdin`. Blank reviews are rejected. Use English product-review text; support-ticket urgency and other languages are outside the evaluated scope.

These confidence values are empirical estimates, not guarantees. Temperature calibration is fit on validation data; raw and calibrated reliability are both measured on held-out data.

## Reproduce

```powershell
.venv\Scripts\python.exe reproduce.py
```

This verifies the saved data manifest, fits both three-class models and both binary ablation models, evaluates all four, runs the Yelp transfer check and calibration, measures CPU latency, regenerates the illustrated report, and runs the test suite. Training took about 3.4 minutes for the three-class BERT model on the execution laptop; allow additional time for the binary model, evaluation, imports and report generation. Dependency installation time depends on the machine and download speed.

```powershell
# Reuse the bundled fitted models and refresh evaluation/reporting only:
.venv\Scripts\python.exe reproduce.py --evaluate-only

# Run the integrity and behavioral checks:
.venv\Scripts\python.exe -m unittest discover -s tests -v

# Verify/restore original public inputs if they were removed:
.venv\Scripts\python.exe scripts/download_inputs.py
```

No GPU, Hugging Face login, paid service or fresh dataset download is needed once dependencies are installed. The original pretrained weights, source CSV, selected snapshot, saved split, fitted models and predictions are bundled. Package versions are pinned in `requirements.txt`; each model's metadata also records the actual execution environment. Different library/BLAS/OS versions can cause small floating-point changes even with deterministic seeds, while split membership remains identical.

Full reproduction intentionally replaces derived model/evaluation files. Keep a copy of the original submitted results if experimenting. If retraining changes which examples are misclassified, the written reviews in `reports/error_reviews.json` must be reviewed again; report generation deliberately refuses to present a correctly classified row as an error.

## Comparison safeguards

- A single `data/splits.json` stores original row IDs, split membership, seed, cleaning counts, class counts and SHA-256 fingerprints.
- Both training functions consume the same verified partitions. Evaluation rejects changed snapshots, split hashes or model training fingerprints. Report generation additionally checks test ID order and gold labels.
- Exact normalized text duplicates are removed before sampling/splitting. All features come only from review bodies; ratings and other metadata never enter either model.
- TF-IDF vocabulary/IDF and inverse-frequency class weights are learned only from training data. Validation selects Logistic Regression C and the BERT checkpoint, and fits scalar temperatures. Test labels never participate in those steps.
- Binary experiments filter neutral rows from those same partitions; they never draw a new split. The report checks that the polar test IDs match exactly.
- Each model's CPU timing uses the same 128 test bodies, same thread limit and warmup/repetition protocol, in separate processes.

## Project contents

| Path | Contents |
| --- | --- |
| `reports/report.html` / `report.md` | Complete assessment-ready analysis, figures and requirement evidence |
| `reports/per_class_metrics.csv` | Precision, recall, F1 and support for every class and model |
| `reports/three_class_*.json`, `binary_*.json` | Full metrics, confusion matrices, calibration and resource measurements |
| `reports/predictions/` | Ordered validation/test/OOD IDs, gold labels, logits and probabilities |
| `reports/error_reviews.json` | Ten real reviewed mistakes, five per model, with specific hypotheses |
| `reports/comparison.json` | Paired uncertainty calculation and binary ablation |
| `reports/deployment.json` | Validation-based model-selection rule and selected CLI default |
| `data/raw/` | Original Kaggle CSV and UCI source archive |
| `data/reviews.jsonl`, `data/splits.json` | Frozen 10,000-review snapshot and shared split |
| `data/ood_yelp.jsonl` | 1,000 separate real restaurant-review sentences |
| `models/pretrained/` | Pinned original BERT-tiny checkpoint and vocabulary |
| `models/three_class/` | Both trained three-class models and calibration |
| `models/binary/` | Both separately trained binary ablation models |
| `sentiment/` | Data preparation, training, inference, evaluation and report-generation code |
| `tests/` | Data leakage, split tampering, metric recomputation, calibration and live inference checks |
| `reproduce.py` | Sequential, logged end-to-end execution |
| `DATA_SOURCES.md` | Sources, exact revisions, licenses and attribution |

The project code is MIT-licensed. Third-party datasets and model weights retain their listed licenses.

Recorded validation: **19 tests passed**. See `reports/logs/final_tests.log`. This includes rejecting tampered snapshots/splits, recomputing all per-class metrics from predictions, verifying binary membership, checking calibration behavior and comparing live model outputs with the saved evaluation logits.

## Development and CI

Install `requirements-dev.txt` in the same environment. The included GitHub Actions workflow checks code style, frozen artifact checksums, all experiment/inference tests, the installed CLI and an independently installed wheel on Ubuntu and Windows with Python 3.12. The workflow uses read-only repository permissions and pinned action revisions. Hosted CI results are available only after pushing the repository; no passing hosted run is claimed here.

```bash
ruff check .
ruff format --check .
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
python -m build
python scripts/check_wheel.py
```

The approximately 73 MiB frozen experiment is deliberately tracked for reproducible grading and offline inference. `.gitattributes` preserves artifact bytes so Windows/Linux line-ending conversion cannot break the hashes. Future larger model versions should use GitHub Releases or Git LFS. The archive/build output, virtual environments, caches, local working files and credentials are ignored.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development rules, [SECURITY.md](SECURITY.md) for security reporting, and [publishing instructions](docs/PUBLISHING.md) for the first GitHub push.
