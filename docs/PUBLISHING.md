# Publish this project to GitHub

The project directory is the repository root. Extract the ZIP first if using the downloadable copy, including its hidden `.github` directory and dotfiles. Do not upload the ZIP itself as the repository's source code.

## First push

Create an empty GitHub repository named `product-review-sentiment-classifier`. Leave GitHub's README, license and `.gitignore` initialization options unchecked because those files are already included.

If starting from the extracted ZIP, initialize and commit it:

```bash
git init -b main
git add .
git commit -m "Add reproducible review sentiment comparison"
```

The prepared local checkout already has a `main` branch and initial commit. From either checkout, replace `YOUR_USERNAME` below and add your new repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/product-review-sentiment-classifier.git
git push -u origin main
```

The first push triggers the included GitHub Actions workflow. Local checks have been run; the GitHub-hosted Ubuntu/Windows jobs can run only after the repository is pushed. Review their results in the Actions tab before advertising CI as passing.

Suggested repository description:

> Reproducible three-class sentiment comparison: TF-IDF + Logistic Regression versus fine-tuned BERT on identical real-review splits, with error analysis, calibration and CLI.

Suggested topics: `sentiment-analysis`, `machine-learning`, `scikit-learn`, `transformers`, `bert`, `nlp`, `product-reviews`, `model-evaluation`.

## Data and model policy

The frozen dataset, split IDs, trained models and prediction evidence are intentionally versioned so a fresh clone can reproduce inference and audit the comparison without downloading model/data artifacts. This snapshot is about 73 MiB before Git compression, with each file below 18 MiB. GitHub's documented ordinary-Git limit is 100 MiB per file; [see GitHub's file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

`.gitattributes` preserves the exact bytes of scientific artifacts, including CSV and JSON files. Do not remove those rules: line-ending conversion would invalidate saved SHA-256 fingerprints across platforms. Original public inputs can be restored with `python scripts/download_inputs.py`; missing fitted weights must come from the trusted recorded repository/archive or be retrained as a new run.

Use GitHub Releases or Git LFS for future larger models and repeated experiment snapshots. The local release ZIP and Python build output are ignored by Git. Dataset/model licenses and attribution are in `DATA_SOURCES.md`.

The wheel contains application code only. For wheel-based inference, set `REVIEW_SENTIMENT_HOME` to a trusted full repository checkout so the package can find the data/models/reports. Editable installation from a clone is the recommended setup.
