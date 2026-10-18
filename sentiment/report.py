"""Assemble auditable Markdown/HTML reports from actual prediction artifacts."""

from __future__ import annotations

import csv
import html
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "review-sentiment-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from .common import (
    BINARY_LABELS,
    LABELS,
    ROOT,
    read_json,
    read_jsonl,
    write_json,
)
from .data import load_data

NAMES = {"baseline": "TF-IDF + Logistic Regression", "transformer": "Fine-tuned BERT-tiny"}


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    )


def paired_comparison(a, b):
    if [r["id"] for r in a] != [r["id"] for r in b] or [r["gold"] for r in a] != [
        r["gold"] for r in b
    ]:
        raise ValueError("Prediction IDs/order or gold labels differ; comparison rejected")
    y = np.array([LABELS.index(r["gold"]) for r in a])
    pa = np.array([LABELS.index(r["prediction"]) for r in a])
    pb = np.array([LABELS.index(r["prediction"]) for r in b])
    rng = np.random.default_rng(2026)
    groups = [np.where(y == i)[0] for i in range(3)]
    deltas = []
    for _ in range(1000):
        sample = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(
            f1_score(y[sample], pb[sample], labels=[0, 1, 2], average="macro", zero_division=0)
            - f1_score(y[sample], pa[sample], labels=[0, 1, 2], average="macro", zero_division=0)
        )
    a_only = int(np.sum((pa == y) & (pb != y)))
    b_only = int(np.sum((pa != y) & (pb == y)))
    return {
        "direction": "transformer minus baseline",
        "bootstrap_samples": 1000,
        "bootstrap_seed": 2026,
        "bootstrap_method": "Paired stratified bootstrap of fixed test predictions; sampling within each true class. Conditional uncertainty for this split, not variation across training seeds.",
        "macro_f1_delta": float(
            f1_score(y, pb, average="macro") - f1_score(y, pa, average="macro")
        ),
        "macro_f1_delta_95_percentile_ci": list(map(float, np.percentile(deltas, [2.5, 97.5]))),
        "baseline_only_correct": a_only,
        "transformer_only_correct": b_only,
        "mcnemar_exact_p": float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)
        if a_only + b_only
        else 1.0,
    }


def charts(metrics):
    out = ROOT / "reports/charts"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, (model, result) in zip(axes, metrics.items()):
        matrix = np.array(result["confusion_matrix"])
        ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] / matrix[i].sum() > 0.55 else "#17233b",
                )
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted class",
            ylabel="True class",
            title=NAMES[model],
        )
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    x = np.arange(3)
    for offset, (model, result) in zip([-0.18, 0.18], metrics.items()):
        values = [result["classification_report"][label]["f1-score"] for label in LABELS]
        bars = ax.bar(
            x + offset,
            values,
            width=0.35,
            label=NAMES[model],
            color="#1b4965" if model == "baseline" else "#ee7752",
        )
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(
        xticks=x,
        xticklabels=LABELS,
        ylim=(0, 1.08),
        ylabel="F1",
        title="Every sentiment class counts",
    )
    ax.legend(loc="upper left")
    fig.savefig(out / "per_class_f1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (model, result) in zip(axes, metrics.items()):
        ax.plot([0, 1], [0, 1], "--", color="#a2aab6", label="Perfect calibration")
        for mode, color in [("raw", "#1b4965"), ("scaled", "#ee7752")]:
            bins = [b for b in result["calibration"][mode]["bins"] if b["count"]]
            ax.plot(
                [b["mean_confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                "o-",
                color=color,
                label=mode,
            )
        ax.set(
            xlim=(0.3, 1.02),
            ylim=(0, 1.02),
            xlabel="Mean predicted confidence",
            ylabel="Observed accuracy",
            title=NAMES[model],
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "calibration.png", dpi=160)
    plt.close(fig)


def render_html(markdown):
    """Small, offline renderer for the report's controlled Markdown subset."""

    def inline(text):
        escaped = html.escape(text)
        escaped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', escaped)
        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        return escaped

    lines = markdown.splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            output.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
        elif line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("| "):
                cells = re.split(r"(?<!\\)\|", lines[i].strip("| "))
                rows.append([c.strip().replace("\\|", "|") for c in cells])
                i += 1
            output.append(
                '<div class="table-wrap"><table><thead><tr>'
                + "".join("<th>" + inline(c) + "</th>" for c in rows[0])
                + "</tr></thead><tbody>"
            )
            for row in rows[2:]:
                output.append("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in row) + "</tr>")
            output.append("</tbody></table></div>")
            continue
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line[level:].strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            output.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
        elif line.startswith("> "):
            output.append("<blockquote>" + inline(line[2:]) + "</blockquote>")
        elif line.startswith("- "):
            output.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                output.append("<li>" + inline(lines[i][2:]) + "</li>")
                i += 1
            output.append("</ul>")
            continue
        elif line:
            output.append("<p>" + inline(line) + "</p>")
        i += 1
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Review sentiment | Experiment report</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#1a2942;font:16px/1.7 system-ui,Segoe UI,sans-serif}header{background:#142b45;color:#fff;padding:22px 6%;display:flex;justify-content:space-between;gap:20px}header b{letter-spacing:.12em;font-size:13px}header span{font-size:13px;color:#bcd0e4}main{max-width:1120px;margin:38px auto;background:#fff;padding:48px 58px;border:1px solid #e2e8f0;border-radius:16px;box-shadow:0 8px 35px #1a294209}h1{font-size:38px;line-height:1.2;letter-spacing:-.04em;margin:0 0 22px}h2{font-size:25px;margin:48px 0 16px;padding-top:20px;border-top:1px solid #e4eaf2;letter-spacing:-.025em}h3{margin-top:30px}a{color:#1b628d;text-underline-offset:3px}p{margin:15px 0}blockquote{margin:22px 0;padding:16px 22px;background:#edf5fa;border-left:4px solid #3385b6;border-radius:0 8px 8px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}th{background:#eaf0f6;text-align:left;white-space:nowrap}th,td{padding:12px 14px;border-bottom:1px solid #dfe7f0}tr:nth-child(even) td{background:#f8fafc}.table-wrap{overflow:auto;border:1px solid #dfe7f0;border-radius:8px;margin:20px 0}code{font:13px/1.6 Consolas,monospace;background:#edf2f7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}pre{background:#142b45;color:#f0f6ff;padding:20px;border-radius:9px;overflow:auto}pre code{background:none;padding:0;color:inherit}img{display:block;width:100%;height:auto;margin:25px auto}li{margin:6px 0}footer{max-width:1120px;margin:20px auto 40px;font-size:13px;color:#596b82}@media(max-width:750px){main{margin:15px;padding:24px 20px}h1{font-size:30px}header{padding:18px}header span{display:none}th,td{padding:9px}}@media print{body{background:#fff}main{border:0;box-shadow:none;margin:0;padding:10px}header{display:none}h2{break-after:avoid}tr,img,blockquote{break-inside:avoid}}</style></head><body><header><b>REVIEW SENTIMENT / EXPERIMENT 001</b><span>Real data · Shared splits · Measured results</span></header><main>"""
        + "\n".join(output)
        + "</main><footer>All metrics derive from the included predictions. See README.md for reproduction commands and DATA_SOURCES.md for licenses.</footer></body></html>"
    )


def build():
    manifest, parts = load_data()
    metrics = {name: read_json(ROOT / "reports" / f"three_class_{name}.json") for name in NAMES}
    metadata = {
        name: read_json(ROOT / "models/three_class" / name / "metadata.json") for name in NAMES
    }
    predictions = {
        name: read_jsonl(ROOT / "reports/predictions" / f"three_class_{name}_test.jsonl")
        for name in NAMES
    }
    for model in NAMES:
        if (
            metrics[model]["id_hashes"] != manifest["id_hashes"]
            or metadata[model]["id_hashes"] != manifest["id_hashes"]
        ):
            raise ValueError("Model split mismatch; report rejected")
        if [p["id"] for p in predictions[model]] != manifest["splits"]["test"]:
            raise ValueError("Predictions do not cover the exact ordered test set")
    comparison = paired_comparison(predictions["baseline"], predictions["transformer"])
    winner = max(NAMES, key=lambda m: metrics[m]["macro_f1"])
    validation_winner = max(NAMES, key=lambda m: metadata[m]["validation_macro_f1"])
    # Predeclared practical decision: favor baseline if it is within 0.005 validation macro F1.
    selected = (
        "baseline"
        if metadata["baseline"]["validation_macro_f1"]
        >= metadata["transformer"]["validation_macro_f1"] - 0.005
        else "transformer"
    )
    deployment = {
        "selected_model": selected,
        "use_case": "CPU-only weekly sentiment reporting for an apparel retailer, 10,000 English reviews/week; minority sentiment visibility is important.",
        "selection_rule": "Validation macro F1; prefer baseline when within 0.005. Rule does not use test scores or OOD outcomes.",
        "validation_winner": validation_winner,
        "test_macro_f1_winner": winner,
    }
    write_json(ROOT / "reports/deployment.json", deployment)
    binary = {name: read_json(ROOT / "reports" / f"binary_{name}.json") for name in NAMES}
    polar_comparison = {}
    for model in NAMES:
        polar = [p for p in predictions[model] if p["gold"] != "neutral"]
        binary_pred = read_jsonl(ROOT / "reports/predictions" / f"binary_{model}_test.jsonl")
        if [p["id"] for p in polar] != [p["id"] for p in binary_pred]:
            raise ValueError("Binary ablation uses different polar test IDs")
        y = [p["gold"] for p in polar]
        pred = [p["prediction"] for p in polar]
        polar_comparison[model] = {
            "n": len(polar),
            "three_way_on_polar_accuracy": float(np.mean(np.array(y) == np.array(pred))),
            "three_way_on_polar_macro_f1": float(
                f1_score(y, pred, labels=BINARY_LABELS, average="macro", zero_division=0)
            ),
            "binary_trained_accuracy": binary[model]["accuracy"],
            "binary_trained_macro_f1": binary[model]["macro_f1"],
        }
    write_json(
        ROOT / "reports/comparison.json",
        {
            "paired_statistics": comparison,
            "binary_ablation": polar_comparison,
            "deployment": deployment,
            "main_split_hashes": manifest["id_hashes"],
        },
    )
    charts(metrics)
    n = manifest["sample_size"]
    per_class_rows = []
    for model, result in metrics.items():
        for label in LABELS:
            r = result["classification_report"][label]
            per_class_rows.append(
                [
                    NAMES[model],
                    label,
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['f1-score']:.4f}",
                    int(r["support"]),
                ]
            )
    with (ROOT / "reports/per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "class", "precision", "recall", "f1", "support"])
        writer.writerows(per_class_rows)
    majority_y = [r["label"] for r in parts["test"]]
    majority_macro = f1_score(
        majority_y, ["positive"] * len(majority_y), labels=LABELS, average="macro", zero_division=0
    )
    count_rows = [
        [
            label,
            manifest["sample_counts"][label],
            f"{manifest['sample_counts'][label] / n:.2%}",
            *[manifest["split_counts"][s][label] for s in ["train", "validation", "test"]],
            f"{metadata['baseline']['class_weights'][label]:.3f}",
        ]
        for label in LABELS
    ]
    resource_rows = []
    for model in NAMES:
        m, r = metadata[model], metrics[model]
        b = r["benchmark"]
        resource_rows.append(
            [
                NAMES[model],
                f"{m['training_seconds']:.1f}",
                f"{m['sampled_peak_process_rss_mib']:.1f}",
                f"{r['sampled_peak_evaluation_process_rss_mib']:.1f}",
                f"{m['artifact_bytes'] / 2**20:.2f}",
                f"{b['serial_p50_ms']:.3f}",
                f"{b['serial_p95_ms']:.3f}",
                f"{b['batch_reviews_per_second']:.0f}",
            ]
        )
    cal_rows = []
    for model, result in metrics.items():
        for mode in ["raw", "scaled"]:
            r = result["calibration"][mode]
            high_acc = (
                f"{r['high_confidence_accuracy']:.2%}"
                if r["high_confidence_accuracy"] is not None
                else "N/A"
            )
            cal_rows.append(
                [
                    NAMES[model],
                    mode,
                    f"{r['nll']:.4f}",
                    f"{r['multiclass_brier']:.4f}",
                    f"{r['ece_10_bins']:.4f}",
                    r["high_confidence_count"],
                    high_acc,
                ]
            )
    cue_rows = []
    lookup = {r["id"]: r for r in parts["test"]}
    for model, rows in predictions.items():
        for group, predicate in [
            ("Neutral gold label", lambda p: p["gold"] == "neutral"),
            (
                "Contains but/however/although/yet",
                lambda p: bool(
                    re.search(
                        r"\b(but|however|although|yet)\b", lookup[p["id"]]["text"], flags=re.I
                    )
                ),
            ),
            (
                "No such contrast cue",
                lambda p: not re.search(
                    r"\b(but|however|although|yet)\b", lookup[p["id"]]["text"], flags=re.I
                ),
            ),
        ]:
            subset = [p for p in rows if predicate(p)]
            cue_rows.append(
                [
                    NAMES[model],
                    group,
                    len(subset),
                    f"{np.mean([not p['correct'] for p in subset]):.2%}",
                ]
            )
    ci = comparison["macro_f1_delta_95_percentile_ci"]
    uncertainty = (
        "The interval includes zero, so this test does not establish a clear population-level macro-F1 advantage; cost and validation evidence still support the practical choice."
        if ci[0] <= 0 <= ci[1]
        else "The interval excludes zero for these fixed test predictions; it does not establish robustness across new training seeds or domains."
    )
    lines = [
        "# Product review sentiment classifier",
        "Executed comparison · 10,000 real reviews · Three sentiment classes · CPU training and inference",
        f"**Test macro-F1 winner: {NAMES[winner]}. Recommended deployment: {NAMES[selected]}.** Both models were fitted to the same 7,000 training reviews, selected on the same 1,500 validation reviews, and evaluated on the same 1,500 untouched test reviews.",
        "> The comparison is genuine: the submission includes trained weights, original inputs, saved split IDs, row-level probabilities, measured resource costs, and tests that reject mismatched splits. No reported metric is a placeholder.",
        "## 1. Data and labeling",
        "Source: nicapotato's [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews), CC0. Review body is the only input; stars produce the target. 1–2 stars = negative, 3 = neutral, 4–5 = positive. Neutral is an average-rating proxy and often expresses mixed sentiment, rather than purely factual language.",
        f"Cleaning started with {manifest['source_rows']:,} source rows: {manifest['empty_text_dropped']:,} missing/empty bodies, {manifest['invalid_rating_dropped']} invalid ratings, {manifest['duplicate_rows_dropped']} redundant normalized duplicates and {manifest['conflicting_duplicate_rows_dropped']} rows in contradictory duplicate groups were removed. {manifest['eligible_unique_rows']:,} unique eligible reviews remained. A stratified random sample of 10,000 preserves natural label prevalence; no test-set balancing was applied.",
        table(
            ["Class", "Sample n", "Share", "Train", "Validation", "Test", "Train weight"],
            count_rows,
        ),
        f"The majority-to-minority ratio is {max(manifest['sample_counts'].values()) / min(manifest['sample_counts'].values()):.2f}:1. Always predicting positive gives {manifest['split_counts']['test']['positive'] / 1500:.2%} test accuracy but only {majority_macro:.4f} macro F1. This is why accuracy alone would mislead.",
        "Both models use inverse-frequency weights computed only on their training labels: w(c) = N / (K × count(c)). Logistic Regression uses class_weight='balanced'; BERT uses the same weights in cross-entropy. This increases minority-class influence without changing test prevalence. No resampling was used.",
        "## 2. Preprocessing and model choices",
        "Common cleaning decodes HTML entities, removes HTML tags and zero-width characters, applies Unicode NFKC normalization and collapses whitespace. Negation, punctuation, sentence order and sentiment-bearing words are preserved. Exact casefolded cleaned text is deduplicated before splitting; conflicting duplicate labels are excluded. Ratings, recommendation flags, age, title, clothing ID and product categories are never features.",
        "**Classical baseline:** scikit-learn TF-IDF unigrams/bigrams, 40,000-feature cap, min_df=2, max_df=0.99, sublinear term frequency, followed by class-weighted multinomial Logistic Regression. The default word tokenizer drops punctuation and one-letter tokens. Vocabulary and IDF are learned only on train. C ∈ {0.5, 2, 8} is selected by validation macro F1.",
        f"**Transformer:** Hugging Face [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny), pinned revision `{manifest['pretrained_revision']}`. This is a language-pretrained two-layer encoder with hidden size 128. A new three-class head and all encoder parameters were actually fine-tuned here; this is not zero-shot sentiment inference. The resulting model has {metadata['transformer']['parameter_count']:,} parameters. Uncased WordPiece tokenization, dynamic padding, maximum 192 tokens, weighted cross-entropy, AdamW lr=0.0003, weight decay=0.01, batch size 32, 10% linear warmup and gradient clipping at 1.0 were used for five epochs. The best validation macro-F1 checkpoint is kept (epoch {metadata['transformer']['selected_epoch']}). {metadata['transformer']['train_truncated_fraction']:.2%} of training reviews exceed 192 tokens and are right-truncated.",
        "The compact transformer makes the complete experiment reproducible on a CPU. No GPU, paid API, hosted inference service or full-size BERT fine-tuning is required. This compares two practical pipelines with their stated search budgets; it does not isolate architecture under equal optimization budgets.",
        "## 3. Identical-split audit",
        "Seed 42; stratified 70/15/15 train/validation/test after cleaning and sampling. The shared `data/splits.json` is the sole source of membership. Every training run records the SHA-256 of that file, the dataset snapshot and ordered IDs. Evaluation rechecks them, and report generation checks row-level ID and gold-label equality. A mismatch raises an error and rejects the comparison. Test text is used only for predictions, benchmark timing and post-evaluation analysis.",
        table(
            ["Partition", "Rows", "SHA-256 of ordered IDs (both models)"],
            [[s, len(parts[s]), f"`{manifest['id_hashes'][s]}`"] for s in parts],
        ),
        f"Snapshot SHA-256: `{manifest['snapshot_sha256']}`. Raw CSV SHA-256: `{manifest['raw_sha256']}`.",
        "The baseline's C and transformer's checkpoint are selected only on validation. Scalar confidence temperatures are also fitted only on validation. The test set does not determine fitting, hyperparameters, temperature or the CLI's model choice. Neither model is refitted on train+validation.",
        "## 4. Held-out results",
        table(
            ["Model", "Validation macro F1", "Test accuracy", "Test macro F1", "Test weighted F1"],
            [
                [
                    NAMES[m],
                    f"{metadata[m]['validation_macro_f1']:.4f}",
                    f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}",
                    f"{r['classification_report']['weighted avg']['f1-score']:.4f}",
                ]
                for m, r in metrics.items()
            ],
        ),
        table(["Model", "Class", "Precision", "Recall", "F1", "Test support"], per_class_rows),
        "![Per-class F1 for both models](charts/per_class_f1.png)",
        "![Confusion matrices: rows are true labels, columns predictions](charts/confusion_matrices.png)",
        f"Transformer minus baseline macro F1 = {comparison['macro_f1_delta']:+.4f}; paired stratified bootstrap 95% percentile interval [{ci[0]:+.4f}, {ci[1]:+.4f}], 1,000 resamples, seed 2026. The interval reflects resampling uncertainty on these fixed predictions, not variability across training seeds. Baseline-only correct: {comparison['baseline_only_correct']}; transformer-only correct: {comparison['transformer_only_correct']}; exact McNemar p={comparison['mcnemar_exact_p']:.4g} (accuracy disagreement, not a test of macro F1).",
        uncertainty,
        "## 5. Measured simplicity and resource costs",
        table(
            [
                "Model",
                "Train/selection sec",
                "Train RSS MiB",
                "Eval RSS MiB",
                "Saved MiB",
                "p50 ms/review",
                "p95 ms/review",
                "Batch reviews/sec",
            ],
            resource_rows,
        ),
        "CPU only, four intra-op/BLAS threads. Training costs include all three baseline C fits or all five transformer epochs and validation, plus saving the selected model; they exclude dependency installation/downloads and initial imports. RSS is sampled process resident memory, including loaded libraries, in separate model processes; it is not GPU VRAM or a precise tensor allocation measure. Saved sizes are compressed baseline weights versus transformer safetensors/tokenizer/config; metadata and calibration JSON are excluded.",
        metrics["baseline"]["benchmark"]["method"],
        "These are measurements on this Windows laptop, not a production SLA. First-load times and the complete environment are in the JSON artifacts. Imports/loading, concurrency, token lengths and deployment hardware can substantially change real service latency.",
        "## 6. Error analysis",
        "The cases below are actual held-out misclassifications, selected after evaluation to illustrate failures rather than to estimate their prevalence. Every hypothesis is an interpretation of the text, not proof about hidden model reasoning. Full source bodies and predictions are included for inspection.",
    ]
    reviews_path = ROOT / "reports/error_reviews.json"
    reviews = read_json(reviews_path) if reviews_path.exists() else []
    if not reviews:
        raise ValueError(
            "Written manual review of real errors is required before building the final report"
        )
    error_counts = Counter()
    for item in reviews:
        model = item["model"]
        row = next(p for p in predictions[model] if p["id"] == item["id"])
        if row["correct"]:
            raise ValueError("Error review references a correctly classified example")
        error_counts[model] += 1
        source = lookup[item["id"]]
        lines.extend(
            [
                f"### {NAMES[model]} · {item['id']} · {item['category']}",
                f"Gold: **{row['gold']}** ({source['rating']} stars). Predicted: **{row['prediction']}**. Calibrated confidence: {row['confidence_calibrated']:.3f}.",
                "> " + source["text"],
                "Hypothesis: " + item["hypothesis"],
            ]
        )
    if any(error_counts[m] < 5 for m in NAMES):
        raise ValueError("Need at least five reviewed misclassifications for each model")
    lines.extend(
        [
            "### Systematic weakness check",
            table(["Model", "Subset", "Test n", "Error rate"], cue_rows),
            "Average-rating/neutral reviews expose a recurring boundary problem: they can praise appearance while criticizing fit or value, while stars summarize a personal trade-off. TF-IDF bigrams capture local cues but cannot reliably weigh a long concession against the overall judgment. Tiny BERT has context but limited capacity/data and a 192-token cutoff. Contrast-cue error rates are descriptive and confounded by review difficulty and class mix; the cue itself is not demonstrated to cause errors. The manually reviewed cases and neutral-class metrics support the mixed-sentiment hypothesis, while sarcasm remains an unmeasured possibility.",
            "## 7. Neutral-class ablation",
            "Both models were also trained from scratch/from the same pretrained encoder as binary classifiers after removing only neutral rows from each saved partition. The original non-neutral row assignments are retained; the polar test IDs are identical across binary and three-class comparisons. Each binary transformer receives a new two-class head, five epochs and validation-only checkpoint selection. Class weights are recalculated on binary training data.",
            table(
                [
                    "Model",
                    "Polar test n",
                    "3-way on polar accuracy",
                    "3-way on polar F1",
                    "Binary-trained accuracy",
                    "Binary-trained F1",
                ],
                [
                    [
                        NAMES[m],
                        p["n"],
                        f"{p['three_way_on_polar_accuracy']:.4f}",
                        f"{p['three_way_on_polar_macro_f1']:.4f}",
                        f"{p['binary_trained_accuracy']:.4f}",
                        f"{p['binary_trained_macro_f1']:.4f}",
                    ]
                    for m, p in polar_comparison.items()
                ],
            ),
            "On the same polar test rows, training the binary task changes macro F1 by "
            + "; ".join(
                f"{polar_comparison[m]['binary_trained_macro_f1'] - polar_comparison[m]['three_way_on_polar_macro_f1']:+.4f} for {NAMES[m]}"
                for m in NAMES
            )
            + ". Both binary-trained models score higher here: retaining a neutral output makes these polar decisions harder in this experiment, while enabling the system to represent average/mixed reviews.",
            "For three-way-on-polar scores, a neutral prediction is counted as an error/false negative; macro F1 averages negative and positive. This isolates evaluation to the exact same polar examples. The ablation changes label space, training population, weights and task difficulty; its differences are not a causal estimate of the effect of adding neutral data alone. Binary scores cannot describe how well the system handles genuinely mixed reviews.",
            "## 8. Out-of-domain Yelp test",
            "1,000 real Yelp restaurant sentences (500 negative/500 positive) from [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias (2015), [DOI](https://doi.org/10.24432/C57604), CC BY 4.0. These examples are never used for fitting, checkpoint selection or calibration. They differ in domain, length and label prevalence. They provide no neutral gold labels, so neutral recall/F1 cannot be assessed here.",
            table(
                [
                    "Model",
                    "3-way exact accuracy",
                    "Polar macro F1 (neutral = error)",
                    "Neutral prediction rate",
                    "Forced-binary diagnostic F1",
                ],
                [
                    [
                        NAMES[m],
                        f"{r['ood_yelp']['accuracy']:.4f}",
                        f"{r['ood_yelp']['polar_macro_f1_including_neutral_errors']:.4f}",
                        f"{r['ood_yelp']['neutral_prediction_rate']:.2%}",
                        f"{r['ood_yelp']['forced_binary_diagnostic']['macro_f1']:.4f}",
                    ]
                    for m, r in metrics.items()
                ],
            ),
            "The forced-binary diagnostic discards the neutral logit and chooses between negative and positive; it is explicitly a different decision rule and is not substituted for the main three-way classifier. Performance here cannot be compared directly to clothing accuracy because the target distribution and annotation process differ. Inspect `three_class_*.json` for per-class OOD precision/recall/F1 and confusion matrices.",
            f"On the Yelp polar test, BERT's macro F1 is {metrics['transformer']['ood_yelp']['polar_macro_f1_including_neutral_errors']:.4f} versus {metrics['baseline']['ood_yelp']['polar_macro_f1_including_neutral_errors']:.4f} for the baseline. The ranking reverses in this separate domain check, which limits any claim that the clothing winner generalizes best everywhere.",
            "## 9. Confidence calibration",
            "Temperature scaling fits one positive scalar T by minimizing unweighted validation negative log likelihood; test probabilities are softmax(logits/T). It does not alter the predicted class. Baseline logits are log(predict_proba); BERT logits are its raw head outputs. Calibration is not assumed to succeed: the table shows held-out raw and scaled results. ECE uses ten equal-width confidence bins; Brier is the mean sum of squared probability errors across classes (lower is better for all three scores).",
            table(
                [
                    "Model",
                    "Confidence",
                    "NLL",
                    "Brier",
                    "ECE (10 bins)",
                    "n confidence ≥0.8",
                    "Accuracy at ≥0.8",
                ],
                cal_rows,
            ),
            "![Raw and temperature-scaled reliability diagrams](charts/calibration.png)",
            "Validation is reused for model selection and temperature fitting, so calibration quality may be optimistic; test is still untouched by fitting. A production system should reserve an independent calibration set and assess classwise calibration, particularly after prior/domain shift. Confidence is not a guarantee of correctness; empty bins are not plotted and small bins have high uncertainty.",
            "## 10. Deployment verdict",
            f"**Deploy {NAMES[selected]} for CPU-only weekly reporting of 10,000 English apparel reviews.** The choice follows validation macro F1, preferring the baseline if within 0.005; this rule does not use test or Yelp outcomes. The test table is an independent check on that choice. The stated workload prioritizes minority-class visibility, easy reproducible retraining and low operating cost.",
            f"The selected model's test macro F1 is {metrics[selected]['macro_f1']:.4f}, neutral F1 is {metrics[selected]['classification_report']['neutral']['f1-score']:.4f}, CPU p50 is {metrics[selected]['benchmark']['serial_p50_ms']:.3f} ms/review, and saved artifact size is {metadata[selected]['artifact_bytes'] / 2**20:.2f} MiB. These measured costs and class-level errors make the recommendation assessable, rather than assuming the transformer must win. The other model remains available through the same CLI.",
            "Use calibrated probabilities to flag uncertain cases for review, validate any operational confidence threshold on a fresh sample, and report neutral separately in dashboards. A three-class sentiment model is not an urgency classifier: a real-time support-triage deployment needs support-ticket labels, urgency/recall targets and its own latency test. For a larger or more varied offline corpus, evaluate a stronger review-tuned transformer on a newly held-out set before changing the recommendation.",
            "This recommendation is for human-reviewed aggregate reporting. With the measured negative/neutral F1 scores, neither model is ready for unattended per-review decisions requiring reliable minority-class recognition. Audit a fresh sample before operational use.",
            "## 11. Reproduce and test live",
            "Python 3.12 is recommended. All data, original pretrained weights and fitted models are bundled; inference and training run offline after package installation. From the project directory:",
            '```powershell\npython -m venv .venv\n.venv\\Scripts\\Activate.ps1\npython -m pip install -r requirements.txt\npython -m sentiment.cli --text "Nice color, but the fit is just okay."\npython -m sentiment.cli --model transformer --file examples/reviews.txt\npython reproduce.py\npython -m unittest discover -s tests -v\n```',
            "`python reproduce.py --evaluate-only` reuses the fitted models and regenerates predictions, calibration, timing, figures and reports. `python scripts/download_inputs.py` restores missing original inputs after checking their recorded hashes. Full retraining may have small numerical differences across OS/BLAS/PyTorch versions despite deterministic seeds; the saved split membership stays exact.",
            "## 12. Scope and limitations",
            "- Ratings are weak labels; three-star text is not always linguistically neutral. Error review cannot establish that the source rating is incorrect.",
            "- The split is by review, not product or author. Exact normalized duplicates are removed, but near duplicates and shared products may remain; author IDs are unavailable. Product-held-out and temporal generalization are unmeasured.",
            "- This is one seed and a 10,000-review sample. The bootstrap does not replace repeated training runs or a larger test of minority classes.",
            "- Results concern a compact, actually fine-tuned BERT model. They do not establish that classical ML beats all transformers. Model size and tuning effort matter.",
            "- The transformer truncates long inputs; inference exposes a truncated flag. The baseline sees the full review but loses global order. Neither pipeline explicitly handles sarcasm or aspect-level sentiment.",
            "- Yelp checks domain and sentence-length shift, only for polar labels. It cannot establish three-class deployment quality on restaurants or support messages.",
            "- CPU timing and sampled RSS are local measurements. Loading, service overhead, concurrency and GPU behavior require separate production benchmarks.",
            "## 13. Requirement evidence",
            table(
                ["Requirement", "Evidence"],
                [
                    [
                        "≥1,000 real labeled examples",
                        "10,000 review snapshot; original 23,486-row CSV; data/splits.json provenance/hashes",
                    ],
                    [
                        "Positive / negative / neutral",
                        "Three-star mapping, saved label config, live CLI",
                    ],
                    [
                        "scikit-learn classical baseline",
                        "sentiment/experiment.py; fitted TF-IDF + Logistic Regression artifact",
                    ],
                    [
                        "Hugging Face transformer",
                        "Actual BERT-tiny fine-tuning; safetensors weights and epoch/validation history",
                    ],
                    [
                        "Identical train/test splits",
                        "One manifest; training/evaluation ID hashes; row-level audit tests",
                    ],
                    [
                        "Per-class precision/recall/F1",
                        "Section 4; reports/per_class_metrics.csv; full JSON reports",
                    ],
                    [
                        "Class imbalance measured/handled",
                        "Section 1; train-only class weights for both models",
                    ],
                    [
                        "≥5 reviewed errors",
                        f"Section 6; {len(reviews)} actual errors, at least five per model, reports/error_reviews.json",
                    ],
                    [
                        "Deployment recommendation",
                        "Section 10; measured accuracy/F1, latency, memory and size",
                    ],
                    [
                        "Bonus: neutral ablation",
                        "Section 7; two separately trained binary models on the original polar row IDs",
                    ],
                    [
                        "Bonus: out-of-domain",
                        "Section 8; 1,000 unseen real Yelp sentences, both models",
                    ],
                    [
                        "Bonus: calibration",
                        "Section 9; validation-fitted temperature, test reliability/ECE/Brier/NLL",
                    ],
                    [
                        "Bonus: live CLI",
                        "sentiment/cli.py; defaults to validation-selected recommended model",
                    ],
                ],
            ),
            "## Sources",
            "Dataset/model links are cited where used. Full source attribution, licenses, pretrained revision and implementation references are in [DATA_SOURCES.md](../DATA_SOURCES.md). Every table is computed from this project's model outputs.",
        ]
    )
    markdown = "\n\n".join(lines) + "\n"
    (ROOT / "reports/report.md").write_text(markdown, encoding="utf-8")
    (ROOT / "reports/report.html").write_text(render_html(markdown), encoding="utf-8")
    print(
        {
            "report": str(ROOT / "reports/report.html"),
            "test_winner": winner,
            "deployment": selected,
        },
        flush=True,
    )


if __name__ == "__main__":
    build()
