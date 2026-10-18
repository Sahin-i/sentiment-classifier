"""Held-out evaluation, validation-only calibration and comparable CPU timing."""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from threadpoolctl import threadpool_limits

from .common import (
    BINARY_LABELS,
    LABELS,
    ROOT,
    digest_file,
    digest_ids,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .data import load_data
from .experiment import MemoryMeter
from .models import load_predictor


def reliability(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> dict:
    pred = probs.argmax(1)
    confidence = probs.max(1)
    correct = pred == y
    rows = []
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (confidence >= lo) & ((confidence < hi) if i < bins - 1 else (confidence <= hi))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_conf)
        rows.append(
            {
                "lower": lo,
                "upper": hi,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_conf,
            }
        )
    high = confidence >= 0.8
    return {
        "ece_10_bins": float(ece),
        "nll": float(log_loss(y, probs, labels=list(range(probs.shape[1])))),
        "multiclass_brier": float(
            np.mean(np.sum((probs - np.eye(probs.shape[1])[y]) ** 2, axis=1))
        ),
        "high_confidence_threshold": 0.8,
        "high_confidence_count": int(high.sum()),
        "high_confidence_accuracy": float(correct[high].mean()) if high.any() else None,
        "bins": rows,
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(log_t):
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(y)), y].mean())

    result = minimize_scalar(
        nll, bounds=(np.log(0.2), np.log(5)), method="bounded", options={"xatol": 1e-6}
    )
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


def scores(rows, probs, labels):
    y = [r["label"] for r in rows]
    pred = [labels[i] for i in probs.argmax(1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y, pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "label_order": labels,
    }


def benchmark(predictor, rows, repetitions=3):
    """End-to-end tokenizer/vectorizer + forward pass; warm CPU, same 128 texts."""
    texts = [r["text"] for r in rows[:128]]
    for _ in range(3):
        predictor.logits(texts[:1])
    times = []
    for _ in range(repetitions):
        for text in texts:
            started = time.perf_counter()
            predictor.logits([text])
            times.append((time.perf_counter() - started) * 1000)
    batch_times = []
    for _ in range(repetitions):
        started = time.perf_counter()
        predictor.logits(texts)
        batch_times.append(time.perf_counter() - started)
    return {
        "serial_p50_ms": float(np.median(times)),
        "serial_p95_ms": float(np.percentile(times, 95)),
        "serial_mean_ms": float(np.mean(times)),
        "batch_reviews_per_second": float(len(texts) / np.median(batch_times)),
        "serial_measurements": len(times),
        "texts": len(texts),
        "repetitions": repetitions,
        "input_id_sha256": digest_ids([r["id"] for r in rows[:128]]),
        "method": "CPU only, 4 intra-op/BLAS threads, three warmups, serial request batch size 1, identical first 128 ordered test reviews repeated three times. Includes tokenization/vectorization and forward pass; excludes imports, model load, disk I/O, queue/network overhead and temperature softmax. Batch throughput uses all 128 inputs (transformer internal batches 32).",
    }


def evaluate(model_name: str, task: str):
    manifest, parts = load_data(binary=task == "binary")
    labels = BINARY_LABELS if task == "binary" else LABELS
    directory = ROOT / "models" / task / model_name
    metadata = read_json(directory / "metadata.json")
    expected = {key: digest_ids([r["id"] for r in value]) for key, value in parts.items()}
    if (
        metadata["id_hashes"] != expected
        or metadata["split_file_sha256"] != digest_file(ROOT / "data/splits.json")
        or metadata["snapshot_sha256"] != manifest["snapshot_sha256"]
    ):
        raise ValueError("Model was trained with different data/splits; comparison rejected")
    with MemoryMeter() as memory, threadpool_limits(limits=4):
        started = time.perf_counter()
        predictor = load_predictor(model_name, task)
        load_seconds = time.perf_counter() - started
        val_logits = predictor.logits([r["text"] for r in parts["validation"]])
        val_y = np.array([labels.index(r["label"]) for r in parts["validation"]])
        temperature = fit_temperature(val_logits, val_y)
        write_json(
            directory / "calibration.json",
            {
                "temperature": temperature,
                "method": "Scalar temperature scaling; minimizes unweighted validation NLL only; never test-fitted.",
                "validation_id_sha256": expected["validation"],
            },
        )
        test = parts["test"]
        logits = predictor.logits([r["text"] for r in test])
        probs = softmax(logits, axis=1)
        calibrated = softmax(logits / temperature, axis=1)
        y = np.array([labels.index(r["label"]) for r in test])
        if not np.array_equal(probs.argmax(1), calibrated.argmax(1)):
            raise ValueError("Temperature scaling changed predicted classes")
        result = scores(test, probs, labels)
        result.update(
            {
                "model": model_name,
                "task": task,
                "n_test": len(test),
                "id_hashes": expected,
                "split_file_sha256": metadata["split_file_sha256"],
                "snapshot_sha256": manifest["snapshot_sha256"],
                "validation": scores(parts["validation"], softmax(val_logits, axis=1), labels),
                "calibration": {
                    "temperature": temperature,
                    "raw": reliability(y, probs),
                    "scaled": reliability(y, calibrated),
                },
                "benchmark": benchmark(predictor, test),
                "model_load_seconds": load_seconds,
            }
        )

        def prediction_rows(rows, logits, probs, calibrated):
            return [
                {
                    "id": row["id"],
                    "gold": row["label"],
                    "prediction": labels[int(p.argmax())],
                    "correct": bool(row["label"] == labels[int(p.argmax())]),
                    "confidence_raw": float(p.max()),
                    "confidence_calibrated": float(cp.max()),
                    "logits": list(map(float, z)),
                    "probabilities_raw": list(map(float, p)),
                    "probabilities_calibrated": list(map(float, cp)),
                }
                for row, z, p, cp in zip(rows, logits, probs, calibrated)
            ]

        write_jsonl(
            ROOT / "reports/predictions" / f"{task}_{model_name}_test.jsonl",
            prediction_rows(test, logits, probs, calibrated),
        )
        write_jsonl(
            ROOT / "reports/predictions" / f"{task}_{model_name}_validation.jsonl",
            prediction_rows(
                parts["validation"],
                val_logits,
                softmax(val_logits, axis=1),
                softmax(val_logits / temperature, axis=1),
            ),
        )
        if task == "three_class":
            provenance = read_json(ROOT / "data/ood_provenance.json")
            if digest_file(ROOT / "data/ood_yelp.jsonl") != provenance["snapshot_sha256"]:
                raise ValueError("OOD dataset changed")
            ood = read_jsonl(ROOT / "data/ood_yelp.jsonl")
            ood_logits = predictor.logits([r["text"] for r in ood])
            ood_probs = softmax(ood_logits, axis=1)
            ood_calibrated = softmax(ood_logits / temperature, axis=1)
            ood_pred = [labels[i] for i in ood_probs.argmax(1)]
            ood_result = scores(ood, ood_probs, labels)
            ood_result["polar_macro_f1_including_neutral_errors"] = float(
                f1_score(
                    [r["label"] for r in ood],
                    ood_pred,
                    labels=BINARY_LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
            ood_result["neutral_prediction_rate"] = float(np.mean(np.array(ood_pred) == "neutral"))
            ood_result["forced_binary_diagnostic"] = scores(
                ood, softmax(ood_logits[:, [0, 2]], axis=1), BINARY_LABELS
            )
            result["ood_yelp"] = ood_result
            write_jsonl(
                ROOT / "reports/predictions" / f"{task}_{model_name}_ood.jsonl",
                prediction_rows(ood, ood_logits, ood_probs, ood_calibrated),
            )
        result["sampled_peak_evaluation_process_rss_mib"] = memory.peak / 2**20
    result["sampled_peak_evaluation_process_rss_mib"] = memory.peak / 2**20
    write_json(ROOT / "reports" / f"{task}_{model_name}.json", result)
    print(
        {
            "evaluated": model_name,
            "task": task,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "serial_p50_ms": result["benchmark"]["serial_p50_ms"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transformer"], required=True)
    parser.add_argument("--task", choices=["three_class", "binary"], default="three_class")
    args = parser.parse_args()
    evaluate(args.model, args.task)
