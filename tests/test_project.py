"""Behavioral and artifact checks for the assessed comparison."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report

from sentiment import data
from sentiment.common import (
    LABELS,
    ROOT,
    clean_text,
    digest_file,
    digest_ids,
    read_json,
    read_jsonl,
)
from sentiment.evaluate import fit_temperature, reliability
from sentiment.models import load_predictor, predict


class DataIntegrityTests(unittest.TestCase):
    def test_real_source_and_minimum_size(self):
        manifest, parts = data.load_data()
        self.assertGreaterEqual(manifest["sample_size"], 1000)
        self.assertEqual(manifest["source_rows"], 23486)
        self.assertEqual(
            digest_file(ROOT / "data/raw/clothing_reviews.csv"), manifest["raw_sha256"]
        )
        self.assertEqual(sum(map(len, parts.values())), manifest["sample_size"])

    def test_all_classes_and_no_leakage(self):
        manifest, parts = data.load_data()
        all_ids, all_hashes = [], []
        for name, rows in parts.items():
            self.assertEqual(set(r["label"] for r in rows), set(LABELS))
            self.assertEqual(digest_ids([r["id"] for r in rows]), manifest["id_hashes"][name])
            all_ids.extend(r["id"] for r in rows)
            all_hashes.extend(r["text_sha256"] for r in rows)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_hashes), len(set(all_hashes)))

    def test_preprocessing_preserves_negation_and_punctuation(self):
        self.assertEqual(
            clean_text("  &lt;b&gt;Not&lt;/b&gt; bad!\nI wouldn't buy again.\u200b "),
            "Not bad! I wouldn't buy again.",
        )
        self.assertEqual(
            [data.rating_label(x) for x in [1, 2, 3, 4, 5]],
            ["negative", "negative", "neutral", "positive", "positive"],
        )
        with self.assertRaises(ValueError):
            data.rating_label(0)

    def test_modified_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            shutil.copy(ROOT / "data/splits.json", root / "data/splits.json")
            shutil.copy(ROOT / "data/reviews.jsonl", root / "data/reviews.jsonl")
            with (root / "data/reviews.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with patch.object(data, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "snapshot hash mismatch"):
                    data.load_data()

    def test_modified_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            shutil.copy(ROOT / "data/reviews.jsonl", root / "data/reviews.jsonl")
            manifest = read_json(ROOT / "data/splits.json")
            manifest["splits"]["test"][0] = manifest["splits"]["train"][0]
            (root / "data/splits.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(data, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "split hash mismatch"):
                    data.load_data()

    def test_binary_is_same_polar_membership(self):
        _, full = data.load_data()
        _, binary = data.load_data(binary=True)
        for name in full:
            self.assertEqual(
                [r["id"] for r in binary[name]],
                [r["id"] for r in full[name] if r["label"] != "neutral"],
            )


class EvaluationArtifactTests(unittest.TestCase):
    def test_both_models_used_same_all_partitions(self):
        manifest, parts = data.load_data()
        for model in ["baseline", "transformer"]:
            metadata = read_json(ROOT / f"models/three_class/{model}/metadata.json")
            result = read_json(ROOT / f"reports/three_class_{model}.json")
            self.assertEqual(metadata["id_hashes"], manifest["id_hashes"])
            self.assertEqual(result["id_hashes"], manifest["id_hashes"])
            self.assertEqual(metadata["split_file_sha256"], digest_file(ROOT / "data/splits.json"))
            rows = read_jsonl(ROOT / f"reports/predictions/three_class_{model}_test.jsonl")
            self.assertEqual([r["id"] for r in rows], manifest["splits"]["test"])
            self.assertEqual([r["gold"] for r in rows], [r["label"] for r in parts["test"]])

    def test_metrics_recompute_from_predictions(self):
        for task in ["three_class", "binary"]:
            labels = LABELS if task == "three_class" else ["negative", "positive"]
            for model in ["baseline", "transformer"]:
                rows = read_jsonl(ROOT / f"reports/predictions/{task}_{model}_test.jsonl")
                result = read_json(ROOT / f"reports/{task}_{model}.json")
                recomputed = classification_report(
                    [r["gold"] for r in rows],
                    [r["prediction"] for r in rows],
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
                for label in labels:
                    for metric in ["precision", "recall", "f1-score", "support"]:
                        self.assertAlmostEqual(
                            recomputed[label][metric],
                            result["classification_report"][label][metric],
                        )
                self.assertAlmostEqual(result["macro_f1"], recomputed["macro avg"]["f1-score"])

    def test_actual_finetuning_and_balanced_weights(self):
        for task in ["three_class", "binary"]:
            _, parts = data.load_data(binary=task == "binary")
            counts = Counter(r["label"] for r in parts["train"])
            for model in ["baseline", "transformer"]:
                meta = read_json(ROOT / f"models/{task}/{model}/metadata.json")
                for label, count in counts.items():
                    self.assertAlmostEqual(
                        meta["class_weights"][label], len(parts["train"]) / (len(counts) * count)
                    )
                self.assertEqual(meta["split_sizes"]["train"], len(parts["train"]))
            trans = read_json(ROOT / f"models/{task}/transformer/metadata.json")
            self.assertEqual(len(trans["history"]), 5)
            self.assertGreater(
                (ROOT / f"models/{task}/transformer/model.safetensors").stat().st_size, 1_000_000
            )
            selected = max(trans["history"], key=lambda row: row["validation_macro_f1"])
            self.assertEqual(trans["selected_epoch"], selected["epoch"])

    def test_five_real_reviewed_errors_per_model(self):
        reviews = read_json(ROOT / "reports/error_reviews.json")
        for model in ["baseline", "transformer"]:
            predictions = {
                r["id"]: r
                for r in read_jsonl(ROOT / f"reports/predictions/three_class_{model}_test.jsonl")
            }
            cases = [r for r in reviews if r["model"] == model]
            self.assertGreaterEqual(len(cases), 5)
            self.assertEqual(len(cases), len(set(r["id"] for r in cases)))
            for row in cases:
                self.assertFalse(predictions[row["id"]]["correct"])
                self.assertGreater(len(row["hypothesis"].split()), 12)

    def test_ood_never_in_main_split(self):
        _, parts = data.load_data()
        main = {r["text"].casefold() for group in parts.values() for r in group}
        ood = read_jsonl(ROOT / "data/ood_yelp.jsonl")
        self.assertEqual(len(ood), 1000)
        self.assertEqual(Counter(r["label"] for r in ood), {"negative": 500, "positive": 500})
        self.assertFalse(main.intersection(r["text"].casefold() for r in ood))

    def test_probability_and_calibration_invariants(self):
        for model in ["baseline", "transformer"]:
            rows = read_jsonl(ROOT / f"reports/predictions/three_class_{model}_test.jsonl")
            for row in rows:
                raw = np.array(row["probabilities_raw"])
                scaled = np.array(row["probabilities_calibrated"])
                self.assertAlmostEqual(raw.sum(), 1, places=6)
                self.assertAlmostEqual(scaled.sum(), 1, places=6)
                self.assertTrue(np.all(raw >= 0) and np.all(scaled >= 0))
                self.assertEqual(raw.argmax(), scaled.argmax())

    def test_latency_compares_same_reviews(self):
        a, b = [
            read_json(ROOT / f"reports/three_class_{m}.json")["benchmark"]
            for m in ["baseline", "transformer"]
        ]
        self.assertEqual(a["input_id_sha256"], b["input_id_sha256"])
        self.assertEqual(a["serial_measurements"], b["serial_measurements"])
        self.assertGreater(a["serial_p50_ms"], 0)
        self.assertGreater(b["serial_p50_ms"], 0)


class CalibrationBehaviorTests(unittest.TestCase):
    def test_overconfident_errors_temperature_improves_nll(self):
        logits = np.array([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
        y = np.array([0, 1, 1, 0])
        temperature = fit_temperature(logits, y)
        self.assertGreater(temperature, 1)
        self.assertLess(
            reliability(y, softmax(logits / temperature, axis=1))["nll"],
            reliability(y, softmax(logits, axis=1))["nll"],
        )

    def test_confidence_one_is_in_last_bin(self):
        result = reliability(np.array([0, 1]), np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(result["bins"][-1]["count"], 2)
        self.assertEqual(result["ece_10_bins"], 0)
        self.assertEqual(result["high_confidence_accuracy"], 1)


class LivePredictionTests(unittest.TestCase):
    def test_each_saved_model_predicts_valid_distribution(self):
        for model in ["baseline", "transformer"]:
            result = predict(
                [
                    "The fabric is wonderful and the fit is perfect.",
                    "It broke immediately. Very disappointed.",
                ],
                model,
            )
            self.assertEqual(len(result), 2)
            for row in result:
                self.assertIn(row["sentiment"], LABELS)
                self.assertEqual(set(row["probabilities"]), set(LABELS))
                self.assertAlmostEqual(sum(row["probabilities"].values()), 1, places=6)

    def test_live_logits_match_saved_test_predictions(self):
        _, parts = data.load_data()
        texts = [r["text"] for r in parts["test"][:8]]
        for model in ["baseline", "transformer"]:
            predictor = load_predictor(model)
            logits = predictor.logits(texts)
            saved = read_jsonl(ROOT / f"reports/predictions/three_class_{model}_test.jsonl")[:8]
            np.testing.assert_allclose(
                logits, np.array([r["logits"] for r in saved]), atol=2e-5, rtol=2e-5
            )

    def test_empty_reviews_are_rejected(self):
        with self.assertRaises(ValueError):
            predict(["   "])

    def test_cli_returns_json_and_rejects_blank_input(self):
        command = [sys.executable, "-m", "sentiment.cli"]
        valid = subprocess.run(
            command + ["--text", "Love the fit and fabric."],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn(json.loads(valid.stdout)["sentiment"], LABELS)
        blank = subprocess.run(
            command + ["--text", "   "], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(blank.returncode, 2)
        self.assertIn("non-empty", blank.stderr)


if __name__ == "__main__":
    unittest.main()
