from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from .common import ROOT, clean_text, read_json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class BaselinePredictor:
    def __init__(self, directory: Path):
        self.meta = read_json(directory / "metadata.json")
        self.labels = self.meta["labels"]
        self.model = joblib.load(directory / "model.joblib")
        if list(self.model.classes_) != self.labels:
            raise ValueError("Saved baseline label order is inconsistent")

    def logits(self, texts: list[str]) -> np.ndarray:
        return np.log(np.clip(self.model.predict_proba(texts), 1e-12, 1))


class TransformerPredictor:
    def __init__(self, directory: Path, threads: int = 4):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.meta = read_json(directory / "metadata.json")
        self.labels = self.meta["labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
            .to("cpu")
            .eval()
        )
        if [self.model.config.id2label[i] for i in range(len(self.labels))] != self.labels:
            raise ValueError("Saved transformer label order is inconsistent")

    def logits(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        with self.torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[offset : offset + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.meta["max_length"],
                    return_tensors="pt",
                )
                outputs.append(self.model(**batch).logits.cpu().numpy())
        return np.concatenate(outputs)


def load_predictor(model: str = "best", task: str = "three_class"):
    if model not in {"best", "baseline", "transformer"}:
        raise ValueError("Model must be best, baseline or transformer")
    if task not in {"three_class", "binary"}:
        raise ValueError("Task must be three_class or binary")
    if not (ROOT / "data/splits.json").is_file():
        raise FileNotFoundError(
            "Experiment artifacts were not found. Install from a full repository checkout "
            "or set REVIEW_SENTIMENT_HOME to its directory."
        )
    if model == "best":
        model = read_json(ROOT / "reports/deployment.json")["selected_model"]
    directory = ROOT / "models" / task / model
    if not (directory / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Missing model artifacts in {directory}. Use the complete repository or retrain this model."
        )
    if model == "baseline":
        return BaselinePredictor(directory)
    if model == "transformer":
        return TransformerPredictor(directory)
    raise ValueError("Model must be best, baseline or transformer")


def predict(texts: list[str], model: str = "best", task: str = "three_class") -> list[dict]:
    cleaned = [clean_text(t) for t in texts]
    if not cleaned or any(not t for t in cleaned):
        raise ValueError("Provide at least one non-empty review")
    predictor = load_predictor(model, task)
    calibration = ROOT / "models" / task / predictor.meta["model"] / "calibration.json"
    temperature = read_json(calibration)["temperature"] if calibration.exists() else 1.0
    logits = predictor.logits(cleaned)
    probs = softmax(logits / temperature, axis=1)
    result = []
    for text, values in zip(cleaned, probs):
        pred = int(np.argmax(values))
        result.append(
            {
                "text": text,
                "sentiment": predictor.labels[pred],
                "confidence": float(values[pred]),
                "probabilities": dict(zip(predictor.labels, map(float, values))),
                "model": predictor.meta["model"],
                "calibrated": calibration.exists(),
                "truncated": len(predictor.tokenizer.encode(text, truncation=False, verbose=False))
                > predictor.meta["max_length"]
                if hasattr(predictor, "tokenizer")
                else False,
            }
        )
    return result
