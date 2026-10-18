"""Train each model in its own process using the verified shared split."""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from collections import Counter

import joblib
import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from .common import (
    BINARY_LABELS,
    LABELS,
    ROOT,
    SEED,
    digest_file,
    digest_ids,
    environment,
    write_json,
)
from .data import load_data


class MemoryMeter:
    """Sample process resident memory; each model runs in a fresh process."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = psutil.Process().memory_info().rss
        self.worker = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop_event.wait(0.02):
            self.peak = max(self.peak, process.memory_info().rss)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.stop_event.set()
        self.worker.join()


def macro_f1(y, pred, labels):
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def train_baseline(parts, labels, directory, threads):
    train, val = parts["train"], parts["validation"]
    texts = [r["text"] for r in train]
    targets = [r["label"] for r in train]
    candidates = []
    best_score = -1
    started = time.perf_counter()
    with threadpool_limits(limits=threads):
        for c in [0.5, 2.0, 8.0]:
            model = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.99,
                            max_features=40000,
                            sublinear_tf=True,
                            lowercase=True,
                            strip_accents=None,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c,
                            class_weight="balanced",
                            max_iter=1000,
                            solver="lbfgs",
                            random_state=SEED,
                        ),
                    ),
                ]
            )
            model.fit(texts, targets)
            if np.max(model["classifier"].n_iter_) >= 1000:
                raise RuntimeError("Logistic Regression did not converge")
            score = macro_f1(
                [r["label"] for r in val], model.predict([r["text"] for r in val]), labels
            )
            candidates.append({"C": c, "validation_macro_f1": score})
            print({"model": "baseline", **candidates[-1]}, flush=True)
            if score > best_score:
                best_score = score
                best = model
                selected_c = c
        joblib.dump(best, directory / "model.joblib", compress=3)
    return {
        "training_seconds": time.perf_counter() - started,
        "validation_macro_f1": best_score,
        "candidates": candidates,
        "selected_C": selected_c,
        "vocabulary_size": len(best["tfidf"].vocabulary_),
        "hyperparameters": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.99,
            "max_features": 40000,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "preprocessing": "Common minimal cleaning; TF-IDF lowercases, keeps unigrams/bigrams including negation; no stemming or stopword removal. Default word tokenizer drops punctuation and one-character tokens. TF-IDF vocabulary/IDF fit only on train.",
    }


def train_transformer(parts, labels, directory, threads, epochs, max_length):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    train, val = parts["train"], parts["validation"]
    class_to_id = {label: i for i, label in enumerate(labels)}
    base = ROOT / "models/pretrained"
    started = time.perf_counter()
    tokenizer = BertTokenizerFast(
        vocab_file=str(base / "vocab.txt"), do_lower_case=True, model_max_length=512
    )
    model = BertForSequenceClassification.from_pretrained(
        base,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=class_to_id,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device)
    encoded = tokenizer([r["text"] for r in train], truncation=True, max_length=max_length)
    examples = [
        {**{k: v[i] for k, v in encoded.items()}, "labels": class_to_id[row["label"]]}
        for i, row in enumerate(train)
    ]
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        examples,
        batch_size=32,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )
    val_tokens = tokenizer([r["text"] for r in val], truncation=True, max_length=max_length)
    val_examples = [{k: v[i] for k, v in val_tokens.items()} for i in range(len(val))]
    val_loader = DataLoader(
        val_examples,
        batch_size=64,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )
    weights = compute_class_weight(
        "balanced", classes=np.array(labels), y=np.array([r["label"] for r in train])
    )
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    best_score = -1
    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_total = 0.0
        for batch in loader:
            target = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, target, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
        model.eval()
        pred = []
        with torch.inference_mode():
            for batch in val_loader:
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                pred.extend(logits.argmax(-1).cpu().tolist())
        score = macro_f1([r["label"] for r in val], [labels[i] for i in pred], labels)
        row = {
            "epoch": epoch,
            "mean_batch_weighted_loss": loss_total / len(loader),
            "validation_macro_f1": score,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print({"model": "transformer", **row}, flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
    full_lengths = [
        len(ids)
        for ids in tokenizer([r["text"] for r in train], truncation=False, verbose=False)[
            "input_ids"
        ]
    ]
    return {
        "training_seconds": time.perf_counter() - started,
        "validation_macro_f1": best_score,
        "selected_epoch": best_epoch,
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "max_length": max_length,
        "train_truncated_fraction": float(np.mean(np.array(full_lengths) > max_length)),
        "pretrained_model": "prajjwal1/bert-tiny",
        "pretrained_revision": "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837",
        "training_mode": "All pretrained encoder parameters and a newly initialized classification head fine-tuned on the saved training split; no pretrained sentiment classifier used.",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "warmup_fraction": 0.1,
            "gradient_clip_norm": 1.0,
            "loss": "class-weighted cross-entropy",
            "device": "cpu",
        },
        "preprocessing": "Common minimal cleaning, uncased BERT WordPiece tokenizer from the pinned checkpoint, dynamic padding, right truncation at max_length including special tokens; punctuation/negation preserved for tokenizer.",
    }


def train(model_name, task, threads=4, epochs=5, max_length=192):
    manifest, parts = load_data(binary=task == "binary")
    for name, expected in manifest["pretrained_files"].items():
        if digest_file(ROOT / "models/pretrained" / name) != expected:
            raise ValueError(f"Pretrained file changed: {name}")
    labels = BINARY_LABELS if task == "binary" else LABELS
    directory = ROOT / "models" / task / model_name
    directory.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["label"] for r in parts["train"])
    weights = {label: len(parts["train"]) / (len(labels) * counts[label]) for label in labels}
    with MemoryMeter() as memory:
        result = (
            train_baseline(parts, labels, directory, threads)
            if model_name == "baseline"
            else train_transformer(parts, labels, directory, threads, epochs, max_length)
        )
    result.update(
        {
            "model": model_name,
            "task": task,
            "labels": labels,
            "seed": SEED,
            "threads": threads,
            "sampled_peak_process_rss_mib": memory.peak / 2**20,
            "class_weights": weights,
            "id_hashes": {k: digest_ids([r["id"] for r in v]) for k, v in parts.items()},
            "split_file_sha256": digest_file(ROOT / "data/splits.json"),
            "snapshot_sha256": manifest["snapshot_sha256"],
            "split_sizes": {k: len(v) for k, v in parts.items()},
            "environment": environment(),
            "selection": "Highest validation macro F1; tie keeps first candidate/earliest epoch. No train+validation refit.",
        }
    )
    result["artifact_bytes"] = sum(
        p.stat().st_size
        for p in directory.iterdir()
        if p.is_file() and p.name not in {"metadata.json", "calibration.json"}
    )
    write_json(directory / "metadata.json", result)
    print(
        {
            "saved": str(directory),
            "validation_macro_f1": result["validation_macro_f1"],
            "training_seconds": result["training_seconds"],
            "peak_rss_mib": result["sampled_peak_process_rss_mib"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transformer"], required=True)
    parser.add_argument("--task", choices=["three_class", "binary"], default="three_class")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    train(args.model, args.task, args.threads, args.epochs, args.max_length)
