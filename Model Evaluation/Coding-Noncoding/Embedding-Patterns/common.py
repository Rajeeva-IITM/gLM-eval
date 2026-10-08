"""Shared helpers: dataset, gLM embeddings, coding direction, reading frame.

Pick the model with the GLM_MODEL environment variable (default: GROVER), e.g.
    GLM_MODEL=LongSafari/hyenadna-large-1m-seqlen-hf pixi run python kmer_baseline.py
"""

import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("GLM_MODEL", "PoetschLab/GROVER")
MODEL_SHORT = MODEL_NAME.split("/")[-1]
DATASET = "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs"
RESULTS = Path(__file__).parent / "results" / MODEL_SHORT
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = RESULTS / "embeddings.npz"
STOPS = {"TAA", "TAG", "TGA"}
device = "cuda" if torch.cuda.is_available() else "cpu"

_model, _tokenizer = None, None


def load_gb():
    """Returns (train_seqs, train_labels, test_seqs, test_labels). Label 0 = coding, 1 = non-coding."""
    d = load_dataset(DATASET)
    return (
        d["train"]["seq"],
        np.array(d["train"]["label"]),
        d["test"]["seq"],
        np.array(d["test"]["label"]),
    )


def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()
    return _model, _tokenizer


@torch.no_grad()
def embed(seqs, batch_size=256):
    """Last hidden layer, masked mean pooling (same as glm-classification-attribution.ipynb).

    HyenaDNA's tokenizer returns no attention mask, so all tokens are averaged.
    That is fine here because every sequence is 200 bp (one token per base, no padding).
    """
    model, tokenizer = load_model()
    out = []
    for i in range(0, len(seqs), batch_size):
        inputs = tokenizer(list(seqs[i : i + batch_size]), return_tensors="pt", padding=True, truncation=True).to(device)
        if "attention_mask" in inputs:
            hidden = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
        else:
            hidden = model(input_ids=inputs["input_ids"]).last_hidden_state
            mask = torch.ones_like(hidden[..., :1])
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out)


def get_embeddings():
    """Embeds train and test once; later calls read the cache."""
    train_seqs, y_train, test_seqs, y_test = load_gb()
    if not CACHE.exists():
        print(f"Embedding train/test with {MODEL_SHORT} (cached after first run)...")
        np.savez(CACHE, X_train=embed(train_seqs), X_test=embed(test_seqs))
    data = np.load(CACHE)
    return data["X_train"], y_train, data["X_test"], y_test, train_seqs, test_seqs


def coding_direction(X_train, y_train):
    """Linear probe; clf.decision_function(X) > 0 means non-coding, so coding score = -decision_function."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return clf.fit(X_train, y_train)


def coding_score(clf, X):
    """Higher = more coding-like."""
    return -clf.decision_function(X)


def find_frame(seq):
    """Reading frame (0/1/2) if exactly one forward frame has no stop codon, else None."""
    open_frames = [f for f in range(3) if not any(seq[i : i + 3] in STOPS for i in range(f, len(seq) - 2, 3))]
    return open_frames[0] if len(open_frames) == 1 else None
