"""Shared helpers: dataset, gLM embeddings, coding direction, reading frame.

Pick the model with the GLM_MODEL environment variable (default: GROVER), e.g.
    GLM_MODEL=LongSafari/hyenadna-large-1m-seqlen-hf pixi run python kmer_baseline.py
DNABERT-2 and Nucleotide Transformer v2 need the transformers 4.x environment:
    GLM_MODEL=zhihan1996/DNABERT-2-117M pixi run -e legacy python kmer_baseline.py
Evo and Evo 2 have their own environments and use their package model names:
    GLM_MODEL=evo-1-8k-base pixi run -e evo python kmer_baseline.py
    GLM_MODEL=evo2_7b pixi run -e evo2 python kmer_baseline.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = os.environ.get("GLM_MODEL", "PoetschLab/GROVER")
MODEL_SHORT = MODEL_NAME.split("/")[-1]
DATASET = "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs"
RESULTS = Path(__file__).parent / "results" / MODEL_SHORT
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = RESULTS / "embeddings.npz"
STOPS = {"TAA", "TAG", "TGA"}
device = "cuda" if torch.cuda.is_available() else "cpu"
IS_DNABERT2 = "DNABERT-2" in MODEL_NAME
# NT v2 exposes its custom code only through the masked-LM class; hidden states come from output_hidden_states.
IS_MLM = "nucleotide-transformer-v2" in MODEL_NAME
IS_EVO = MODEL_NAME.startswith(("evo-", "evo2_"))

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
    if _model is None and IS_EVO:
        if MODEL_NAME.startswith("evo2_"):
            from evo2 import Evo2

            wrapper = Evo2(MODEL_NAME)
        else:
            from evo import Evo

            wrapper = Evo(MODEL_NAME, device=device)
        _model, _tokenizer = wrapper.model.eval(), wrapper.tokenizer
    if _model is None:
        # Imported here because the evo2 environment has no transformers.
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, BertConfig

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model_cls = AutoModelForMaskedLM if IS_MLM else AutoModel
        # DNABERT-2 needs its config passed explicitly, or the remote config class clashes with BertConfig.
        kwargs = {"config": BertConfig.from_pretrained(MODEL_NAME)} if IS_DNABERT2 else {}
        _model = model_cls.from_pretrained(MODEL_NAME, trust_remote_code=True, **kwargs).to(device).eval()
        if IS_DNABERT2:
            # Its Triton attention kernel does not compile with current Triton; None selects the PyTorch path.
            sys.modules[type(_model).__module__].flash_attn_qkvpacked_func = None
    return _model, _tokenizer


def last_hidden(out):
    """Last-layer token states from the different output types (tuple for DNABERT-2, masked-LM output for NT v2)."""
    if isinstance(out, tuple):
        return out[0]
    if IS_MLM:
        return out.hidden_states[-1]
    return out.last_hidden_state


@torch.no_grad()
def embed(seqs, batch_size=256):
    """Last hidden layer, masked mean pooling (same as glm-classification-attribution.ipynb).

    HyenaDNA's tokenizer returns no attention mask, so all tokens are averaged.
    That is fine here because every sequence is 200 bp (one token per base, no padding).
    """
    if IS_EVO:
        # Small batches: a 7B model leaves little room on a shared 24 GB A10 (Evo 2 needs more activation memory).
        return embed_evo(seqs, min(batch_size, 8 if MODEL_NAME.startswith("evo2_") else 32))
    model, tokenizer = load_model()
    out = []
    for i in range(0, len(seqs), batch_size):
        inputs = tokenizer(list(seqs[i : i + batch_size]), return_tensors="pt", padding=True, truncation=True).to(device)
        kwargs = {"output_hidden_states": True} if IS_MLM else {}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"]
        hidden = last_hidden(model(input_ids=inputs["input_ids"], **kwargs))
        mask = inputs["attention_mask"].unsqueeze(-1).float() if "attention_mask" in inputs else torch.ones_like(hidden[..., :1])
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def embed_evo(seqs, batch_size):
    """Evo/Evo 2: mean of the final-norm output (their last hidden layer), captured with a forward hook.

    Character-level tokens, no special tokens; all sequences are 200 bp, so no padding is needed.
    """
    model, tokenizer = load_model()
    captured = {}
    handle = model.norm.register_forward_hook(lambda module, inputs, output: captured.update(hidden=output))
    out = []
    try:
        for i in range(0, len(seqs), batch_size):
            ids = torch.tensor([tokenizer.tokenize(s) for s in seqs[i : i + batch_size]], dtype=torch.long, device=device)
            model(ids)
            out.append(captured["hidden"].float().mean(1).cpu().numpy())
    finally:
        handle.remove()
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
