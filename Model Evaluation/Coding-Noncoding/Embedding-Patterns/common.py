"""Shared helpers: dataset, gLM embeddings, coding direction, reading frame.

Pick the model with the GLM_MODEL environment variable (default: GROVER), e.g.
    GLM_MODEL=LongSafari/hyenadna-large-1m-seqlen-hf pixi run python kmer_baseline.py
DNABERT-2 and Nucleotide Transformer v2 need the transformers 4.x environment:
    GLM_MODEL=zhihan1996/DNABERT-2-117M pixi run -e legacy python kmer_baseline.py
Evo and Evo 2 have their own environments and use their package model names:
    GLM_MODEL=evo-1-8k-base pixi run -e evo python kmer_baseline.py
    GLM_MODEL=evo2_7b pixi run -e evo2 python kmer_baseline.py
GLM_DATASET picks the data: "gb" (default, Genomic Benchmarks 200 bp) or "full" (full-length human CDS vs
intergenic, lncRNA and intron sequences from prepare_full_length.py; every script then runs all three datasets).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = os.environ.get("GLM_MODEL", "PoetschLab/GROVER")
MODEL_SHORT = MODEL_NAME.split("/")[-1]
DATASET = "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs"
FULL = os.environ.get("GLM_DATASET", "gb") == "full"
HERE = Path(__file__).parent
NONCODING_SETS = ["intergenic", "lncrna", "intron"]
RESULTS = HERE / "results" / ("full" if FULL else MODEL_SHORT)
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = RESULTS / "embeddings.npz"
STOPS = {"TAA", "TAG", "TGA"}
device = "cuda" if torch.cuda.is_available() else "cpu"
IS_DNABERT2 = "DNABERT-2" in MODEL_NAME
# NT v2 exposes its custom code only through the masked-LM class; hidden states come from output_hidden_states.
IS_MLM = "nucleotide-transformer-v2" in MODEL_NAME
IS_EVO = MODEL_NAME.startswith(("evo-", "evo2_"))
# Longest chunk (bp) each model embeds in one pass; longer sequences are split into near-equal chunks and averaged.
# GROVER/DNABERT-2: 1.5 kb stays under 512 BPE tokens even for low-compressing sequence (2 kb can reach ~550).
# Evo windows are limited by memory on a shared 24 GB GPU, not context (Evo 1 at 8 kb peaks at ~16 GB).
# "seqlen-1k" = the Caduceus models trained on 1 kb.
WINDOWS = {"GROVER": 1500, "DNABERT-2": 1500, "hyenadna-tiny-1k": 1000, "seqlen-1k": 1000, "evo-1": 4000, "evo2_": 2000}
WINDOW = next((w for key, w in WINDOWS.items() if key in MODEL_NAME), 10_000)

# Caduceus (bidirectional Mamba) also takes no attention mask, and padding would leak into its reverse pass.
NO_MASK = IS_EVO or "hyenadna" in MODEL_NAME or "caduceus" in MODEL_NAME
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


def embed(seqs, batch_size=256):
    """Embedding of each sequence: chunks of at most WINDOW bp, embedded and averaged weighted by chunk length.

    With 200 bp inputs every sequence is a single chunk, so this is the plain per-sequence embedding.
    """
    chunks, owner = [], []
    for i, seq in enumerate(seqs):
        n = -(-len(seq) // WINDOW)
        bounds = np.linspace(0, len(seq), n + 1).astype(int)
        chunks += [seq[a:b] for a, b in zip(bounds, bounds[1:])]
        owner += [i] * n
    # Sort by length so batches hold similar lengths, embed, then restore order.
    order = np.argsort([len(c) for c in chunks], kind="stable")
    X_sorted = embed_chunks([chunks[j] for j in order], batch_size)
    X = np.empty_like(X_sorted)
    X[order] = X_sorted
    weights = np.array([len(c) for c in chunks], dtype=float)
    owner = np.array(owner)
    out = np.zeros((len(seqs), X.shape[1]))
    np.add.at(out, owner, X * weights[:, None])
    return (out / np.bincount(owner, weights=weights)[:, None]).astype(np.float32)


def batches(chunks, batch_size):
    """Batches of consecutive chunks (sorted by length) capped at about batch_size x 200 bp.

    Models without an attention mask (HyenaDNA, Evo) only batch chunks of identical length, because padding would
    change their (causal) hidden states.
    """
    budget, i = batch_size * 200, 0
    while i < len(chunks):
        j = i + 1
        while j < len(chunks) and (j - i + 1) * len(chunks[j]) <= budget:
            if NO_MASK and len(chunks[j]) != len(chunks[i]):
                break
            j += 1
        yield chunks[i:j]
        i = j


@torch.no_grad()
def embed_chunks(seqs, batch_size):
    """Last hidden layer, masked mean pooling (same as glm-classification-attribution.ipynb)."""
    if IS_EVO:
        # Small batches: a 7B model leaves little room on a shared 24 GB A10 (Evo 2 needs more activation memory).
        return embed_evo(seqs, min(batch_size, 8 if MODEL_NAME.startswith("evo2_") else 32))
    model, tokenizer = load_model()
    out = []
    for batch in batches(seqs, batch_size):
        inputs = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True).to(device)
        kwargs = {"output_hidden_states": True} if IS_MLM else {}
        if "attention_mask" in inputs and not NO_MASK:
            kwargs["attention_mask"] = inputs["attention_mask"]
        hidden = last_hidden(model(input_ids=inputs["input_ids"], **kwargs))
        mask = inputs["attention_mask"].unsqueeze(-1).float() if "attention_mask" in inputs else torch.ones_like(hidden[..., :1])
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def embed_evo(seqs, batch_size):
    """Evo/Evo 2: mean of the final-norm output (their last hidden layer), captured with a forward hook.

    Character-level tokens, no special tokens; batches() only groups equal-length chunks, so there is no padding.
    """
    model, tokenizer = load_model()
    captured = {}
    handle = model.norm.register_forward_hook(lambda module, inputs, output: captured.update(hidden=output))
    out = []
    try:
        for batch in batches(seqs, batch_size):
            ids = torch.tensor([tokenizer.tokenize(s) for s in batch], dtype=torch.long, device=device)
            model(ids)
            out.append(captured["hidden"].float().mean(1).cpu().numpy())
    finally:
        handle.remove()
    return np.vstack(out)


def load_full_set(name):
    df = pd.read_csv(HERE / "data" / "full" / f"{name}.csv.gz")
    return df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)


def full_set_embeddings(name):
    """(X_train, X_test) for one full-length set, cached per model so the coding set is embedded only once."""
    cache = RESULTS / "embeddings" / MODEL_SHORT / f"{name}.npz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        train, test = load_full_set(name)
        print(f"Embedding {name} with {MODEL_SHORT} (cached after first run)...")
        np.savez(cache, X_train=embed(train.seq.tolist()), X_test=embed(test.seq.tolist()))
    data = np.load(cache)
    return data["X_train"], data["X_test"]


def get_datasets():
    """Datasets to analyse, as dicts with name, out_dir, X/y/seqs for train and test, and test reading frames.

    GB: one dataset, results in results/<model>/. Full: CDS vs each non-coding set, results in
    results/full/cds_vs_<set>/<model>/. Label 0 = coding, 1 = non-coding. Coding frames are known (0) for full
    CDS and inferred with find_frame for GB; non-coding sequences are read in frame 0 as a control.
    """
    if not FULL:
        X_train, y_train, X_test, y_test, train_seqs, test_seqs = get_embeddings()
        frames = [find_frame(s) if y == 0 else 0 for s, y in zip(test_seqs, y_test)]
        return [dict(name="gb", out_dir=RESULTS, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                     train_seqs=list(train_seqs), test_seqs=list(test_seqs), test_frames=frames)]
    coding_train, coding_test = load_full_set("coding")
    Xc_train, Xc_test = full_set_embeddings("coding")
    datasets = []
    for name in NONCODING_SETS:
        nc_train, nc_test = load_full_set(name)
        Xn_train, Xn_test = full_set_embeddings(name)
        out_dir = RESULTS / f"cds_vs_{name}" / MODEL_SHORT
        out_dir.mkdir(parents=True, exist_ok=True)
        datasets.append(dict(
            name=f"cds_vs_{name}", out_dir=out_dir,
            X_train=np.vstack([Xc_train, Xn_train]), X_test=np.vstack([Xc_test, Xn_test]),
            y_train=np.r_[np.zeros(len(Xc_train), int), np.ones(len(Xn_train), int)],
            y_test=np.r_[np.zeros(len(Xc_test), int), np.ones(len(Xn_test), int)],
            train_seqs=coding_train.seq.tolist() + nc_train.seq.tolist(),
            test_seqs=coding_test.seq.tolist() + nc_test.seq.tolist(),
            test_frames=[0] * (len(coding_test) + len(nc_test)),
        ))
    return datasets


def get_embeddings():
    """GB: embeds train and test once; later calls read the cache."""
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
