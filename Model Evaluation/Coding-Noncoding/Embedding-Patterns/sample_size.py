"""Robustness to training-set size: how much do probe accuracy and the probe direction change with fewer sequences?

For every model and dataset (200 bp GB, and CDS vs intergenic / lncRNA / intron), a reference probe is trained on the
whole training set, as in the other experiments. Probes are then trained on balanced random subsets of n training
sequences (10 subsets per n) and compared with the reference on the unchanged test set:
  - test accuracy (balanced accuracy for full length),
  - cosine similarity of the probe weights with the reference, in raw embedding space (coef / scaler scale),
  - Pearson correlation of test-set coding scores with the reference scores. This is the cosine between the two
    directions measured on the embedding distribution, so it ignores directions the data never varies along.
    Embedding dimensions are strongly correlated, so very different weight vectors can give the same scores.
  - Spearman correlation of the scores, and the fraction of identical coding / non-coding calls.

Uses cached embeddings only (no GPU):
    pixi run python sample_size.py
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from common import HERE, NONCODING_SETS, coding_direction, coding_score, load_gb

RESULTS = HERE / "results"
SIZES = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
N_SEEDS = 10


def datasets():
    """(model, dataset, X_train, y_train, X_test, y_test) for every cached model; label 0 = coding, 1 = non-coding."""
    _, y_train, _, y_test = load_gb()
    for cache in sorted(RESULTS.glob("*/embeddings.npz")):
        d = np.load(cache)
        yield cache.parent.name, "gb", d["X_train"], y_train, d["X_test"], y_test
    for model_dir in sorted((RESULTS / "full" / "embeddings").iterdir()):
        coding = np.load(model_dir / "coding.npz")
        for name in NONCODING_SETS:
            nc = np.load(model_dir / f"{name}.npz")
            yield (model_dir.name, name,
                   np.vstack([coding["X_train"], nc["X_train"]]),
                   np.r_[np.zeros(len(coding["X_train"]), int), np.ones(len(nc["X_train"]), int)],
                   np.vstack([coding["X_test"], nc["X_test"]]),
                   np.r_[np.zeros(len(coding["X_test"]), int), np.ones(len(nc["X_test"]), int)])


def raw_weights(clf):
    """Probe weights in raw embedding space, so probes fitted with different scalers are comparable."""
    scaler, lr = clf.steps[0][1], clf.steps[-1][1]
    return lr.coef_[0] / scaler.scale_


def fit_subset(X_train, y_train, X_test, y_test, n, seed, ref_w, ref_score, metric):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y_train == c)[0], n // 2, replace=False) for c in (0, 1)])
    assert (y_train[idx] == 0).sum() == (y_train[idx] == 1).sum() == n // 2
    clf = coding_direction(X_train[idx], y_train[idx])
    w, score = raw_weights(clf), coding_score(clf, X_test)
    return {"n": n, "seed": seed, "accuracy": metric(y_test, clf.predict(X_test)),
            "cosine_to_reference": float(w @ ref_w / np.linalg.norm(w) / np.linalg.norm(ref_w)),
            "pearson_to_reference": pearsonr(score, ref_score)[0],
            "spearman_to_reference": spearmanr(score, ref_score)[0],
            "same_call_as_reference": float(((score > 0) == (ref_score > 0)).mean())}


if __name__ == "__main__":
    rows = []
    with Parallel(n_jobs=32) as parallel:
        for model, ds, X_train, y_train, X_test, y_test in datasets():
            metric = accuracy_score if ds == "gb" else balanced_accuracy_score
            ref = coding_direction(X_train, y_train)
            ref_w, ref_score = raw_weights(ref), coding_score(ref, X_test)
            ref_acc = metric(y_test, ref.predict(X_test))
            max_n = 2 * np.bincount(y_train).min()
            sizes = [n for n in SIZES if n < max_n] + [max_n]
            # A subset as large as the whole (balanced) training set is the same every time, so fit it once.
            jobs = [(n, s) for n in sizes for s in range(1 if n == len(y_train) else N_SEEDS)]
            out = parallel(delayed(fit_subset)(X_train, y_train, X_test, y_test, n, s, ref_w, ref_score, metric)
                           for n, s in jobs)
            rows += [dict(r, model=model, dataset=ds, reference_accuracy=ref_acc, n_train_total=len(y_train)) for r in out]
            print(f"{model} {ds}: reference {ref_acc:.3f}; n={sizes[-1]} mean acc "
                  f"{np.mean([r['accuracy'] for r in out if r['n'] == sizes[-1]]):.3f}", flush=True)

    df = pd.DataFrame(rows)[["model", "dataset", "n", "seed", "n_train_total", "reference_accuracy", "accuracy",
                             "cosine_to_reference", "pearson_to_reference", "spearman_to_reference",
                             "same_call_as_reference"]]
    df.to_csv(RESULTS / "sample_size.csv", index=False)
    metrics = ["accuracy", "cosine_to_reference", "pearson_to_reference", "spearman_to_reference", "same_call_as_reference"]
    summary = df.groupby(["model", "dataset", "n"]).agg(
        reference_accuracy=("reference_accuracy", "first"), **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")})
    summary.reset_index().to_csv(RESULTS / "sample_size_summary.csv", index=False)
    print(summary.round(3).to_string())
