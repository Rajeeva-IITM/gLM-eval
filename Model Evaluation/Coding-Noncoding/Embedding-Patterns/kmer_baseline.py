"""Experiment A: is the coding/non-coding separation just k-mer composition?

1. Classify with plain k-mer frequencies (k = 1..6) and compare with gLM embeddings.
2. Remove everything k-mer counts can linearly explain from the embeddings, then classify the residual.
   Residual accuracy well above 50% = the gLM encodes something beyond k-mer composition.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from common import MODEL_SHORT, coding_direction, get_datasets


def kmer_freqs(k, train_seqs, test_seqs):
    """k-mer frequencies (counts / total), so sequences of different lengths are comparable."""
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    K_train, K_test = vec.fit_transform(train_seqs), vec.transform(test_seqs)
    return normalize(K_train, norm="l1").astype(np.float32), normalize(K_test, norm="l1").astype(np.float32)


def scores(y_true, y_pred):
    """Accuracy plus balanced accuracy (chance = 0.5 even when the test classes differ in size)."""
    return {"test_accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}


def run(ds):
    X_train, y_train, X_test, y_test = ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"]
    rows, kmer_train, kmer_test = [], [], []
    for k in range(1, 7):
        K_train, K_test = kmer_freqs(k, ds["train_seqs"], ds["test_seqs"])
        kmer_train.append(K_train.toarray())
        kmer_test.append(K_test.toarray())
        clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
        rows.append({"features": f"{k}-mer frequencies", "dims": K_train.shape[1],
                     **scores(y_test, clf.fit(K_train, y_train).predict(K_test))})
        print(rows[-1])

    rows.append({"features": f"{MODEL_SHORT} embedding", "dims": X_train.shape[1],
                 **scores(y_test, coding_direction(X_train, y_train).predict(X_test))})
    print(rows[-1])

    # Residual test: strip the k-mer-explainable part of each embedding.
    K_train, K_test = np.hstack(kmer_train), np.hstack(kmer_test)
    scaler = StandardScaler().fit(K_train)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(K_train), X_train)
    pred_train, pred_test = ridge.predict(scaler.transform(K_train)), ridge.predict(scaler.transform(K_test))
    print(f"R^2 of 1-6-mer frequencies -> embedding (test): {r2_score(X_test, pred_test):.3f}")

    rows.append({"features": f"{MODEL_SHORT} residual after 1-6-mer regression", "dims": X_train.shape[1],
                 **scores(y_test, coding_direction(X_train - pred_train, y_train).predict(X_test - pred_test))})
    print(rows[-1])

    pd.DataFrame(rows).to_csv(ds["out_dir"] / "kmer_baseline.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    for ds in get_datasets():
        print(f"=== {ds['name']}")
        run(ds)
