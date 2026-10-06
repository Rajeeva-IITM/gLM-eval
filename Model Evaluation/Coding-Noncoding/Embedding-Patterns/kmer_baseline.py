"""Experiment A: is the coding/non-coding separation just k-mer composition?

1. Classify with plain k-mer counts (k = 1..6) and compare with GROVER embeddings.
2. Remove everything k-mer counts can linearly explain from the embeddings, then classify the residual.
   Residual accuracy well above 50% = GROVER encodes something beyond k-mer composition.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS, coding_direction, get_embeddings

X_train, y_train, X_test, y_test, train_seqs, test_seqs = get_embeddings()


def kmer_counts(k):
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    return vec.fit_transform(train_seqs).astype(np.float32), vec.transform(test_seqs).astype(np.float32)


rows = []
kmer_train, kmer_test = [], []
for k in range(1, 7):
    K_train, K_test = kmer_counts(k)
    kmer_train.append(K_train.toarray())
    kmer_test.append(K_test.toarray())
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
    acc = accuracy_score(y_test, clf.fit(K_train, y_train).predict(K_test))
    rows.append({"features": f"{k}-mer counts", "dims": K_train.shape[1], "test_accuracy": acc})
    print(rows[-1])

acc = accuracy_score(y_test, coding_direction(X_train, y_train).predict(X_test))
rows.append({"features": "GROVER embedding", "dims": X_train.shape[1], "test_accuracy": acc})
print(rows[-1])

# Residual test: strip the k-mer-explainable part of each embedding.
K_train, K_test = np.hstack(kmer_train), np.hstack(kmer_test)
scaler = StandardScaler().fit(K_train)
ridge = Ridge(alpha=10.0).fit(scaler.transform(K_train), X_train)
pred_train, pred_test = ridge.predict(scaler.transform(K_train)), ridge.predict(scaler.transform(K_test))
print(f"R^2 of 1-6-mer counts -> embedding (test): {r2_score(X_test, pred_test):.3f}")

acc = accuracy_score(y_test, coding_direction(X_train - pred_train, y_train).predict(X_test - pred_test))
rows.append({"features": "GROVER residual after 1-6-mer regression", "dims": X_train.shape[1], "test_accuracy": acc})
print(rows[-1])

pd.DataFrame(rows).to_csv(RESULTS / "kmer_baseline.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
