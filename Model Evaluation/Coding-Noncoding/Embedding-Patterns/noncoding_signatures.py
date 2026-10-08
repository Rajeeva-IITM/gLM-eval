"""Which features of non-coding DNA make the models call it non-coding?

1. Label every non-coding test sequence with interpretable features:
   - best-matching transposable-element family (local alignment to Dfam consensus sequences, both strands;
     a hit must beat the 99.5th percentile score of shuffled sequences for that family),
   - simple-repeat and low-complexity scores, longest homopolymer, GC content, CpG observed/expected.
2. For every model with cached embeddings (results/<model>/embeddings.npz), compute the coding score of each
   non-coding sequence and ask which features separate sequences called non-coding from those called coding.

Model-independent; uses cached embeddings only, so it needs no GPU. Run in the default environment:
    pixi run python noncoding_signatures.py
"""

import json
import random
import urllib.request
from collections import Counter
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from common import coding_direction, coding_score, load_gb

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CONSENSUS = HERE / "data" / "repeat_consensus.fa"
FEATURES = RESULTS / "noncoding_features.csv"
random.seed(0)

# Dfam family name -> repeat class. A few representative families per major class of human repeats.
# LINEs use their 3' end, which is the part most genomic copies keep (they are usually 5'-truncated).
FAMILIES = {
    "AluY": "Alu", "AluSx": "Alu", "AluJb": "Alu",
    "MIR": "MIR",
    "L1HS_3end": "L1", "L1PA7_3end": "L1", "L1MB7_3end": "L1",
    "L2a_3end": "L2", "L2b_3end": "L2",
    "THE1B": "LTR", "MLT1A0": "LTR", "MSTA": "LTR", "LTR12C": "LTR",
    "MER5A": "DNA transposon", "Tigger1": "DNA transposon",
}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

aligner = PairwiseAligner(mode="local", match_score=2, mismatch_score=-3, open_gap_score=-5, extend_gap_score=-2)


def fetch_consensus():
    """Download the Dfam consensus sequences once and keep them in data/."""
    CONSENSUS.parent.mkdir(exist_ok=True)
    records = []
    for name in FAMILIES:
        url = f"https://dfam.org/api/families?name={name}&format=summary&limit=100"
        hits = json.load(urllib.request.urlopen(url))["results"]
        accession = next(h["accession"] for h in hits if h["name"] == name)
        fasta = urllib.request.urlopen(f"https://dfam.org/api/families/{accession}/sequence?format=fasta").read().decode()
        records.append(f">{name}\n" + "".join(fasta.splitlines()[1:]).upper() + "\n")
    CONSENSUS.write_text("".join(records))


def best_scores(seq):
    """Best local-alignment score of seq (either strand) against every consensus."""
    rc = seq.translate(COMPLEMENT)[::-1]
    return [max(aligner.score(seq, cons), aligner.score(rc, cons)) for cons in CONSENSUS_SEQS.values()]


def simple_features(seq):
    n = len(seq)
    kmers3 = Counter(seq[i : i + 3] for i in range(n - 2))
    p = np.array(list(kmers3.values())) / (n - 2)
    c, g = seq.count("C"), seq.count("G")
    return {
        "gc": (c + g) / n,
        "cpg_obs_exp": seq.count("CG") * n / (c * g) if c and g else 0.0,
        "longest_homopolymer": max(len(list(run)) for _, run in groupby(seq)),
        # fraction of positions equal to the base p steps later; microsatellites score high for their period
        "tandem_repeat_score": max(np.mean([seq[i] == seq[i + k] for i in range(n - k)]) for k in range(1, 7)),
        "trinucleotide_entropy": float(-(p * np.log2(p)).sum()),  # low = low complexity
    }


def build_features(seqs):
    with Pool(32) as pool:
        scores = np.array(pool.map(best_scores, seqs, chunksize=50))
        # Null: the same alignment on shuffled copies of 1,000 sequences gives a per-family score threshold.
        shuffled = ["".join(random.sample(s, len(s))) for s in random.sample(seqs, 1000)]
        thresholds = np.percentile(pool.map(best_scores, shuffled, chunksize=50), 99.5, axis=0)
    names = list(CONSENSUS_SEQS)
    passed = scores > thresholds
    best = np.where(passed, scores - thresholds, -np.inf).argmax(1)
    df = pd.DataFrame([simple_features(s) for s in seqs])
    df["repeat_family"] = [names[b] if passed[i].any() else "none" for i, b in enumerate(best)]
    df["repeat_class"] = df.repeat_family.map(FAMILIES).fillna("none")
    df.loc[(df.repeat_class == "none") & (df.tandem_repeat_score > 0.6), "repeat_class"] = "simple repeat"
    print("Per-family score thresholds:", dict(zip(names, thresholds.round(1))))
    return df


if __name__ == "__main__":
    if not CONSENSUS.exists():
        fetch_consensus()
    CONSENSUS_SEQS = {r.id: str(r.seq) for r in SeqIO.parse(CONSENSUS, "fasta")}

    train_seqs, y_train, test_seqs, y_test = load_gb()
    nc_idx = np.where(y_test == 1)[0]
    if not FEATURES.exists():
        build_features([test_seqs[i] for i in nc_idx]).to_csv(FEATURES, index=False)
    feats = pd.read_csv(FEATURES)
    print(feats.repeat_class.value_counts().to_string())

    continuous = ["gc", "cpg_obs_exp", "longest_homopolymer", "tandem_repeat_score", "trinucleotide_entropy"]
    design = pd.concat([feats[continuous], pd.get_dummies(feats.repeat_class, dtype=float)], axis=1)

    by_class, correlations, explained = [], [], []
    for cache in sorted(RESULTS.glob("*/embeddings.npz")):
        model = cache.parent.name
        data = np.load(cache)
        score = coding_score(coding_direction(data["X_train"], y_train), data["X_test"][nc_idx])
        called_noncoding = score < 0
        for cls, group in feats.groupby("repeat_class"):
            by_class.append({"model": model, "repeat_class": cls, "n": len(group),
                             "frac_called_noncoding": called_noncoding[group.index].mean(),
                             "mean_coding_score": score[group.index].mean()})
        for f in continuous:
            correlations.append({"model": model, "feature": f, "spearman_vs_coding_score": spearmanr(feats[f], score)[0]})
        r2 = cross_val_score(LinearRegression(), design, score, cv=5, scoring="r2").mean()
        explained.append({"model": model, "r2_features_to_coding_score": r2})
        print(f"{model}: features explain R^2 = {r2:.3f} of the coding score within non-coding sequences")

    by_class, correlations, explained = pd.DataFrame(by_class), pd.DataFrame(correlations), pd.DataFrame(explained)
    by_class.to_csv(RESULTS / "noncoding_by_repeat_class.csv", index=False)
    correlations.to_csv(RESULTS / "noncoding_feature_correlations.csv", index=False)
    explained.to_csv(RESULTS / "noncoding_features_explained.csv", index=False)

    heat = by_class.pivot(index="model", columns="repeat_class", values="frac_called_noncoding")
    counts = feats.repeat_class.value_counts()
    heat = heat[counts.index]
    heat.columns = [f"{c}\n(n={counts[c]})" for c in heat.columns]
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(heat) + 2))
    im = ax.imshow(heat.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns, fontsize=8)
    ax.set_yticks(range(heat.shape[0]), heat.index, fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if heat.values[i, j] > 0.6 else "white")
    fig.colorbar(im, label="Fraction of non-coding sequences called non-coding")
    ax.set_title("Which non-coding sequences do the models recognise as non-coding?")
    fig.tight_layout()
    fig.savefig(RESULTS / "noncoding_by_repeat_class.png", dpi=200)
    print(heat.round(2).to_string())
    print(correlations.pivot(index="model", columns="feature", values="spearman_vs_coding_score").round(2).to_string())
