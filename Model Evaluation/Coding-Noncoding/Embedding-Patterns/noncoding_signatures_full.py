"""Full-length version of noncoding_signatures.py: which features of non-coding DNA make models call it non-coding?

For each non-coding test sequence (intergenic, lncRNA, intron), features are:
  - fraction of its bases covered by each RepeatMasker class (UCSC hg38 rmsk; Alu, L1, MIR, L2, LTR, DNA transposon,
    simple repeat, low complexity, other), from the sequence's genomic intervals,
  - the length-normalised simple features from noncoding_signatures.py (GC, CpG o/e, homopolymer, tandem repeat,
    trinucleotide entropy).
For every model with cached full-length embeddings, a probe trained on CDS vs that set scores the test sequences,
and we report which features go with being called non-coding.

    pixi run python noncoding_signatures_full.py
"""

import gzip
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from common import HERE, NONCODING_SETS, coding_direction, coding_score, load_full_set
from noncoding_signatures import simple_features

FULL_RESULTS = HERE / "results" / "full"
RMSK = HERE / "data" / "full" / "raw" / "rmsk.txt.gz"
FAMILY_CLASSES = {"Alu": "Alu", "L1": "L1", "L2": "L2", "MIR": "MIR"}
CLASS_NAMES = {"LTR": "LTR", "DNA": "DNA transposon", "Simple_repeat": "simple repeat", "Low_complexity": "low complexity"}
REPEAT_CLASSES = ["Alu", "L1", "MIR", "L2", "LTR", "DNA transposon", "simple repeat", "low complexity", "other repeat"]
SIMPLE = ["gc", "cpg_obs_exp", "longest_homopolymer", "tandem_repeat_score", "trinucleotide_entropy"]


def load_rmsk():
    """chrom (Ensembl naming) -> (starts, ends, class index), sorted by start; coordinates 0-based half-open."""
    df = pd.read_csv(gzip.open(RMSK), sep="\t", header=None, usecols=[5, 6, 7, 11, 12],
                     names=["chrom", "start", "end", "rep_class", "rep_family"])
    df = df[df.chrom.str.fullmatch(r"chr(\d+|X|Y)")]
    cls = df.rep_family.map(FAMILY_CLASSES).fillna(df.rep_class.map(CLASS_NAMES)).fillna("other repeat")
    df = df.assign(chrom=df.chrom.str[3:], cls=cls.map(REPEAT_CLASSES.index)).sort_values(["chrom", "start"])
    return {c: (g.start.to_numpy(), g.end.to_numpy(), g.cls.to_numpy()) for c, g in df.groupby("chrom")}


def repeat_coverage(chrom, intervals):
    """Fraction of the sequence's bases covered by each repeat class."""
    starts, ends, cls = RM[chrom]
    max_len = MAX_LEN[chrom]
    covered, total = np.zeros(len(REPEAT_CLASSES)), 0
    for iv in intervals.split(";"):
        s, e = map(int, iv.split("-"))
        s0, e0 = s - 1, e  # 0-based half-open
        lo, hi = np.searchsorted(starts, s0 - max_len), np.searchsorted(starts, e0)
        mask = np.zeros((len(REPEAT_CLASSES), e0 - s0), bool)
        for a, b, c in zip(starts[lo:hi], ends[lo:hi], cls[lo:hi]):
            if b > s0:
                mask[c, max(a, s0) - s0 : min(b, e0) - s0] = True
        covered += mask.sum(1)
        total += e0 - s0
    return dict(zip(REPEAT_CLASSES, covered / total))


def features(row):
    return {**repeat_coverage(row["chrom"], row["intervals"]), **simple_features(row["seq"])}


def set_features(name):
    path = FULL_RESULTS / f"noncoding_features_{name}.csv"
    if not path.exists():
        _, test = load_full_set(name)
        test["chrom"] = test.chrom.astype(str)
        with Pool(32) as pool:
            pd.DataFrame(pool.map(features, test.to_dict("records"), chunksize=20)).to_csv(path, index=False)
    return pd.read_csv(path)


def heatmap(by_class, counts, title, path):
    heat = by_class.pivot(index="model", columns="dominant_class", values="frac_called_noncoding")[counts.index]
    heat.columns = [f"{c}\n(n={counts[c]})" for c in heat.columns]
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(heat) + 2))
    im = ax.imshow(heat.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns, fontsize=8)
    ax.set_yticks(range(heat.shape[0]), heat.index, fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if heat.values[i, j] > 0.6 else "white")
    fig.colorbar(im, label="Fraction called non-coding")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    RM = load_rmsk()
    MAX_LEN = {c: int((e - s).max()) for c, (s, e, _) in RM.items()}
    model_dirs = sorted((FULL_RESULTS / "embeddings").iterdir())

    for name in NONCODING_SETS:
        feats = set_features(name)
        # Dominant repeat class: the class covering the most bases if it covers at least 10%, else "none".
        cov = feats[REPEAT_CLASSES]
        feats["dominant_class"] = np.where(cov.max(1) >= 0.1, cov.idxmax(1), "none")
        print(f"=== cds_vs_{name}\n" + feats.dominant_class.value_counts().to_string())
        design = feats[REPEAT_CLASSES + SIMPLE]

        by_class, correlations, explained = [], [], []
        for model_dir in model_dirs:
            if not ((model_dir / "coding.npz").exists() and (model_dir / f"{name}.npz").exists()):
                continue
            coding, noncoding = np.load(model_dir / "coding.npz"), np.load(model_dir / f"{name}.npz")
            X_train = np.vstack([coding["X_train"], noncoding["X_train"]])
            y_train = np.r_[np.zeros(len(coding["X_train"])), np.ones(len(noncoding["X_train"]))]
            score = coding_score(coding_direction(X_train, y_train), noncoding["X_test"])
            for cls, group in feats.groupby("dominant_class"):
                by_class.append({"model": model_dir.name, "dominant_class": cls, "n": len(group),
                                 "frac_called_noncoding": (score[group.index] < 0).mean(),
                                 "mean_coding_score": score[group.index].mean()})
            for f in REPEAT_CLASSES + SIMPLE:
                correlations.append({"model": model_dir.name, "feature": f,
                                     "spearman_vs_coding_score": spearmanr(feats[f], score)[0]})
            r2 = cross_val_score(LinearRegression(), design, score, cv=5, scoring="r2").mean()
            explained.append({"model": model_dir.name, "r2_features_to_coding_score": r2})
            print(f"{model_dir.name}: features explain R^2 = {r2:.3f}")

        out = FULL_RESULTS / f"cds_vs_{name}"
        out.mkdir(exist_ok=True)
        by_class, correlations = pd.DataFrame(by_class), pd.DataFrame(correlations)
        by_class.to_csv(out / "noncoding_by_repeat_class.csv", index=False)
        correlations.to_csv(out / "noncoding_feature_correlations.csv", index=False)
        pd.DataFrame(explained).to_csv(out / "noncoding_features_explained.csv", index=False)
        heatmap(by_class, feats.dominant_class.value_counts(),
                f"CDS vs {name}: which non-coding sequences are recognised as non-coding?", out / "noncoding_by_repeat_class.png")
        print(correlations.pivot(index="model", columns="feature", values="spearman_vs_coding_score").round(2).to_string())
