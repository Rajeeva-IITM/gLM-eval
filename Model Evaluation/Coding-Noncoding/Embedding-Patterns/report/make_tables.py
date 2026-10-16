"""Builds the tables and figures for report.typ from the result CSVs .

    pixi run python report/make_tables.py
    typst compile report/report.typ
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
FULL_DATA = HERE.parent / "data" / "full"
DATA, FIGS = HERE / "data", HERE / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

DATASETS = {"gb": RESULTS, "intergenic": RESULTS / "full" / "cds_vs_intergenic",
            "lncrna": RESULTS / "full" / "cds_vs_lncrna", "intron": RESULTS / "full" / "cds_vs_intron"}
TITLES = {"gb": "200 bp (Genomic Benchmarks)", "intergenic": "Full length: CDS vs intergenic",
          "lncrna": "Full length: CDS vs lncRNA", "intron": "Full length: CDS vs intron"}
# Result directory name -> short display name, in display order.
MODELS = {
    "GROVER": "GROVER",
    "DNABERT-2-117M": "DNABERT-2",
    "nucleotide-transformer-v2-50m-multi-species": "NT-v2 50M",
    "nucleotide-transformer-v2-100m-multi-species": "NT-v2 100M",
    "nucleotide-transformer-v2-250m-multi-species": "NT-v2 250M",
    "nucleotide-transformer-v2-500m-multi-species": "NT-v2 500M",
    "gena-lm-bert-base-t2t": "GENA BERT-base",
    "gena-lm-bert-base-t2t-multi": "GENA BERT-base multi",
    "gena-lm-bert-large-t2t": "GENA BERT-large",
    "gena-lm-bigbird-base-t2t": "GENA BigBird-base",
    "hyenadna-tiny-1k-seqlen-hf": "Hyena tiny-1k d128",
    "hyenadna-tiny-1k-seqlen-d256-hf": "Hyena tiny-1k d256",
    "hyenadna-tiny-16k-seqlen-d128-hf": "Hyena tiny-16k",
    "hyenadna-small-32k-seqlen-hf": "Hyena small-32k",
    "hyenadna-medium-160k-seqlen-hf": "Hyena medium-160k",
    "hyenadna-medium-450k-seqlen-hf": "Hyena medium-450k",
    "hyenadna-large-1m-seqlen-hf": "Hyena large-1m",
    "caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3": "Caduceus-Ph 1k d118",
    "caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3": "Caduceus-Ps 1k d118",
    "caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3": "Caduceus-Ph 1k d256",
    "caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3": "Caduceus-Ps 1k d256",
    "caduceus-ph_seqlen-131k_d_model-256_n_layer-16": "Caduceus-Ph 131k",
    "caduceus-ps_seqlen-131k_d_model-256_n_layer-16": "Caduceus-Ps 131k",
    "evo-1-8k-base": "Evo 1",
    "evo2_7b": "Evo 2",
}
EDITS = ["original", "nucleotide shuffle (keeps GC)", "dinucleotide Markov (keeps 2-mers)",
         "codon shuffle (keeps codon usage)", "synonymous swap (keeps protein)", "stops in frame",
         "stops out of frame (control)", "reverse complement"]
EDIT_SHORT = ["original", "nt shuffle", "dinuc Markov", "codon shuffle", "syn swap", "stops in", "stops out", "rev comp"]
GROUPS = {"coding": "coding", "non-coding (frame 0)": "noncoding"}


def per_model(root, filename):
    """Concatenate results/<dataset>/<model>/<filename> over the report's models, with a 'model' column."""
    frames = [pd.read_csv(root / m / filename).assign(model=short) for m, short in MODELS.items()
              if (root / m / filename).exists()]
    return pd.concat(frames, ignore_index=True)


def ordered(df, index="model"):
    return df.reindex([m for m in MODELS.values() if m in df.index])


def save(df, name, digits=3):
    """CSV for Typst, numbers rounded and formatted as text."""
    df = df.copy()
    for c in df.select_dtypes("float"):
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    df.to_csv(DATA / f"{name}.csv")


def heat(ax, df, title, vmin=0, vmax=1, cmap="viridis", fmt="{:.2f}"):
    im = ax.imshow(df.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=7)
    mid = (vmin + vmax) / 2
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            if pd.notna(v):
                dark = v > mid if cmap == "viridis" else abs(v - mid) < (vmax - vmin) / 4
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6, color="black" if dark else "white")
    ax.set_title(title, fontsize=9)
    return im


# --- Full-length data-set statistics -----------------------------------------------------------------------------
sets = {name: pd.read_csv(FULL_DATA / f"{name}.csv.gz") for name in ["coding", "intergenic", "lncrna", "intron"]}
coding_len = sets["coding"].seq.str.len()
rows = []
for name, df in sets.items():
    length = df.seq.str.len()
    gc = df.seq.map(lambda s: (s.count("G") + s.count("C")) / len(s))
    rows.append({"set": name, "n": len(df), "train": int((df.split == "train").sum()), "test": int((df.split == "test").sum()),
                 "median length": int(length.median()), "length IQR": f"{int(length.quantile(.25))}-{int(length.quantile(.75))}",
                 "mean GC": gc.mean(), "KS D vs coding": ks_2samp(length, coding_len).statistic})
save(pd.DataFrame(rows).set_index("set"), "full_sets")

fig, ax = plt.subplots(figsize=(6, 3))
bins = np.logspace(np.log10(300), np.log10(10_001), 30)
for name, df in sets.items():
    ax.hist(df.seq.str.len(), bins=bins, histtype="step", label=name, lw=1.2)
ax.set(xscale="log", xlabel="Sequence length (bp)", ylabel="Count")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "full_lengths.png", dpi=200)
plt.close(fig)

# --- Per-dataset tables and figures ------------------------------------------------------------------------------
for ds, root in DATASETS.items():
    # Summary (one row per model)
    summary = pd.read_csv(root / "summary.csv")
    summary = summary[summary.model.isin(MODELS)].assign(model=lambda d: d.model.map(MODELS)).set_index("model")
    summary = ordered(summary)
    save(summary, f"{ds}_summary")

    # Experiment A: k-mer baseline
    kmer = per_model(root, "kmer_baseline.csv")
    # Older GB runs only recorded accuracy (equal to balanced accuracy there, since the GB test set is balanced).
    metric = "balanced_accuracy" if ds != "gb" else "accuracy"
    kmer[metric] = kmer.get("balanced_accuracy", kmer.test_accuracy).fillna(kmer.test_accuracy)
    kmer["feature"] = kmer.features.str.extract(r"^(\d-mer)")[0].fillna(
        kmer.features.str.contains("residual").map({True: "residual", False: "embedding"}))
    wide = ordered(kmer.pivot(index="model", columns="feature", values=metric))
    wide = wide[[f"{k}-mer" for k in range(1, 7)] + ["embedding", "residual"]]
    dims = ordered(kmer[kmer.feature == "embedding"].set_index("model").dims)
    # k-mer probes do not depend on the model, so they are one row; the model table holds embedding and residual.
    save(wide.iloc[:1, :6].set_axis(["k-mer probe"]), f"{ds}_kmer_only")
    save(wide[["embedding", "residual"]].assign(dims=dims.astype(str)), f"{ds}_kmer")

    # Experiment B: perturbations
    pert = per_model(root, "perturbation.csv")
    pert["edit"] = pert.edit.str.replace(r"^3 ", "", regex=True)  # GB runs used the label "3 stops ..."
    for group, short in GROUPS.items():
        g = pert[pert.group == group]
        for value, suffix in [("frac_called_coding", ""), ("mean_coding_score", "_score")]:
            w = ordered(g.pivot(index="model", columns="edit", values=value))[EDITS]
            w.columns = EDIT_SHORT
            save(w, f"{ds}_perturb_{short}{suffix}")

    # Experiment C: mutagenesis
    pos = per_model(root, "codon_position.csv")
    pos_w = pos.pivot_table(index="model", columns=["group", "codon_pos"], values="mean_abs_delta")
    pos_tab = pd.DataFrame({f"{GROUPS[g]} pos{p}": pos_w[(g, p)] for g in GROUPS for p in (1, 2, 3)})
    for g in GROUPS:
        pos_tab[f"{GROUPS[g]} ratio"] = pos_w[(g, 3)] / ((pos_w[(g, 1)] + pos_w[(g, 2)]) / 2)
    save(ordered(pos_tab), f"{ds}_codon_pos")
    cls = per_model(root, "mutation_class.csv")
    cls_w = cls.pivot_table(index="model", columns=["group", "mut_class"], values="mean_delta")
    cls_tab = ordered(pd.DataFrame({f"{GROUPS[g]} {c}": cls_w[(g, c)]
                                    for g in GROUPS for c in ("synonymous", "missense", "nonsense")}))
    save(cls_tab, f"{ds}_mut_class")

    # Experiment D: non-coding signatures
    short = {"DNA transposon": "DNA", "simple repeat": "simple", "low complexity": "low cx", "other repeat": "other",
             "gc": "GC", "cpg_obs_exp": "CpG o/e", "longest_homopolymer": "homopol.", "tandem_repeat_score": "tandem",
             "trinucleotide_entropy": "entropy"}
    by_class = pd.read_csv(root / "noncoding_by_repeat_class.csv")
    class_col = "repeat_class" if ds == "gb" else "dominant_class"
    by_class = by_class[by_class.model.isin(MODELS)].assign(model=lambda d: d.model.map(MODELS))
    counts = by_class.groupby(class_col).n.first().sort_values(ascending=False)
    bc = ordered(by_class.pivot(index="model", columns=class_col, values="frac_called_noncoding"))[counts.index]
    save(bc.set_axis([f"{short.get(c, c)} ({counts[c]})" for c in bc.columns], axis=1), f"{ds}_nc_by_class")
    corr = pd.read_csv(root / "noncoding_feature_correlations.csv")
    corr = corr[corr.model.isin(MODELS)].assign(model=lambda d: d.model.map(MODELS))
    features = list(dict.fromkeys(corr.feature))
    cw = ordered(corr.pivot(index="model", columns="feature", values="spearman_vs_coding_score"))[features]
    r2 = pd.read_csv(root / "noncoding_features_explained.csv")
    r2 = ordered(r2[r2.model.isin(MODELS)].assign(model=lambda d: d.model.map(MODELS)).set_index("model"))
    save(cw.rename(columns=short).assign(**{"CV R²": r2.r2_features_to_coding_score}), f"{ds}_nc_corr", digits=2)

    # Figures
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(wide))
    ax.bar(x - 0.2, wide.embedding, 0.4, label="embedding probe")
    ax.bar(x + 0.2, wide.residual, 0.4, label="residual probe (k-mer part removed)")
    for k, ls in [("4-mer", ":"), ("6-mer", "--")]:
        ax.axhline(wide[k].iloc[0], color="grey", ls=ls, lw=1, label=f"{k} frequency probe")
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xticks(x, wide.index, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Balanced accuracy" if metric == "balanced_accuracy" else "Accuracy")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / f"{ds}_kmer.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 1.2 + 0.24 * len(wide)), layout="constrained")
    for ax, (group, short) in zip(axes, GROUPS.items()):
        w = ordered(pert[pert.group == group].pivot(index="model", columns="edit", values="frac_called_coding"))[EDITS]
        w.columns = EDIT_SHORT
        im = heat(ax, w, f"{group} sequences")
    axes[1].set_yticklabels([])
    fig.colorbar(im, ax=axes, label="Fraction called coding", shrink=0.8)
    fig.suptitle(TITLES[ds], fontsize=10)
    fig.savefig(FIGS / f"{ds}_perturb.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    pt = ordered(pos_tab)
    x = np.arange(len(pt))
    axes[0].bar(x - 0.2, pt["coding ratio"], 0.4, label="coding")
    axes[0].bar(x + 0.2, pt["noncoding ratio"], 0.4, label="non-coding control (frame 0)")
    axes[0].axhline(1, color="grey", lw=0.8)
    lo = min(pt["coding ratio"].min(), pt["noncoding ratio"].min())
    axes[0].set_ylim(min(0.8, lo - 0.05), 1.1)
    axes[0].set(ylabel="|Δ| pos 3 / mean |Δ| pos 1-2", title="Codon position 3 relative to positions 1-2")
    for ax in axes:
        ax.set_xticks(x, pt.index, rotation=40, ha="right", fontsize=7)
    for i, c in enumerate(["synonymous", "missense", "nonsense"]):
        axes[1].bar(x + (i - 1) * 0.27, cls_tab[f"coding {c}"], 0.27, label=c)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set(ylabel="Mean signed Δ coding score", title="Coding sequences: shift by mutation class")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.title.set_fontsize(9)
    fig.suptitle(TITLES[ds], fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / f"{ds}_codon.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 1.6 + 0.24 * len(wide)), layout="constrained", gridspec_kw={"width_ratios": [len(bc.columns), len(cw.columns)]})
    bc_lab = bc.copy()
    bc_lab.columns = [f"{c} (n={counts[c]})" for c in bc.columns]
    im1 = heat(axes[0], bc_lab, "Fraction called non-coding, by repeat class")
    fig.colorbar(im1, ax=axes[0], shrink=0.8)
    im2 = heat(axes[1], cw, "Spearman ρ, feature vs coding score", vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)
    fig.suptitle(TITLES[ds], fontsize=10)
    fig.savefig(FIGS / f"{ds}_nc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"{ds}: {len(wide)} models")

# --- Robustness to training-set size (sample_size.py) ------------------------------------------------------------
ss = pd.read_csv(RESULTS / "sample_size_summary.csv")
ss = ss[ss.model.isin(MODELS)].assign(model=lambda d: d.model.map(MODELS))
FAMILY_COLORS = {"GROVER": "tab:blue", "DNABERT-2": "tab:orange", "NT-v2": "tab:green", "GENA": "tab:olive", "Hyena": "tab:red",
                 "Caduceus": "tab:purple", "Evo": "tab:brown"}


def family(model):
    return next(f for f in FAMILY_COLORS if model.startswith(f))


for ds in DATASETS:
    d = ss[ss.dataset == ds].set_index("model")
    at = lambda n, col: d[d.n == n][col]
    tab = pd.DataFrame({
        "all data": at(d.n.max(), "reference_accuracy"),
        "1k": at(1000, "accuracy_mean"),
        "10k": at(10_000, "accuracy_mean"),
        "10k SD": at(10_000, "accuracy_std"),
        "Pearson 1k": at(1000, "pearson_to_reference_mean"),
        "Pearson 10k": at(10_000, "pearson_to_reference_mean"),
        "same call 10k": at(10_000, "same_call_as_reference_mean"),
        "cosine 10k": at(10_000, "cosine_to_reference_mean"),
    })
    save(ordered(tab), f"{ds}_sample_size")

fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex="col")
rows = [("accuracy", "Test accuracy"), ("pearson_to_reference", "Pearson r of scores\nwith reference probe"),
        ("cosine_to_reference", "Cosine of weights\nwith reference probe")]
for j, ds in enumerate(DATASETS):
    for i, (metric, label) in enumerate(rows):
        ax = axes[i, j]
        for model, g in ss[ss.dataset == ds].groupby("model"):
            g = g.sort_values("n")
            m, s = g[f"{metric}_mean"], g[f"{metric}_std"].fillna(0)
            c = FAMILY_COLORS[family(model)]
            ax.plot(g.n, m, color=c, lw=1, alpha=0.8)
            ax.fill_between(g.n, np.clip(m - s, -1, 1), np.clip(m + s, -1, 1), color=c, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.axvline(10_000, color="grey", ls=":", lw=1)
        if i == 0:
            ax.set_title(TITLES[ds], fontsize=9)
        if j == 0:
            ax.set_ylabel(label, fontsize=9)
        if i == 2:
            ax.set_xlabel("Training sequences (balanced)", fontsize=9)
axes[0, 0].legend(handles=[plt.Line2D([], [], color=c, label=f) for f, c in FAMILY_COLORS.items()], fontsize=7)
fig.tight_layout()
fig.savefig(FIGS / "sample_size.png", dpi=200)
plt.close(fig)
