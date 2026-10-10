"""Experiment C: does the gLM know about codons?

In-silico mutagenesis: every position of a sequence is mutated to each of the 3 other bases.
Effect of a mutation = change in coding score (logistic-regression probe on embeddings).
We average effects by codon position (1/2/3) and by mutation class (synonymous/missense/nonsense).
If the model has learned codon structure, coding sequences should show:
  - smaller effects at codon position 3 (wobble), and
  - nonsense (new stop codon) mutations pushing the score towards non-coding.
Non-coding sequences, read in frame 0, are the control and should show no such pattern.
"""

import random

import matplotlib.pyplot as plt
import pandas as pd
from Bio.Data.CodonTable import standard_dna_table
from tqdm import tqdm

from common import FULL, MODEL_SHORT, coding_direction, coding_score, embed, get_datasets

# Full-length genes are long, so there we mutate a random subset of codons in fewer sequences.
N_SEQS = 100 if FULL else 300
N_CODONS = 60 if FULL else None  # None = every codon
random.seed(0)
CODE = standard_dna_table.forward_table  # codon -> amino acid (stops are absent)


def mutation_class(old_codon, new_codon):
    if new_codon not in CODE:
        return "nonsense" if old_codon in CODE else "stop-to-stop"
    if old_codon not in CODE:
        return "stop-loss"
    return "synonymous" if CODE[old_codon] == CODE[new_codon] else "missense"


def mutagenesis(seq, frame, probes):
    """One row per (single-base mutation inside a complete codon, probe); probes: dataset name -> fitted probe."""
    n_codons = (len(seq) - frame) // 3
    codons = range(n_codons) if N_CODONS is None else sorted(random.sample(range(n_codons), min(N_CODONS, n_codons)))
    variants, info = [], []
    for c in codons:
        codon_start = frame + 3 * c
        old_codon = seq[codon_start : codon_start + 3]
        for p in range(codon_start, codon_start + 3):
            for base in "ACGT":
                if base == seq[p]:
                    continue
                mutated = seq[:p] + base + seq[p + 1 :]
                variants.append(mutated)
                info.append((p - codon_start + 1, mutation_class(old_codon, mutated[codon_start : codon_start + 3])))
    X_seq, X_var = embed([seq]), embed(variants, batch_size=len(variants))
    rows = []
    for name, clf in probes.items():
        deltas = coding_score(clf, X_var) - coding_score(clf, X_seq)[0]
        rows += [{"dataset": name, "codon_pos": pos, "mut_class": cls, "delta": d} for (pos, cls), d in zip(info, deltas)]
    return rows


def summarise(df, ds):
    out = ds["out_dir"]

    by_pos = df.groupby(["group", "codon_pos"]).agg(mean_abs_delta=("abs_delta", "mean"), n=("delta", "size")).reset_index()
    by_class = df.groupby(["group", "mut_class"]).agg(
        mean_delta=("delta", "mean"), mean_abs_delta=("abs_delta", "mean"), n=("delta", "size")
    ).reset_index()
    by_pos_class = df.groupby(["group", "codon_pos", "mut_class"]).agg(
        mean_abs_delta=("abs_delta", "mean"), n=("delta", "size")
    ).reset_index()
    by_pos.to_csv(out / "codon_position.csv", index=False)
    by_class.to_csv(out / "mutation_class.csv", index=False)
    by_pos_class.to_csv(out / "codon_position_by_class.csv", index=False)
    print(by_pos.to_string(index=False), by_class.to_string(index=False), by_pos_class.to_string(index=False), sep="\n\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    by_pos.pivot(index="codon_pos", columns="group", values="mean_abs_delta").plot.bar(ax=ax1, rot=0)
    ax1.set(xlabel="Codon position", ylabel="Mean |Δ coding score|", title="Sensitivity by codon position")
    ax1.legend(loc="lower right")
    classes = ["synonymous", "missense", "nonsense"]
    by_class.pivot(index="mut_class", columns="group", values="mean_delta").reindex(classes).plot.bar(ax=ax2, rot=0)
    ax2.axhline(0, color="grey", lw=1)
    ax2.set(xlabel="Mutation class", ylabel="Mean Δ coding score (signed)", title="Direction of shift by mutation class")
    fig.suptitle(f"{MODEL_SHORT} ({ds['name']}) in-silico mutagenesis")
    fig.tight_layout()
    fig.savefig(out / "codon_position.png", dpi=200)


if __name__ == "__main__":
    datasets = get_datasets()
    probes = {ds["name"]: coding_direction(ds["X_train"], ds["y_train"]) for ds in datasets}
    # Variants are embedded once and scored by every probe. All full-length datasets share the coding set; the
    # control group is the first dataset's non-coding set (GB non-coding, or intergenic), read in frame 0.
    first = list(zip(datasets[0]["test_seqs"], datasets[0]["test_frames"], datasets[0]["y_test"]))
    coding = [(s, f) for s, f, y in first if y == 0 and f is not None]
    noncoding = [(s, 0) for s, f, y in first if y == 1]
    groups = {"coding": random.sample(coding, N_SEQS), "non-coding (frame 0)": random.sample(noncoding, N_SEQS)}

    rows = []
    for group, seqs in groups.items():
        for s, f in tqdm(seqs, desc=group):
            rows += [dict(r, group=group) for r in mutagenesis(s, f, probes)]
    all_rows = pd.DataFrame(rows)
    all_rows["abs_delta"] = all_rows.delta.abs()
    for ds in datasets:
        summarise(all_rows[all_rows.dataset == ds["name"]], ds)
