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

from common import MODEL_SHORT, RESULTS, coding_direction, coding_score, embed, find_frame, get_embeddings

N_SEQS = 300
random.seed(0)
CODE = standard_dna_table.forward_table  # codon -> amino acid (stops are absent)


def mutation_class(old_codon, new_codon):
    if new_codon not in CODE:
        return "nonsense" if old_codon in CODE else "stop-to-stop"
    if old_codon not in CODE:
        return "stop-loss"
    return "synonymous" if CODE[old_codon] == CODE[new_codon] else "missense"


def mutagenesis(seq, frame, clf):
    """Returns one row per single-base mutation inside complete codons."""
    n_codons = (len(seq) - frame) // 3
    variants, info = [], []
    for p in range(frame, frame + 3 * n_codons):
        codon_start = p - (p - frame) % 3
        old_codon = seq[codon_start : codon_start + 3]
        for base in "ACGT":
            if base == seq[p]:
                continue
            mutated = seq[:p] + base + seq[p + 1 :]
            variants.append(mutated)
            info.append(((p - frame) % 3 + 1, mutation_class(old_codon, mutated[codon_start : codon_start + 3])))
    base_score = coding_score(clf, embed([seq]))[0]
    deltas = coding_score(clf, embed(variants, batch_size=len(variants))) - base_score
    return [{"codon_pos": pos, "mut_class": cls, "delta": d} for (pos, cls), d in zip(info, deltas)]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, test_seqs = get_embeddings()
    clf = coding_direction(X_train, y_train)

    coding = [(s, f) for s, y in zip(test_seqs, y_test) if y == 0 and (f := find_frame(s)) is not None]
    noncoding = [(s, 0) for s, y in zip(test_seqs, y_test) if y == 1]
    groups = {"coding": random.sample(coding, N_SEQS), "non-coding (frame 0)": random.sample(noncoding, N_SEQS)}

    rows = []
    for group, seqs in groups.items():
        for s, f in tqdm(seqs, desc=group):
            rows += [dict(r, group=group) for r in mutagenesis(s, f, clf)]
    df = pd.DataFrame(rows)
    df["abs_delta"] = df.delta.abs()

    by_pos = df.groupby(["group", "codon_pos"]).agg(mean_abs_delta=("abs_delta", "mean"), n=("delta", "size")).reset_index()
    by_class = df.groupby(["group", "mut_class"]).agg(
        mean_delta=("delta", "mean"), mean_abs_delta=("abs_delta", "mean"), n=("delta", "size")
    ).reset_index()
    by_pos_class = df.groupby(["group", "codon_pos", "mut_class"]).agg(
        mean_abs_delta=("abs_delta", "mean"), n=("delta", "size")
    ).reset_index()
    by_pos.to_csv(RESULTS / "codon_position.csv", index=False)
    by_class.to_csv(RESULTS / "mutation_class.csv", index=False)
    by_pos_class.to_csv(RESULTS / "codon_position_by_class.csv", index=False)
    print(by_pos.to_string(index=False), by_class.to_string(index=False), by_pos_class.to_string(index=False), sep="\n\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    by_pos.pivot(index="codon_pos", columns="group", values="mean_abs_delta").plot.bar(ax=ax1, rot=0)
    ax1.set(xlabel="Codon position", ylabel="Mean |Δ coding score|", title="Sensitivity by codon position")
    ax1.legend(loc="lower right")
    classes = ["synonymous", "missense", "nonsense"]
    by_class.pivot(index="mut_class", columns="group", values="mean_delta").reindex(classes).plot.bar(ax=ax2, rot=0)
    ax2.axhline(0, color="grey", lw=1)
    ax2.set(xlabel="Mutation class", ylabel="Mean Δ coding score (signed)", title="Direction of shift by mutation class")
    fig.suptitle(f"{MODEL_SHORT} in-silico mutagenesis")
    fig.tight_layout()
    fig.savefig(RESULTS / "codon_position.png", dpi=200)
