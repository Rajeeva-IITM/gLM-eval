"""Experiment B: which sequence edit makes GROVER stop seeing a coding sequence as coding?

Each edit keeps some property and destroys others. Every edit keeps the length at 200 bp.
We re-embed the edited sequences and measure how far they move along the coding direction
(a logistic-regression probe trained on the original embeddings).
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Data.CodonTable import standard_dna_table

from common import RESULTS, coding_direction, coding_score, embed, find_frame, get_embeddings

N_SEQS = 2000
random.seed(0)

SYNONYMS = {}
for codon, aa in standard_dna_table.forward_table.items():
    SYNONYMS.setdefault(aa, []).append(codon)
CODON_TO_SYNONYMS = {c: SYNONYMS[aa] for c, aa in standard_dna_table.forward_table.items()}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def split_codons(seq, frame):
    n = (len(seq) - frame) // 3
    return seq[:frame], [seq[frame + 3 * i : frame + 3 * i + 3] for i in range(n)], seq[frame + 3 * n :]


def nucleotide_shuffle(seq, frame):
    return "".join(random.sample(seq, len(seq)))


def dinucleotide_markov(seq, frame):
    """Resample from a first-order Markov chain fitted to the sequence: keeps dinucleotide frequencies (approximately)."""
    nxt = {}
    for a, b in zip(seq, seq[1:]):
        nxt.setdefault(a, []).append(b)
    out = [seq[0]]
    while len(out) < len(seq):
        out.append(random.choice(nxt.get(out[-1], list(seq))))
    return "".join(out)


def codon_shuffle(seq, frame):
    head, codons, tail = split_codons(seq, frame)
    return head + "".join(random.sample(codons, len(codons))) + tail


def synonymous_swap(seq, frame):
    head, codons, tail = split_codons(seq, frame)
    return head + "".join(random.choice(CODON_TO_SYNONYMS.get(c, [c])) for c in codons) + tail


def insert_stops(seq, frame, offset):
    """Overwrite 3 random codon slots with stop codons; offset 0 = in frame, offset 1 = same edit shifted out of frame."""
    s = list(seq)
    n = (len(seq) - frame - offset) // 3
    for i in random.sample(range(n), 3):
        start = frame + offset + 3 * i
        s[start : start + 3] = random.choice(["TAA", "TAG", "TGA"])
    return "".join(s)


def reverse_complement(seq, frame):
    return seq.translate(COMPLEMENT)[::-1]


EDITS = {
    "original": lambda s, f: s,
    "nucleotide shuffle\n(keeps GC)": nucleotide_shuffle,
    "dinucleotide Markov\n(keeps 2-mers)": dinucleotide_markov,
    "codon shuffle\n(keeps codon usage)": codon_shuffle,
    "synonymous swap\n(keeps protein)": synonymous_swap,
    "3 stops in frame": lambda s, f: insert_stops(s, f, 0),
    "3 stops out of frame\n(control)": lambda s, f: insert_stops(s, f, 1),
    "reverse complement": reverse_complement,
}

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, test_seqs = get_embeddings()
    clf = coding_direction(X_train, y_train)

    coding = [(s, find_frame(s)) for s, y in zip(test_seqs, y_test) if y == 0]
    resolved = [(s, f) for s, f in coding if f is not None]
    print(f"Coding test seqs with exactly one open frame: {len(resolved)}/{len(coding)}")
    groups = {
        "coding": random.sample(resolved, N_SEQS),
        "non-coding (frame 0)": [(s, 0) for s in random.sample([s for s, y in zip(test_seqs, y_test) if y == 1], N_SEQS)],
    }

    rows = []
    for group, seqs in groups.items():
        for edit, fn in EDITS.items():
            score = coding_score(clf, embed([fn(s, f) for s, f in seqs]))
            rows.append({
                "group": group,
                "edit": edit.replace("\n", " "),
                "mean_coding_score": score.mean(),
                "frac_called_coding": (score > 0).mean(),
            })
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "perturbation.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(EDITS))
    for i, group in enumerate(groups):
        ax.bar(x + (i - 0.5) * 0.4, df[df.group == group].frac_called_coding, width=0.4, label=group)
    ax.set_xticks(x, EDITS.keys(), fontsize=8)
    ax.set_ylabel("Fraction classified as coding")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.legend()
    ax.set_title("GROVER: effect of sequence edits on the coding/non-coding probe")
    fig.tight_layout()
    fig.savefig(RESULTS / "perturbation.png", dpi=200)
