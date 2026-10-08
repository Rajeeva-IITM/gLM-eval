"""Collects the per-model results in results/<model>/ into one summary table (results/summary.csv)."""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).parent / "results"

rows = []
for model_dir in sorted(p for p in RESULTS.iterdir() if (p / "codon_position.csv").exists()):
    kmer = pd.read_csv(model_dir / "kmer_baseline.csv").test_accuracy.tolist()
    pert = pd.read_csv(model_dir / "perturbation.csv").set_index(["group", "edit"]).frac_called_coding
    pos = pd.read_csv(model_dir / "codon_position.csv").set_index(["group", "codon_pos"]).mean_abs_delta
    cls = pd.read_csv(model_dir / "mutation_class.csv").set_index(["group", "mut_class"]).mean_delta
    rows.append({
        "model": model_dir.name,
        "embedding_acc": kmer[-2],
        "6mer_acc": kmer[5],
        "residual_acc": kmer[-1],
        "coding_shuffled_called_coding": pert["coding", "nucleotide shuffle (keeps GC)"],
        "noncoding_shuffled_called_coding": pert["non-coding (frame 0)", "nucleotide shuffle (keeps GC)"],
        "coding_stops_in_frame": pert["coding", "3 stops in frame"],
        "coding_stops_out_of_frame": pert["coding", "3 stops out of frame (control)"],
        "coding_revcomp_called_coding": pert["coding", "reverse complement"],
        # < 1 means codon position 3 matters less than positions 1-2 (wobble awareness)
        "coding_pos3_vs_pos12": pos["coding", 3] / ((pos["coding", 1] + pos["coding", 2]) / 2),
        "coding_nonsense_delta": cls["coding", "nonsense"],
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "summary.csv", index=False)
print(df.round(3).to_string(index=False))
