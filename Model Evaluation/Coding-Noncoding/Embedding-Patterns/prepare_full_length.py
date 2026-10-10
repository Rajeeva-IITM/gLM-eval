"""Builds the full-length human sequence sets (Ensembl GRCh38 release 116).

Sets (300 bp - 10 kb, A/C/G/T only, one sequence per gene; 10,000 each):
  coding      CDS of each protein-coding gene's Ensembl_canonical transcript (ATG ... stop, frame 0)
  lncrna      canonical transcript of each lncRNA gene (spliced)
  intron      one random intron of each canonical protein-coding transcript (sense strand)
  intergenic  random genome segments >= 10 kb from any annotated gene
Non-coding sets are sampled to match the coding length distribution.
Train/test split by chromosome: chr2, chr4, chr6, chr8 are test.

Run once (needs the files in data/full/raw/, see RAW_FILES):
    pixi run python prepare_full_length.py
"""

import gzip
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import ks_2samp

HERE = Path(__file__).parent
RAW = HERE / "data" / "full" / "raw"
OUT = HERE / "data" / "full"
ENSEMBL = "https://ftp.ensembl.org/pub/release-116"
RAW_FILES = {  # downloaded into data/full/raw/ with curl -O
    "cds": f"{ENSEMBL}/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz",
    "ncrna": f"{ENSEMBL}/fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz",
    "gtf": f"{ENSEMBL}/gtf/homo_sapiens/Homo_sapiens.GRCh38.116.gtf.gz",
    "genome": f"{ENSEMBL}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "rmsk": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz",
}
CHROMS = [str(c) for c in range(1, 23)] + ["X", "Y"]
TEST_CHROMS = {"2", "4", "6", "8"}
MIN_LEN, MAX_LEN, N_PER_SET = 300, 10_000, 10_000
INTERGENIC_FLANK = 10_000
STOPS = {"TAA", "TAG", "TGA"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
random.seed(0)


def raw(key):
    return RAW / RAW_FILES[key].split("/")[-1]


def attr(field, key):
    m = re.search(f'{key} "([^"]+)"', field)
    return m.group(1) if m else None


def parse_gtf():
    """Gene spans (all biotypes) plus exon/CDS intervals of canonical protein-coding and lncRNA transcripts."""
    genes, canonical, exons, cds = [], {}, {}, {}
    with gzip.open(raw("gtf"), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = line.rstrip("\n").split("\t")
            if chrom not in CHROMS:
                continue
            start, end = int(start), int(end)
            if feature == "gene":
                genes.append((chrom, start, end))
                continue
            if 'tag "Ensembl_canonical"' not in attrs:
                continue
            biotype = attr(attrs, "transcript_biotype")
            if biotype not in ("protein_coding", "lncRNA"):
                continue
            tid = attr(attrs, "transcript_id")
            if feature == "transcript":
                canonical[tid] = {"gene_id": attr(attrs, "gene_id"), "chrom": chrom, "strand": strand, "biotype": biotype}
            elif feature == "exon":
                exons.setdefault(tid, []).append((start, end))
            elif feature in ("CDS", "stop_codon"):
                cds.setdefault(tid, []).append((start, end))
    return genes, canonical, exons, cds


def fmt(intervals):
    return ";".join(f"{s}-{e}" for s, e in sorted(intervals))


def read_fasta_by_transcript(path):
    """Transcript id (no version) -> sequence."""
    return {rec.id.split(".")[0]: str(rec.seq).upper() for rec in SeqIO.parse(gzip.open(path, "rt"), "fasta")}


def valid_length(seq):
    return MIN_LEN <= len(seq) <= MAX_LEN and set(seq) <= set("ACGT")


def length_matched(candidates, target_lengths, n):
    """Sample candidates so their lengths follow target_lengths (20 log-spaced bins)."""
    bins = np.logspace(np.log10(MIN_LEN), np.log10(MAX_LEN + 1), 21)
    target = np.histogram(target_lengths, bins)[0] / len(target_lengths) * n
    cand_bin = np.digitize([len(c["seq"]) for c in candidates], bins) - 1
    chosen = []
    for b in range(20):
        pool = [c for c, cb in zip(candidates, cand_bin) if cb == b]
        chosen += random.sample(pool, min(len(pool), round(target[b])))
    return chosen


def intergenic_regions(genes):
    """Per chromosome: sorted (start, end) regions at least INTERGENIC_FLANK from any gene."""
    regions = {}
    for chrom in CHROMS:
        spans = sorted((max(1, s - INTERGENIC_FLANK), e + INTERGENIC_FLANK) for c, s, e in genes if c == chrom)
        merged = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        regions[chrom] = [(a[1] + 1, b[0] - 1) for a, b in zip(merged, merged[1:]) if b[0] - a[1] > MIN_LEN]
    return regions


def sample_intergenic(regions, genome, lengths):
    flat = [(c, s, e) for c, rs in regions.items() for s, e in rs]
    out = []
    for i, length in enumerate(lengths):
        while True:
            chrom, s, e = random.choices(flat, weights=[e - s for _, s, e in flat])[0]
            if e - s + 1 < length:
                continue
            start = random.randint(s, e - length + 1)
            seq = genome[chrom][start - 1 : start - 1 + length]
            if set(seq) <= set("ACGT"):
                out.append({"seq": seq, "gene_id": f"intergenic_{i}", "chrom": chrom, "strand": "+",
                            "intervals": fmt([(start, start + length - 1)])})
                break
    return out


if __name__ == "__main__":
    print("Parsing GTF...")
    genes, canonical, exons, cds_iv = parse_gtf()

    print("Coding set...")
    cds_seqs = read_fasta_by_transcript(raw("cds"))
    coding = []
    for tid, t in canonical.items():
        seq = cds_seqs.get(tid)
        if t["biotype"] != "protein_coding" or seq is None or not valid_length(seq) or len(seq) % 3:
            continue
        codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
        if codons[0] != "ATG" or codons[-1] not in STOPS or any(c in STOPS for c in codons[:-1]):
            continue
        coding.append({"seq": seq, "gene_id": t["gene_id"], "chrom": t["chrom"], "strand": t["strand"],
                       "intervals": fmt(cds_iv.get(tid, []))})
    coding = random.sample(coding, min(N_PER_SET, len(coding)))
    coding_lengths = [len(c["seq"]) for c in coding]

    print("lncRNA set...")
    nc_seqs = read_fasta_by_transcript(raw("ncrna"))
    lnc = [{"seq": nc_seqs[tid], "gene_id": t["gene_id"], "chrom": t["chrom"], "strand": t["strand"],
            "intervals": fmt(exons[tid])}
           for tid, t in canonical.items() if t["biotype"] == "lncRNA" and tid in nc_seqs and valid_length(nc_seqs[tid])]
    lnc = length_matched(lnc, coding_lengths, N_PER_SET)

    print("Loading genome...")
    genome = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(gzip.open(raw("genome"), "rt"), "fasta") if rec.id in CHROMS}

    print("Intron set...")
    introns = []
    for tid, t in canonical.items():
        if t["biotype"] != "protein_coding" or len(exons.get(tid, [])) < 2:
            continue
        ex = sorted(exons[tid])
        gaps = [(a[1] + 1, b[0] - 1) for a, b in zip(ex, ex[1:]) if MIN_LEN <= b[0] - a[1] - 1 <= MAX_LEN]
        random.shuffle(gaps)
        for s, e in gaps:
            seq = genome[t["chrom"]][s - 1 : e]
            if t["strand"] == "-":
                seq = seq.translate(COMPLEMENT)[::-1]
            if valid_length(seq):
                introns.append({"seq": seq, "gene_id": t["gene_id"], "chrom": t["chrom"], "strand": t["strand"],
                                "intervals": fmt([(s, e)])})
                break
    introns = length_matched(introns, coding_lengths, N_PER_SET)

    print("Intergenic set...")
    regions = intergenic_regions(genes)
    intergenic = sample_intergenic(regions, genome, random.choices(coding_lengths, k=N_PER_SET))

    # Checks
    for c in coding:
        assert c["seq"].startswith("ATG") and not any(c["seq"][i : i + 3] in STOPS for i in range(0, len(c["seq"]) - 3, 3))
    genes_by_chrom = {ch: [(s, e) for c, s, e in genes if c == ch] for ch in CHROMS}
    for r in intergenic[:2000]:  # spot check: no gene overlaps the segment
        s, e = map(int, r["intervals"].split("-"))
        assert not any(gs <= e and ge >= s for gs, ge in genes_by_chrom[r["chrom"]])

    for name, rows in [("coding", coding), ("lncrna", lnc), ("intron", introns), ("intergenic", intergenic)]:
        df = pd.DataFrame(rows)
        df["split"] = np.where(df.chrom.isin(TEST_CHROMS), "test", "train")
        df.to_csv(OUT / f"{name}.csv.gz", index=False)
        ks = ks_2samp(df.seq.str.len(), coding_lengths)
        print(f"{name:11s} n={len(df):6d}  test={int((df.split == 'test').sum()):5d}  "
              f"median length={int(df.seq.str.len().median()):5d}  KS vs coding: D={ks.statistic:.3f}")
