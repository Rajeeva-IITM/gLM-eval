# Embedding Patterns: why coding and non-coding embeddings differ

Three experiments on frozen gLM embeddings (GROVER, seven HyenaDNA sizes, DNABERT-2, four Nucleotide Transformer v2 sizes, Evo and Evo 2) using Genomic Benchmarks `coding_vs_intergenomic_seqs` (200 bp). They ask what property of the DNA the frozen embeddings actually encode.

| Script | Question |
|---|---|
| `kmer_baseline.py` | Do plain k-mer counts separate the classes as well as the embeddings? What is left after regressing the k-mer signal out of the embeddings? |
| `perturb.py` | Which sequence edit (shuffles, codon shuffle, synonymous swap, in-frame vs out-of-frame stops, reverse complement) moves coding sequences over to non-coding? |
| `codon_position.py` | In-silico mutagenesis: is the model less sensitive at codon position 3 (wobble) and to synonymous mutations? |

`common.py` holds the shared helpers: data loading, model embedding (cached to `results/<model>/embeddings.npz`), the logistic-regression "coding score", and reading-frame detection.

## Run
From the repository root, `pixi install` once, then:
```bash
cd "Model Evaluation/Coding-Noncoding/Embedding-Patterns"
pixi run python kmer_baseline.py   # builds the embedding cache on its first run
pixi run python perturb.py
pixi run python codon_position.py
```
Outputs (CSV and PNG files) are written to `results/<model>/`. GROVER is the default model. Choose another one with `GLM_MODEL`:
```bash
GLM_MODEL=LongSafari/hyenadna-large-1m-seqlen-hf pixi run python kmer_baseline.py
```

Some models need their own pixi environment:

| Models | Environment | `GLM_MODEL` example |
|---|---|---|
| GROVER, HyenaDNA | `default` | `LongSafari/hyenadna-large-1m-seqlen-hf` |
| DNABERT-2, Nucleotide Transformer v2 | `legacy` (transformers 4.x) | `zhihan1996/DNABERT-2-117M` |
| Evo | `evo` | `evo-1-8k-base` |
| Evo 2 (7B only on non-Hopper GPUs) | `evo2` | `evo2_7b` |

Example: `GLM_MODEL=evo2_7b pixi run -e evo2 python kmer_baseline.py`

`pixi run python compare.py` collects every model's results into `results/summary.csv`.

Note: Genomic Benchmarks coding windows are not aligned to the reading frame. A frame is inferred only when exactly one forward frame has no stop codon. Sequences without a unique frame are dropped from `perturb.py` and `codon_position.py`.
