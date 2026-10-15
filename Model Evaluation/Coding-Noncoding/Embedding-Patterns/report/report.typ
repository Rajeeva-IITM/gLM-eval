// Coding vs non-coding: what genomic language model embeddings encode.
// Build: pixi run python report/make_tables.py && typst compile report/report.typ

#set document(title: "Coding vs non-coding signals in genomic language model embeddings")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { pagebreak(weak: true); it }
#show figure.where(kind: table): set figure.caption(position: top)
#show link: set text(fill: blue)

#let D = "data/"
#let F = "figures/"
#let datasets = (
  ("gb", "200 bp Genomic Benchmarks"),
  ("intergenic", "full length, CDS vs intergenic"),
  ("lncrna", "full length, CDS vs lncRNA"),
  ("intron", "full length, CDS vs intron"),
)

// Table from a CSV written by make_tables.py. The first column is the row label.
#let csvtable(name, caption, size: 7.5pt, header: none) = {
  let rows = csv(D + name + ".csv")
  let head = if header == none { rows.at(0) } else { header }
  let body = rows.slice(1)
  figure(
    kind: table,
    caption: caption,
    text(size: size, table(
      columns: (auto,) + (auto,) * (head.len() - 1),
      align: (x, y) => if x == 0 { left } else { right },
      stroke: none,
      inset: (x: 4pt, y: 2.5pt),
      table.hline(),
      table.header(..head.map(h => strong(h))),
      table.hline(stroke: 0.5pt),
      ..body.flatten(),
      table.hline(),
    )),
  )
}

#align(center)[
  #text(size: 17pt, weight: "bold")[Coding vs non-coding signals in \ genomic language model embeddings]
  #v(0.4em)
  Methods and results #h(1em) · #h(1em) #datetime.today().display("[day] [month repr:long] [year]")
]

#v(1em)
*Scope.* This report describes four experiments, and a training-set size robustness analysis, that probe what frozen embeddings of genomic language models (gLMs)
encode about protein-coding versus non-coding DNA, and lists their results. Each experiment was run on two kinds of
data: the 200 bp Genomic Benchmarks coding vs intergenomic windows, and full-length human sequences (complete coding
sequences against intergenic, lncRNA and intron sequences). Twenty-one models from six families were evaluated.
The report gives methods, rationale and results only. Interpretation and discussion are left out.

#outline(depth: 2, indent: auto)

= Models

All models are used frozen, with no fine-tuning. @tab-models lists them. The _window_ column is the longest
stretch of DNA embedded in one forward pass; longer sequences are split into chunks (@sec-chunk).

#figure(
  kind: table,
  caption: [Models evaluated. Embedding width is the size of the last hidden layer.],
  text(size: 8pt, {
  // Let long model IDs break after "_", "-" and "/".
  show regex("[_/-]"): it => it + "\u{200B}"
  table(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, left, left, right, right),
    stroke: none,
    inset: (x: 4pt, y: 2.5pt),
    table.hline(),
    table.header([*Model*], [*Hugging Face ID*], [*Architecture / tokens*], [*Training genomes*], [*Width*], [*Window (bp)*]),
    table.hline(stroke: 0.5pt),
    [GROVER], [PoetschLab/GROVER], [BERT, BPE], [human], [768], [1,500],
    [DNABERT-2], [zhihan1996/DNABERT-2-117M], [BERT + ALiBi, BPE], [multi-species], [768], [1,500],
    [NT-v2 50M], [InstaDeepAI/nucleotide-transformer-v2-50m-multi-species], [ESM-style, 6-mer], [multi-species], [512], [10,000],
    [NT-v2 100M], […-v2-100m-multi-species], [ESM-style, 6-mer], [multi-species], [512], [10,000],
    [NT-v2 250M], […-v2-250m-multi-species], [ESM-style, 6-mer], [multi-species], [768], [10,000],
    [NT-v2 500M], […-v2-500m-multi-species], [ESM-style, 6-mer], [multi-species], [1024], [10,000],
    [Hyena tiny-1k d128], [LongSafari/hyenadna-tiny-1k-seqlen-hf], [Hyena, single nucleotide], [human], [128], [1,000],
    [Hyena tiny-1k d256], […/hyenadna-tiny-1k-seqlen-d256-hf], [Hyena, single nucleotide], [human], [256], [1,000],
    [Hyena tiny-16k], […/hyenadna-tiny-16k-seqlen-d128-hf], [Hyena, single nucleotide], [human], [128], [10,000],
    [Hyena small-32k], […/hyenadna-small-32k-seqlen-hf], [Hyena, single nucleotide], [human], [256], [10,000],
    [Hyena medium-160k], […/hyenadna-medium-160k-seqlen-hf], [Hyena, single nucleotide], [human], [256], [10,000],
    [Hyena medium-450k], […/hyenadna-medium-450k-seqlen-hf], [Hyena, single nucleotide], [human], [256], [10,000],
    [Hyena large-1m], […/hyenadna-large-1m-seqlen-hf], [Hyena, single nucleotide], [human], [256], [10,000],
    [Caduceus-Ph 1k d118], [kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3], [BiMamba, single nucleotide], [human], [118], [1,000],
    [Caduceus-Ps 1k d118], […/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3], [BiMamba RC-equivariant, single nucleotide], [human], [236], [1,000],
    [Caduceus-Ph 1k d256], […/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3], [BiMamba, single nucleotide], [human], [256], [1,000],
    [Caduceus-Ps 1k d256], […/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3], [BiMamba RC-equivariant, single nucleotide], [human], [512], [1,000],
    [Caduceus-Ph 131k], […/caduceus-ph_seqlen-131k_d_model-256_n_layer-16], [BiMamba, single nucleotide], [human], [256], [10,000],
    [Caduceus-Ps 131k], […/caduceus-ps_seqlen-131k_d_model-256_n_layer-16], [BiMamba RC-equivariant, single nucleotide], [human], [512], [10,000],
    [Evo 1], [togethercomputer/evo-1-8k-base (`evo-1-8k-base`)], [StripedHyena 7B, single nucleotide], [prokaryotes and phages], [4096], [4,000],
    [Evo 2], [arcinstitute/evo2_7b (`evo2_7b`)], [StripedHyena 2 7B, single nucleotide], [all domains of life], [4096], [2,000],
    table.hline(),
  )}),
) <tab-models>

Caduceus-Ph and -Ps are the two Caduceus variants: Ph is trained with reverse-complement data augmentation, while Ps
is reverse-complement equivariant by construction. Its hidden state holds a forward and a reverse-complement half,
so its embedding width is twice the model dimension. The two halves are used as they are, without averaging.

= Common methodology

== Sequence embeddings
For each sequence the last hidden layer of the model is taken and mean-pooled over tokens, using the attention mask
so that padding tokens are excluded. This is the same pooling as in the repository's attribution notebook. The
result is one fixed-length vector per sequence, of the width shown in @tab-models.

Model-specific details, needed for reproduction:
- *GROVER* and the *HyenaDNA* models load with `AutoModel` (transformers ≥ 5.18).
- *DNABERT-2* needs transformers 4.44–4.46. Its config is passed explicitly as a `BertConfig`, and its Triton
  flash-attention kernel is disabled so the PyTorch attention path is used.
- *NT-v2* needs the same older transformers. It loads with `AutoModelForMaskedLM` with `output_hidden_states=True`,
  and the last hidden state is taken.
- *HyenaDNA* and *Caduceus* take no attention mask. Chunks of different lengths are therefore never batched
  together, so no padding enters their hidden states (causal for HyenaDNA, bidirectional for Caduceus). Caduceus
  needs transformers 4.46 with `mamba-ssm` 2.2.5 and `causal-conv1d` 1.5.2 and loads with `AutoModel`.
- *Evo 1* and *Evo 2* load through their own packages (`evo-model`, `evo2`). Embeddings are the output of the
  final normalisation layer (the last hidden layer), captured with a forward hook and mean-pooled. Tokens are single
  characters with no special tokens, and only equal-length chunks are batched, so there is no padding.

== Long sequences: chunk and average <sec-chunk>
Full-length sequences (300 bp to 10 kb) are often longer than a model's context. A sequence of length $L$ is split
into $ceil(L slash W)$ contiguous chunks of near-equal length, where $W$ is the model's window (@tab-models). Each
chunk is embedded, and the sequence embedding is the mean of the chunk embeddings, weighted by chunk length.

The windows were chosen as follows:
- GROVER and DNABERT-2 are limited to 512 tokens. 1.5 kb stays below that even for poorly compressible sequence;
  2 kb reached about 550 tokens.
- The HyenaDNA and Caduceus 1k models use their 1 kb training length.
- Evo 1 (4 kb) and Evo 2 (2 kb) are limited by GPU memory on a shared 24 GB A10, not by context length. Evo 1
  at 8 kb ran out of memory.
- The other models use 10 kb, so a sequence is never split for them.

Every 200 bp window is a single chunk, so in that case the embedding is the plain per-sequence embedding.

== Linear probe and coding score
Every experiment uses the same probe:
- Standardisation followed by L2-regularised logistic regression (scikit-learn `LogisticRegression`, `C = 1`,
  `max_iter = 2000`).
- Trained on the training-set embeddings with labels 0 = coding and 1 = non-coding.
- The _coding score_ of a sequence is the negative of the probe's decision function, so higher means more
  coding-like. A sequence is _called coding_ when its coding score is above 0.

Experiments B–D keep the probe fixed and only change the input sequences, so any change in the coding score comes
from the model's response to the edit.

== Metrics
- 200 bp data: accuracy, because the test set is balanced.
- Full-length data: balanced accuracy (the mean of per-class recall). The chromosome-held-out test sets are
  unbalanced (@tab-sets), and with plain accuracy chance level would differ between datasets (up to 0.585).

== Software
- Analyses are in `Model Evaluation/Coding-Noncoding/Embedding-Patterns/`.
- They run under pixi with five environments: `default` (transformers 5; GROVER, HyenaDNA), `legacy`
  (transformers 4.44–4.46; DNABERT-2, NT-v2), `caduceus`, `evo` and `evo2` (PyPI torch 2.7.1 with prebuilt
  CUDA kernels).
- Hardware: NVIDIA A10 GPUs (24 GB).

= Datasets

== 200 bp: Genomic Benchmarks
- Data set: `katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs`, 200 bp human windows labelled
  coding or intergenomic.
- Train: 75,000 windows (37,500 per class). Test: 25,000 windows (12,500 per class).
- The original split is used.

The reading frame of the coding windows is not given. Experiments B and C need it, so it is inferred: a window is
assigned the frame if exactly one of its three forward frames has no stop codon. This holds for 8,824 of the
12,500 coding test windows (frames 0/1/2: 2,942/2,936/2,946). The other 3,676 windows have no reading frame
assigned and are excluded wherever a frame is needed.

== Full-length human sequences
The 200 bp windows are short (about 66 codons), cut out of genes, and not aligned to the reading frame. Complete
coding sequences remove these limits, and having several kinds of non-coding DNA separates "coding vs
non-coding" from "coding vs one particular background".

All sets come from Ensembl GRCh38 release 116 (`prepare_full_length.py`). Rules common to every set:
- Sequences are 300 bp to 10 kb long and contain only A/C/G/T.
- One sequence per gene.
- Chromosomes 1–22, X and Y only.

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 4pt, y: 3pt),
  table.hline(),
  [*Set*], [*Construction*],
  table.hline(stroke: 0.5pt),
  [coding], [CDS of the `Ensembl_canonical` transcript of each protein-coding gene. It must start with ATG, end with a stop codon, have a length divisible by 3 and no internal in-frame stop codon. Reading frame 0 is known for every sequence.],
  [lncrna], [Canonical transcript of each lncRNA gene (spliced, from the Ensembl ncRNA FASTA).],
  [intron], [One randomly chosen intron per canonical protein-coding transcript, taken from the gaps between GTF exons. It is reverse-complemented on minus-strand genes, so it reads in the sense direction.],
  [intergenic], [Random genome segments at least 10 kb from any annotated gene of any biotype. They are sampled with exactly the lengths drawn from the coding length distribution, and must contain no N.],
  table.hline(),
)

*Length matching.*
- Sequence length strongly affects composition statistics and chunking, so each non-coding set is sampled to follow
  the coding length distribution.
- lncRNA and intron sequences are matched in 20 log-spaced length bins. Intergenic segments copy coding lengths
  directly.
- @tab-sets gives the Kolmogorov–Smirnov statistic against the coding lengths, and @fig-lengths shows the
  distributions.

*Train/test split by chromosome.* Sequences on chromosomes 2, 4, 6 and 8 form the test set; all others form the
training set. A random split would let paralogous genes, or a gene's CDS and its own intron or neighbouring lncRNA,
appear in both training and test sets.

*Datasets.* The three binary datasets `cds_vs_intergenic`, `cds_vs_lncrna` and `cds_vs_intron` share the same
coding set. In each, the probe is trained on the coding training sequences against that non-coding set's training
sequences.

#csvtable("full_sets", [Full-length sequence sets. GC is the mean per-sequence GC fraction.], size: 8.5pt) <tab-sets>

#figure(image(F + "full_lengths.png", width: 75%), caption: [Length distributions of the full-length sets.]) <fig-lengths>

= Experiment A: k-mer baseline and residual probe

== Rationale
Coding and non-coding DNA differ strongly in base composition (GC content, codon-driven k-mer frequencies). A probe
on gLM embeddings may separate the classes only because the embeddings encode composition. Experiment A measures
how far composition alone gets, and whether anything useful remains in the embeddings once composition is removed.

== Method
1. *k-mer probes.*
   - For each $k = 1, dots, 6$, every sequence is represented by the frequencies of its overlapping $k$-mers
     (counts divided by the total, so sequences of different lengths are comparable).
   - A logistic regression (scaled, `max_iter = 2000`) is trained on these frequencies.
   - k-mer probes do not involve a gLM, so they are the same for every model.
2. *Embedding probe.* The standard probe on the gLM embeddings.
3. *Residual probe.*
   - The 1- to 6-mer frequencies are concatenated (5,460 features) and standardised.
   - A multi-output ridge regression ($alpha = 10$) is fitted on the training set to predict the embedding from them.
   - The prediction is subtracted from the embedding for both training and test sequences, and the standard probe
     is trained and tested on these residuals.
   - The residual is the part of the embedding that k-mer frequencies cannot linearly explain. Residual accuracy
     above chance (0.5) means that part still carries class information.

== Results
#for (ds, title) in datasets {
  csvtable(ds + "_kmer_only", [k-mer frequency probes, #title (#if ds == "gb" [accuracy] else [balanced accuracy]).], size: 8.5pt)
  csvtable(ds + "_kmer", [Embedding and residual probes, #title. _dims_ = embedding width.], size: 8.5pt,
    header: ("Model", "Embedding", "Residual", "dims"))
  figure(image(F + ds + "_kmer.png", width: 100%),
    caption: [Embedding and residual probe accuracy per model, #title. Dotted and dashed lines: 4-mer and 6-mer probes.])
}

= Experiment B: perturbation ladder

== Rationale
Experiment A shows _how much_ is composition, but not _which_ structure the models respond to. Experiment B edits
test sequences so that each edit keeps some properties and destroys others. It then measures whether the unchanged
probe still calls them coding. The edits form a ladder, from keeping only base composition to keeping the encoded
protein. Comparing the steps shows which level of structure (composition, dinucleotides, codon usage, codon order,
reading frame, strand) the coding score depends on.

== Edits
All edits keep the sequence length. The reading frame is 0 for full-length CDS and the inferred frame for 200 bp
coding windows. Non-coding sequences are edited as if read in frame 0, as a control.

#table(
  columns: (auto, 1fr, 1fr),
  stroke: none,
  inset: (x: 4pt, y: 3pt),
  table.hline(),
  [*Edit*], [*Keeps*], [*Destroys*],
  table.hline(stroke: 0.5pt),
  [original], [everything], [nothing (baseline)],
  [nucleotide shuffle], [base composition (GC)], [all order],
  [dinucleotide Markov], [approx. dinucleotide frequencies: resampled from a first-order Markov chain fitted to the sequence], [longer-range order, codons],
  [codon shuffle], [the multiset of codons (codon usage) and the frame], [codon order, so the protein sequence],
  [synonymous swap], [the encoded protein: each codon is replaced by a random synonymous codon], [codon usage and the original nucleotide sequence],
  [stops in frame], [everything except a few codon slots, which are overwritten with random stop codons in frame], [the open reading frame],
  [stops out of frame (control)], [the same number of stop codons written at frame offset +1], [same nucleotide damage as the in-frame edit, but the reading frame stays open],
  [reverse complement], [composition up to strand symmetry (GC is unchanged)], [the strand and direction of the coding sequence],
  table.hline(),
)

*Stop-codon density.* The number of inserted stops is $max(3, "round"(3 L slash 200))$, which is three stops per
200 bp in both datasets. The out-of-frame control changes the same number of bases, so any difference between the
two stop edits isolates the reading frame from the damage itself.

*Sampling.* Up to 2,000 test sequences per group, sampled with seed 0:

#table(
  columns: (auto, 1fr, 1fr),
  stroke: none,
  inset: (x: 4pt, y: 3pt),
  table.hline(),
  [*Data*], [*Coding group*], [*Non-coding group*],
  table.hline(stroke: 0.5pt),
  [200 bp], [2,000 of the 8,824 windows with an inferred frame], [2,000 of the 12,500 intergenomic windows],
  [Full length], [all 1,835 coding test sequences], [2,000 intergenic / 2,000 lncRNA / all 1,808 intron test sequences],
  table.hline(),
)

Edited sequences are re-embedded with the same model, chunking and pooling, and scored with the dataset's probe. The
reported value is the fraction called coding (coding score > 0). In the full-length data the edited coding
sequences are embedded once and scored by each of the three probes.

== Results
In the tables: _nt shuffle_ = nucleotide shuffle, _dinuc Markov_ = dinucleotide Markov, _syn swap_ = synonymous
swap, _stops in/out_ = stops in frame / out of frame, _rev comp_ = reverse complement.

#for (ds, title) in datasets {
  csvtable(ds + "_perturb_coding", [Fraction of edited *coding* sequences called coding, #title.])
  csvtable(ds + "_perturb_noncoding", [Fraction of edited *non-coding* sequences (frame 0) called coding, #title.])
  figure(image(F + ds + "_perturb.png", width: 100%), caption: [Perturbation ladder, #title.])
}

Mean coding scores for every edit are in the appendix (Appendix A.1).

= Experiment C: in-silico mutagenesis

== Rationale
If a model represents codons, single-base mutations should affect the coding score depending on where they fall in
the codon and what they do to the encoded protein:
- Under the genetic code, most changes at codon position 3 (the wobble position) are synonymous.
- Changes at positions 1 and 2 usually alter the amino acid.
- A nonsense mutation (a new stop codon) interrupts the reading frame.

Experiment C measures, base by base, how the coding score responds to every possible substitution. Non-coding
sequences mutated the same way (read in frame 0) give the control, where codon position and mutation class have no
biological meaning.

== Method
- For each selected sequence and each selected codon, every base of the codon is changed to each of the three other
  bases (9 variants per codon).
- Each variant is embedded and scored, and the effect is $Delta$ = coding score(variant) − coding score(original).
- Each variant is labelled with its codon position (1, 2 or 3) and its mutation class from the standard genetic
  code:
  - _synonymous_: same amino acid;
  - _missense_: different amino acid;
  - _nonsense_: sense codon to stop codon;
  - _stop-loss_ and _stop-to-stop_: these occur only when the mutated codon was already a stop codon.

#table(
  columns: (auto, 1fr, 1fr),
  stroke: none,
  inset: (x: 4pt, y: 3pt),
  table.hline(),
  [], [*200 bp*], [*Full length*],
  table.hline(stroke: 0.5pt),
  [Coding sequences], [300 windows with an inferred frame], [100 coding test sequences],
  [Control sequences], [300 intergenomic windows (frame 0)], [100 intergenic test sequences (frame 0)],
  [Codons mutated], [every complete codon (66 per window, 594 variants)], [60 random codons per sequence (540 variants)],
  [Probes], [the 200 bp probe], [each of the three full-length probes scores the same variants],
  table.hline(),
)

In the full-length data the same 100 coding and 100 intergenic sequences, and the same variants, are used for all
three datasets; only the probe changes. The intergenic control therefore also serves as the control for the lncRNA
and intron datasets. Because the final codon of each CDS is its stop codon, a small number of full-length coding
variants are stop-loss or stop-to-stop; these are left out of the tables below.

*Summary statistics.*
- _Codon position_: mean $|Delta|$ at each codon position, and the ratio of position 3 to the mean of positions 1
  and 2. A ratio below 1 means wobble-position mutations change the score less.
- _Mutation class_: mean signed $Delta$ per class. A negative value means mutations of that class move sequences
  towards non-coding.

== Results
#for (ds, title) in datasets {
  csvtable(ds + "_codon_pos", [Mean $|Delta|$ coding score by codon position, and the position-3 / position-1-2 ratio, #title.], size: 7pt,
    header: ("Model", "cod. pos1", "cod. pos2", "cod. pos3", "ctrl pos1", "ctrl pos2", "ctrl pos3", "cod. ratio", "ctrl ratio"))
  csvtable(ds + "_mut_class", [Mean signed $Delta$ coding score by mutation class, #title.], size: 7pt,
    header: ("Model", "cod. syn", "cod. mis", "cod. non", "ctrl syn", "ctrl mis", "ctrl non"))
  figure(image(F + ds + "_codon.png", width: 100%),
    caption: [Codon-position ratio (coding vs control) and the signed shift by mutation class for coding sequences, #title.])
}

= Experiment D: non-coding signatures

== Rationale
Experiments A–C ask what makes a sequence look coding. Non-coding DNA, however, is heterogeneous: in the human
genome it is largely transposable elements (Alu, L1, MIR, L2, LTR elements, DNA transposons), simple repeats and
low-complexity sequence, with distinct composition. Experiment D asks which properties of the _non-coding_ test
sequences go with being recognised as non-coding. Each non-coding sequence gets interpretable features, and these
are related to its coding score under each model.

== Features, 200 bp data (`noncoding_signatures.py`)
*Repeat family by alignment.*
- Each non-coding test window (12,500) is locally aligned on both strands to the Dfam consensus sequences of
  representative human repeat families:
  - Alu: AluY, AluSx, AluJb
  - MIR
  - L1: the 3′ ends of L1HS, L1PA7 and L1MB7
  - L2: the 3′ ends of L2a and L2b
  - LTR: THE1B, MLT1A0, MSTA, LTR12C
  - DNA transposons: MER5A, Tigger1
- The 3′ ends are used for LINEs because most genomic LINE copies are 5′-truncated.
- Alignment scoring: match +2, mismatch −3, gap open −5, gap extension −2.

*Significance threshold.* For each family, the threshold is the 99.5th percentile of best scores for 1,000 shuffled
windows. A window is assigned the class of the family it beats by the largest margin. Windows with no significant
hit are labelled _none_, or _simple repeat_ if their tandem-repeat score (below) exceeds 0.6.

== Features, full-length data (`noncoding_signatures_full.py`)
*Repeat coverage.*
- RepeatMasker coverage is taken from the UCSC hg38 `rmsk` table, using each sequence's genomic intervals (exons
  for spliced lncRNAs).
- For each repeat class, the feature is the fraction of the sequence's bases it covers. Classes: Alu, L1, MIR, L2,
  LTR, DNA transposon, simple repeat, low complexity, other repeat.
- The real annotation replaces consensus alignment because full-length sequences contain many repeats of every age.
- A sequence's _dominant class_ is the class with the highest coverage if that coverage is at least 10%; otherwise
  it is _none_.

== Sequence features (both data types)
Computed from the sequence and normalised for length:
- GC fraction.
- CpG observed/expected: $n_"CG" dot L slash (n_C dot n_G)$.
- Longest homopolymer run.
- Tandem-repeat score: the maximum over periods $p = 1..6$ of the fraction of positions equal to the base $p$
  positions later.
- Trinucleotide Shannon entropy (low means low complexity).

== Statistics
For each model and dataset:
- The probe trained on that dataset scores the non-coding test sequences.
- The fraction called non-coding (coding score < 0) is reported per repeat class.
- The Spearman correlation of each feature with the coding score is computed.
- The cross-validated $R^2$ (5-fold, ordinary least squares) is computed for predicting the coding score from all
  features: the continuous features plus one-hot repeat class for 200 bp; the coverages plus continuous features for
  full length.

== Results
Column headers of the repeat-class tables give the number of sequences in each class in parentheses. Abbreviations:
_DNA_ = DNA transposon, _simple_ = simple repeat, _low cx_ = low complexity, _other_ = other repeat, _CpG o/e_ = CpG
observed/expected, _homopol._ = longest homopolymer, _tandem_ = tandem-repeat score, _entropy_ = trinucleotide
entropy. In the full-length correlation tables the repeat-class columns are coverage fractions.

#set page(flipped: true)
#for (ds, title) in datasets {
  csvtable(ds + "_nc_by_class", [Fraction of non-coding test sequences called non-coding, by #if ds == "gb" [best-matching repeat class] else [dominant repeat class], #title.], size: 7.5pt)
  csvtable(ds + "_nc_corr", [Spearman correlation of each feature with the coding score of non-coding test sequences, and 5-fold CV $R^2$ of a linear model from all features to the coding score, #title.], size: 7.5pt)
  figure(image(F + ds + "_nc.png", width: 100%),
    caption: [Fraction called non-coding by repeat class (left) and feature correlations with the coding score (right), #title.])
}
#set page(flipped: false)

= Robustness to training-set size

== Rationale
The full-length sets were capped at 10,000 sequences per class, and every experiment uses a probe trained on all
available training sequences. This section measures how probe accuracy and the probe itself change when the probe
is trained on fewer sequences. If both are already stable well below the sizes used, the results do not depend on
the choice of cap.

== Method
- *Reference probe:* the standard probe trained on the whole training set, as in Experiments A–D (200 bp: 75,000
  windows; full length: 15,575–16,219 sequences, 7,410–8,165 per class).
- *Subset probes:* the same probe trained on balanced random subsets of $n$ training sequences ($n slash 2$ per
  class), for $n$ = 500, 1,000, 2,000, 5,000 and 10,000, plus 20,000 and 50,000 for 200 bp, plus the largest
  balanced subset available (twice the size of the smaller class). Ten subsets are drawn per size (seeds 0–9); a
  subset equal to the whole training set is fitted once.
- Each subset probe is evaluated on the unchanged full test set and compared with the reference probe:
  - *test accuracy* (balanced accuracy for full length);
  - *Pearson correlation of coding scores* with the reference scores over the test set. For two linear probes this
    is the cosine similarity of their directions measured under the covariance of the test embeddings, so it
    compares the probes along directions in which the embeddings actually vary;
  - *Spearman correlation* of the scores, and the *fraction of identical calls* (coding / non-coding);
  - *cosine similarity of the weight vectors* in raw embedding space (logistic-regression coefficients divided by
    the standardisation scale). Embedding dimensions are strongly correlated, so different weight vectors can
    produce nearly identical scores; this metric is reported for completeness.
- Script: `sample_size.py` (cached embeddings only). Full per-run results: `results/sample_size.csv`; mean and SD
  over seeds: `results/sample_size_summary.csv`.

== Results
Table columns: _all data_ = accuracy of the reference probe; _1k_, _10k_ = mean test accuracy of subset probes
trained on 1,000 or 10,000 sequences; _10k SD_ = its standard deviation over the 10 subsets; _Pearson_, _same
call_ and _cosine_ = agreement with the reference probe, averaged over subsets.

#for (ds, title) in datasets {
  csvtable(ds + "_sample_size", [Probe accuracy and agreement with the reference probe by training-set size, #title.], size: 7pt)
}

#figure(image(F + "sample_size.png", width: 100%),
  caption: [Test accuracy (top), Pearson correlation of coding scores with the reference probe (middle) and cosine
  similarity of probe weights with the reference probe (bottom) against the number of training sequences. One line
  per model, coloured by family; bands show ±1 SD over 10 subsets. Dotted line: 10,000 sequences.]) <fig-sample-size>

#heading(numbering: none)[Appendix]

#heading(level: 2, numbering: none)[A.1 Mean coding scores under each edit]
#for (ds, title) in datasets {
  csvtable(ds + "_perturb_coding_score", [Mean coding score of edited *coding* sequences, #title.])
  csvtable(ds + "_perturb_noncoding_score", [Mean coding score of edited *non-coding* sequences, #title.])
}

#heading(level: 2, numbering: none)[A.2 Summary tables]
These are the per-model summaries written by `compare.py`, restricted to the models in this report.
Column names: _emb_ = embedding probe, _6mer_ = 6-mer probe, _resid_ = residual probe, _cod/nc shuf_ = fraction of
nucleotide-shuffled coding / non-coding sequences called coding, _stop in/out_ = fraction of coding sequences
called coding after in-frame / out-of-frame stop insertion, _revcomp_ = same after reverse complement, _p3/p12_ =
codon-position ratio, _nons Δ_ = mean signed Δ of nonsense mutations in coding sequences.
#for (ds, title) in datasets {
  csvtable(ds + "_summary", [Summary, #title.], size: 7pt,
    header: ("Model", "emb", "6mer", "resid", "cod shuf", "nc shuf", "stop in", "stop out", "revcomp", "p3/p12", "nons Δ"))
}

#heading(level: 2, numbering: none)[A.3 Reproduction]
From `Model Evaluation/Coding-Noncoding/Embedding-Patterns/`:

```sh
# 200 bp analyses for one model (default environment; DNABERT-2 and NT-v2 use -e legacy)
GLM_MODEL=PoetschLab/GROVER pixi run python kmer_baseline.py
GLM_MODEL=PoetschLab/GROVER pixi run python perturb.py
GLM_MODEL=PoetschLab/GROVER pixi run python codon_position.py

# Full-length data (downloads listed in the script go to data/full/raw/)
pixi run python prepare_full_length.py
GLM_DATASET=full ./run_models.sh <gpu> <env> <model ids...>   # runs A, B and C for each model

# Summaries and Experiment D (cached embeddings only; no GPU needed)
pixi run python compare.py
pixi run python noncoding_signatures.py
pixi run python noncoding_signatures_full.py

# This report
pixi run python report/make_tables.py && typst compile report/report.typ
```
