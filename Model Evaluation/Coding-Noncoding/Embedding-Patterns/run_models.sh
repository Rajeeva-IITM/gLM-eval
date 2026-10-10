#!/usr/bin/env bash
# Runs kmer_baseline, perturb and codon_position for each model in turn on one GPU.
# Usage: ./run_models.sh <gpu> <pixi-env> <model>...      (set GLM_DATASET=full for the full-length data)
# Example: GLM_DATASET=full ./run_models.sh 1 legacy zhihan1996/DNABERT-2-117M
cd "$(dirname "$0")"
gpu=$1; env=$2; shift 2
logs=results/${GLM_DATASET:-gb}_logs
mkdir -p "$logs"
for m in "$@"; do
  for script in kmer_baseline perturb codon_position; do
    CUDA_VISIBLE_DEVICES=$gpu GLM_MODEL=$m pixi run -e "$env" python $script.py > "$logs/${m##*/}_$script.log" 2>&1 \
      && echo "DONE ${m##*/} $script" || echo "FAILED ${m##*/} $script"
  done
done
