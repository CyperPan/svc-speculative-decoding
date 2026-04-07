#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate nanovllm

export HF_HOME=/scratch/pan.zhiyu/hf_cache
export HF_HUB_CACHE=/scratch/pan.zhiyu/hf_cache/hub

export SVC_MODEL="Qwen/Qwen2.5-7B"
export SVC_OUTDIR="./diag_results"

cd ~/svc_speculative/src
python -u diagnose_failures.py

echo "=== Job finished: $(date) ==="
