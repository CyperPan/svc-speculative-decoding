#!/bin/bash
#SBATCH --job-name=svc_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate nanovllm

cd ~/svc_speculative/src
python -u test_svc.py

echo "=== Job finished: $(date) ==="
