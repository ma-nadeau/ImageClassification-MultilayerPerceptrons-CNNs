#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:4 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=fall2024-comp551
#SBATCH -e %N-%j.err # STDERR

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

python ../../ImageClassification-MultilayerPerceptrons-CNNs/src/Code/PrepareDataset.py
