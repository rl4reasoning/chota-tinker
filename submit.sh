#!/bin/bash
#SBATCH --job-name=550_600_s1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --partition=a100_dev,a100_short,a100_long

source ~/.bashrc

conda activate
conda activate ct

pwd

python collect_trajectories_budget_forcing.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --start-problem 550 \
        --num-problems 50 \
        --num-samples 32 \
        \
        --fast-eval \
        --eval-workers 8 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/550_600_s1

# 100_150_s1
# 250_300_s1
# 500_550_s1
# 550_600_s1


echo "Run finished at: "
date
exit