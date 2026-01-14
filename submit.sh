#!/bin/bash
#SBATCH --job-name=tinker
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100_dev,a100_short,a100_long

source ~/.bashrc

conda activate
conda activate ct

pwd

python collect_trajectories_budget_forcing.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --num-problems 30 \
        --num-samples 32 \
        --num-attempts 5 \
        --push-to-hub bicycleman15/qwen3_4b_very_hard_s1_x5


echo "Run finished at: "
date
exit