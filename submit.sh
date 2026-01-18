#!/bin/bash
#SBATCH --job-name=s1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
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
        --vllm-multi-gpu \
        --vllm-gpu-ids 0,1,2,3 \
        --num-problems 1000 \
        --num-samples 32 \
        \
        --fast-eval \
        --eval-workers 8 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/1k_32_s1


echo "Run finished at: "
date
exit