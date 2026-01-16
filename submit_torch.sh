#!/bin/bash
#SBATCH --job-name=tinker
#SBATCH --open-mode=append
#SBATCH --output=/scratch/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_235_cds
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16

START_TIME=$(date +%s)
echo "Job started at: $(date)"
################################################

source ~/.bashrc

conda activate
conda activate ct

python collect_trajectories_single_turn.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --num-problems 100 \
        --num-samples 160 \
        \
        --fast-eval \
        --eval-workers 16 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/1k_160_single_turn

################################################

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Time taken: $((ELAPSED_TIME / 3600))h $((ELAPSED_TIME % 3600 / 60))m $((ELAPSED_TIME % 60))s"