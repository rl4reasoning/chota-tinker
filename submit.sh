#!/bin/bash
#SBATCH --job-name=tinker
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100_dev,a100_short,a100_long

source ~/.bashrc

conda activate
conda activate ct

pwd

# python collect_trajectories.py \
#         --dataset bicycleman15/intellect_3_code_very_hard \
#         --model Qwen/Qwen3-4B-Instruct-2507 \
#         --backend vllm \
#         --start-problem 500 \
#         --num-problems 50 \
#         --num-samples 32 \
#         \
#         --fast-eval \
#         --eval-workers 8 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 1.0 \
#         --push-to-hub bicycleman15/500_550_interations_test

# python collect_trajectories_single_turn.py \
#     --resume-from checkpoints/20260131_121204_5eec55e4 \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --start-problem 0 \
#     --num-problems 50 \
#     --num-samples 320 \
#     --fast-eval \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 1.0 \
#     --push-to-hub bicycleman15/single_0_50

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 0 \
#     --num-problems 100 \
#     --num-samples 350 \
#     \
#     --fast-eval \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 1.0 \
#     --resume-from checkpoints/20260203_181639_abb82faa \
#     --push-to-hub bicycleman15/new_prompt_single_turn_0_100

python collect_trajectories_budget_forcing.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --start-problem 50 \
        --num-problems 50 \
        --num-samples 35 \
        --num-attempts 10 \
        \
        --fast-eval \
        --eval-workers 16 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --resume-from checkpoints/20260203_190405_af3e55b3 \
        --push-to-hub bicycleman15/new_prompt_s1_50_100

# 100_150_s1 # done!
# 250_300_s1 # checkpoints/20260119_015555_76d3d6ec
# 500_550_s1 # checkpoints/20260119_015555_e8808fef
# 550_600_s1 # done!


echo "Run finished at: "
date
exit