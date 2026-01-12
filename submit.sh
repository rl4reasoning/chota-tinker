#!/bin/bash
#SBATCH --job-name=tinker
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:0
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100_dev,a100_short,a100_long

conda activate
conda activate tinker

pwd

export TINKER_API_KEY="tml-xUnO6eCIzrJex0Ge9C5ji3frSTRivhhj3Bi1yAVhEcOgw6Pe6YdLTQzPwfn1toHTnAAAA"

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --num-problems 20 \
#     --num-samples 8 \
#     --push-to-hub bicycleman15/qwen3_4b_instruct_very_hard_single_turn

python collect_trajectories.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --num-problems 20 \
    --num-samples 8 \
    --push-to-hub bicycleman15/qwen3_4b_instruct_very_hard


echo "Run finished at: "
date
exit