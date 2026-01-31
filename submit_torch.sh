#!/bin/bash
#SBATCH --job-name=tinker
#SBATCH --open-mode=append
#SBATCH --output=/scratch/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_235_cds
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16

START_TIME=$(date +%s)
echo "Job started at: $(date)"
################################################

source ~/.bashrc # so that we can read HF_AUTH_TOKEN :)

conda activate
conda activate ct

# We want to trigger both of these
# first command runs on 1 GPU -- change your slurm script and `push-to-hub` commands accordingly
# first check on smaller number of problems if everything is working and getting pushed to HF correctly or not :)
# when all done, trigger!

# needs 1 GPU only
# reduce number of problems to check :)
# python collect_trajectories.py \
#         --dataset bicycleman15/intellect_3_code_very_hard \
#         --model Qwen/Qwen3-4B-Instruct-2507 \
#         --backend vllm \
#         --start-problem 0 \
#         --num-problems 500 \
#         --num-samples 32 \
#         \
#         --fast-eval \
#         --eval-workers 8 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 1.0 \
#         --push-to-hub bicycleman15/0_500_interactions

python collect_trajectories_single_turn.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --start-problem 50 \
    --num-problems 100 \
    --num-samples 320 \
    \
    --fast-eval \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 1.0 \
    --push-to-hub bicycleman15/single_50_100

# also change file name to `collect_trajectories_budget_forcing.py` to collect s1 scaling
# change start-problem to 500, so that we collect 
# change push-to-hub accordingly when running different filename and start-problem

################################################

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Time taken: $((ELAPSED_TIME / 3600))h $((ELAPSED_TIME % 3600 / 60))m $((ELAPSED_TIME % 60))s"
