#!/bin/bash
#SBATCH --job-name=i3_gpt_mt_5_300
#SBATCH --open-mode=append
#SBATCH --output=/scratch/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_235_cds
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8

START_TIME=$(date +%s)
echo "Job started at: $(date)"
################################################

source ~/.bashrc # so that we can read HF_AUTH_TOKEN :)

conda activate
conda activate ct

python collect_trajectories.py \
    --dataset anirudhb11/intellect_3_code_very_hard_top_400_hardest \
    --model openai/gpt-oss-120b \
    --backend vllm \
    --start-problem 300 \
    --num-problems 100 \
    --num-samples 64 \
    --max-turns 5 \
    --gpu-memory-utilization 0.7 \
    \
    --fast-eval \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 5.0 \
    --push-to-hub bicycleman15/i3_gpt_mt_5_300

# python collect_trajectories_single_turn.py \
# --dataset anirudhb11/lcb_v6_feb_may_2025_formatted \
# --model openai/gpt-oss-120b \
# --backend vllm \
# --start-problem 65 \
# --num-problems 66 \
# --num-samples 80 \
# --max-tokens 16384 \
# --gpu-memory-utilization 0.7 \
# \
# --fast-eval \
# --eval-workers 8 \
# --eval-batch-size 8 \
# --eval-timeout-s 5.0 \
# --push-to-hub bicycleman15/lcb_gpt_st16k_65

# gpt-oss st after fix !!!
# python eval_checkpoint_single_turn.py \
#     checkpoints/20260211_150212_bac4e507 \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/lcb_gpt_st8k_0

# single-turn 30b 0
# python eval_checkpoint_single_turn.py \
#     checkpoints/20260211_135945_e066b0ec \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/30b_lcb_st_0

# single-turn 30b 65
# python eval_checkpoint_single_turn.py \
#     checkpoints/20260211_140030_fbc52a0b \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/30b_lcb_st_65

# multi-turn 30b 10turns 0
# python eval_checkpoint_single_turn.py \
#     checkpoints/20260211_140030_fbc52a0b \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/30b_lcb_st_65


# python collect_trajectories.py \
#     --dataset anirudhb11/lcb_v6_feb_may_2025_formatted \
#     --model openai/gpt-oss-120b \
#     --backend vllm \
#     --start-problem 0 \
#     --num-problems 65 \
#     --num-samples 32 \
#     --max-turns 10 \
#     --gpu-memory-utilization 0.7 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/lcb_gpt_mt_10_0

# 30b mt 5 turns 0
# python collect_trajectories.py \
#     --dataset anirudhb11/lcb_v6_feb_may_2025_formatted \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 0 \
#     --num-problems 65 \
#     --num-samples 70 \
#     --max-turns 5 \
#     \
#     --fast-eval \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/lcb_30b_mt_5_0 \
#     --resume-from checkpoints/20260211_141225_966a06d4

# python collect_trajectories_single_turn.py \
#     --dataset anirudhb11/lcb_v6_feb_may_2025_formatted \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 65 \
#     --num-problems 66 \
#     --num-samples 350 \
#     \
#     --fast-eval \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/lcb_30b_st_65

# python collect_trajectories_single_turn.py \
#     --dataset anirudhb11/intellect_3_code_very_hard_top_400_hardest \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 150 \
#     --num-problems 50 \
#     --num-samples 350 \
#     \
#     --fast-eval \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/prompt_v4_single_turn_top400dataset_150 \
#     --resume-from checkpoints/20260208_175126_a78b2dfc

# python eval_checkpoint_single_turn.py checkpoints/20260208_175126_a78b2dfc \
#         --eval-workers 16 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 5.0 \
#         --push-to-hub bicycleman15/prompt_v4_single_turn_top400dataset_150

# checkpoints/20260208_172417_bee66f7a -- 0
# checkpoints/20260208_172425_e76a5312 -- 50
# checkpoints/20260208_172417_07267d72 -- 100
# checkpoints/20260208_175126_a78b2dfc -- 150

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
#         --max-model-len 120000 \
#         --max-interaction-output-tokens 4000 \
#         \
#         --fast-eval \
#         --eval-workers 8 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 1.0 \
#         --push-to-hub bicycleman15/0_500_interactions

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 50 \
#     --num-problems 100 \
#     --num-samples 320 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 1.0 \
#     --push-to-hub bicycleman15/single_50_100

# python collect_trajectories_budget_forcing.py \
#         --dataset bicycleman15/intellect_3_code_very_hard \
#         --model Qwen/Qwen3-4B-Instruct-2507 \
#         --backend vllm \
#         --start-problem 75 \
#         --num-problems 25 \
#         --num-samples 32 \
#         --num-attempts 10 \
#         \
#         --fast-eval \
#         --eval-workers 8 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 1.0 \
#         --push-to-hub bicycleman15/75_100_s1_10_attempts

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
#     --push-to-hub bicycleman15/new_prompt_single_turn_0_100

# python collect_trajectories_budget_forcing.py \
#         --dataset bicycleman15/intellect_3_code_very_hard \
#         --model Qwen/Qwen3-4B-Instruct-2507 \
#         --backend vllm \
#         --start-problem 50 \
#         --num-problems 50 \
#         --num-samples 35 \
#         --num-attempts 10 \
#         \
#         --fast-eval \
#         --eval-workers 16 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 1.0 \
#         --push-to-hub bicycleman15/new_prompt_s1_50_100

# python collect_trajectories.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --backend vllm \
#     --start-problem 50 \
#     --num-problems 50 \
#     --num-samples 32 \
#     --max-turns 10 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 1.0 \
#     --push-to-hub bicycleman15/interaction_llama_50_100

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --backend vllm \
#     --start-problem 50 \
#     --num-problems 50 \
#     --num-samples 350 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 1.0 \
#     --push-to-hub bicycleman15/llama_single_turn_50_100

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 50 \
#     --num-problems 25 \
#     --num-samples 350 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 1.0 \
#     --push-to-hub bicycleman15/prompt_v2_single_turn_50_75

# python collect_trajectories_budget_forcing.py \
#         --dataset bicycleman15/intellect_3_code_very_hard \
#         --model Qwen/Qwen3-4B-Instruct-2507 \
#         --backend vllm \
#         --start-problem 50 \
#         --num-problems 25 \
#         --num-samples 35 \
#         --num-attempts 10 \
#         \
#         --fast-eval \
#         --eval-workers 8 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 1.0 \
#         --push-to-hub bicycleman15/prompt_v2_s1_50_75

# python collect_trajectories.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 0 \
#     --num-problems 50 \
#     --num-samples 32 \
#     --max-turns 20 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/30b_20turns_0 \
#     --resume-from checkpoints/20260207_024859_5ea526b6

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 0 \
#     --num-problems 50 \
#     --num-samples 320 \
#     \
#     --fast-eval \
#     --eval-workers 8 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/prompt_v3_single_turn_0_50 \
#     --resume-from checkpoints/20260205_134734_bba79239

# python collect_trajectories_single_turn.py \
#     --dataset bicycleman15/intellect_3_code_very_hard \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --backend vllm \
#     --start-problem 50 \
#     --num-problems 50 \
#     --num-samples 320 \
#     \
#     --fast-eval \
#     --eval-workers 16 \
#     --eval-batch-size 8 \
#     --eval-timeout-s 5.0 \
#     --push-to-hub bicycleman15/30b_single_turn_50_100

# python eval_checkpoint_single_turn.py checkpoints/20260205_135600_f443060c \
#         --eval-workers 16 \
#         --eval-batch-size 8 \
#         --eval-timeout-s 5.0 \
#         --push-to-hub bicycleman15/30b_single_turn_0_50

# also change file name to `collect_trajectories_budget_forcing.py` to collect s1 scaling
# change start-problem to 500, so that we collect 
# change push-to-hub accordingly when running different filename and start-problem



################################################

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Time taken: $((ELAPSED_TIME / 3600))h $((ELAPSED_TIME % 3600 / 60))m $((ELAPSED_TIME % 60))s"
