#!/bin/bash
# Submit all single-turn jobs for problems 0-1000 in 50-problem intervals
# Usage: bash submit_single_turn.sh

# To submit all jobs at once, run this script
# To submit individual jobs, copy the sbatch command for the desired range

for start in $(seq 0 25 975); do
    end=$((start + 25))
    
    sbatch --job-name="st_${start}_${end}" \
           --open-mode=append \
           --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out \
           --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err \
           --export=ALL \
           --time=12:00:00 \
           --gres=gpu:a100:1 \
           --mem=100G \
           --cpus-per-task=8 \
           --partition=a100_dev,a100_short,a100_long \
           --wrap="source ~/.bashrc && conda activate && conda activate ct && \
python collect_trajectories_single_turn.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --start-problem ${start} \
    --num-problems 25 \
    --num-samples 160 \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 1.0 \
    --push-to-hub bicycleman15/st_k160_${start}_${end}"
    
    echo "Submitted job for problems ${start}-${end}"
done

echo "All jobs submitted!"
