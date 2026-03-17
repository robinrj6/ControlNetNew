#!/bin/bash
#SBATCH --job-name=controlnet_metrics_experiment
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_metrics_experiment_%j.log
#SBATCH --error=logs/controlnet_metrics_experiment_%j.err

python metrics.py 