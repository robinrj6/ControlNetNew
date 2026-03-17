#!/bin/bash
#SBATCH --job-name=controlnet_metrics_cfg_7.5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_metrics_cfg_7.5_%j.log
#SBATCH --error=logs/controlnet_metrics_cfg_7.5_%j.err

python metrics.py 