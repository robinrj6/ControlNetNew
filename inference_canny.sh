#!/bin/bash
#SBATCH --job-name=controlnet_canny_inference_experiment_no_prompt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_canny_inference_experiment_no_prompt_%j.log
#SBATCH --error=logs/controlnet_canny_inference_experiment_no_prompt_%j.err

python inference_canny.py 