#!/bin/bash
#SBATCH --job-name=controlnet_inference_experiment
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_inference_experiment_%j.log
#SBATCH --error=logs/controlnet_inference_experiment_%j.err

python inference_single_ckpt.py \
 --checkpoint_path="output/canny_coco_controlnet_experiment/checkpoint-29000" \
 --data_dir="datasets/coco/depth_val"