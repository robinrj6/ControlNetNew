#!/bin/bash
#SBATCH --job-name=controlnet_coco_metadata
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_coco_metadata_%j.log
#SBATCH --error=logs/controlnet_coco_metadata_%j.err

python coco_metadata.py