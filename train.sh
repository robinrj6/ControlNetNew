#!/bin/bash
#SBATCH --job-name=controlnet_train_fill50k
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_train_Fill50k_%j.log
#SBATCH --error=logs/controlnet_train_Fill50k_%j.err

export MODEL_DIR="models/sd15/"
export OUTPUT_DIR="output/fill50k_controlnet"
export DATASET_PATH="datasets/fill50k/"

export HF_HOME="/home/woody/rlvl/rlvl165v/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATASET_PATH \
 --image_column="image" \
 --caption_column="text" \
 --conditioning_image_column="conditioning_image" \
 --resolution=512 \
 --lr_warmup_steps=1000 \
 --learning_rate=5e-6 \
 --validation_steps=500 \
 --num_validation_images=2 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --checkpointing_steps=1000 \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --proportion_empty_prompts=0.05