from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
from pathlib import Path

base_model_path = "models/sd15/"
controlnet_path = "output/fill50k_controlnet/"

# Find all checkpoint directories
checkpoints = sorted([d for d in os.listdir(controlnet_path) if d.startswith('checkpoint-')])

if not checkpoints:
    print(f"No checkpoints found in {controlnet_path}")
    print(f"Available directories: {os.listdir(controlnet_path)}")
    exit(1)

print(f"Found {len(checkpoints)} checkpoints: {checkpoints}\n")

# Create output directory for results
os.makedirs("inference_outputs", exist_ok=True)

control_image = load_image("./inference.jpg")
prompt = "glowing pale blue circle with light yellow background"

# Generate images for each checkpoint
for checkpoint in checkpoints:
    checkpoint_path = os.path.join(controlnet_path, checkpoint, "controlnet")
    print(f"Loading checkpoint: {checkpoint}")
    
    try:
        # Load controlnet from checkpoint
        controlnet = ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16
        )
        
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        
        # generate image
        generator = torch.manual_seed(0)
        image = pipe(
            prompt, num_inference_steps=50, generator=generator, image=control_image
        ).images[0]
        
        # Save with checkpoint name
        output_path = f"inference_outputs/{checkpoint}.png"
        image.save(output_path)
        print(f"  ✓ Saved to {output_path}\n")
        
        # Clean up to free memory
        del pipe
        del controlnet
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Error loading {checkpoint}: {e}\n")
        continue

print("Done! All images saved to inference_outputs/")