from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import os 
import json
from pathlib import Path

base_model_path = "models/sd15/"

# Create output directory for results
os.makedirs("inference_outputs/sd_only", exist_ok=True)

prompt_path = "datasets/coco/depth_val/metadata.jsonl"  # path to metadata file containing captions

# Load all prompts first
prompts = []
for line in open(prompt_path, 'r'):
    data = json.loads(line)
    prompts.append(data)

print(f"Found {len(prompts)} prompts\n")

# Initialize pipeline once
print("Initializing pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

print("Generating images...\n")

# Generate images for each prompt
for idx, data in enumerate(prompts):
    try:
        prompt = data['text'] if isinstance(data['text'], str) else data['text'][0]  # handle both string and list
        image_filename = Path(data['image_file_name']).stem  # get filename without extension
        
        print(f"[{idx+1}/{len(prompts)}] Generating: {prompt}")
        
        # generate image
        generator = torch.manual_seed(idx)  # use index for deterministic but varied seeds
        image = pipe(
            prompt, num_inference_steps=50, generator=generator
        ).images[0]
        
        # Save image
        output_path = f"inference_outputs/sd_only/{image_filename}.png"
        image.save(output_path)
        print(f"  ✓ Saved to {output_path}\n")
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        continue

# Clean up to free memory
print("Cleaning up...")
pipe.to("cpu")
del pipe
torch.cuda.empty_cache()
import gc
gc.collect()

print("Done! All images saved to inference_outputs/sd_only/")