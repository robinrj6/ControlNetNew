from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import os
from pathlib import Path

base_model_path = "models/sd15/"

# Create output directory for results
os.makedirs("inference_outputs/sd_only", exist_ok=True)

prompt = "A man holding a baby whose petting a horse."

print(f"Generating image with prompt: {prompt}\n")

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None
    )
    
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    # generate image
    generator = torch.manual_seed(0)
    image = pipe(
        prompt, num_inference_steps=20, generator=generator
    ).images[0]
    
    # Save image
    output_path = "inference_outputs/sd_only/generated_image.png"
    image.save(output_path)
    print(f"✓ Saved to {output_path}\n")
    
    # Clean up to free memory
    pipe.to("cpu")
    del pipe
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
except Exception as e:
    print(f"✗ Error: {e}\n")

print("Done!")