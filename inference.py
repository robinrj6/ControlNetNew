from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
from pathlib import Path

base_model_path = "models/sd15/"
controlnet_path = "output/depth_coco_controlnet/"

# Find all checkpoint directories
checkpoints = sorted([d for d in os.listdir(controlnet_path) if d.startswith('checkpoint-')])

if not checkpoints:
    print(f"No checkpoints found in {controlnet_path}")
    print(f"Available directories: {os.listdir(controlnet_path)}")
    exit(1)

print(f"Found {len(checkpoints)} checkpoints: {checkpoints}\n")

# Create output directory for results
os.makedirs("inference_outputs/5e-5", exist_ok=True)

control_image = load_image("./000000012748.png").convert("RGB").resize((512, 512))
# save the control image for reference
control_image.save("inference_outputs/5e-5/control_image.png")
prompt = "A man holding a baby whose petting a horse."

# Generate images for each checkpoint
for checkpoint in checkpoints:
    checkpoint_path = os.path.join(controlnet_path, checkpoint)
    print(f"Loading checkpoint: {checkpoint}")
    
    try:
        # Load controlnet from checkpoint (config.json is in checkpoint-*/controlnet/)
        controlnet = ControlNetModel.from_pretrained(
            checkpoint_path, 
            subfolder="controlnet",
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None

        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None
        )
        
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        # generate image
        generator = torch.manual_seed(0)
        image = pipe(
            prompt, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=1.0
        ).images[0]
        
        # Save with checkpoint name
        output_path = f"inference_outputs/5e-5/{checkpoint}.png"
        image.save(output_path)
        print(f"  ✓ Saved to {output_path}\n")
        
        # Clean up to free memory
        pipe.to("cpu")
        del pipe
        del controlnet
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"  ✗ Error loading {checkpoint}: {e}\n")
        continue

print("Done! All images saved to inference_outputs/")