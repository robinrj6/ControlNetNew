from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
import json
import argparse
from pathlib import Path

def main(args):
    base_model_path = "models/sd15/"
    checkpoint_path = args.checkpoint_path
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    checkpoint_name = os.path.basename(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_name}\n")
    
    # Create output directory for results
    os.makedirs(f"inference_outputs/{checkpoint_name}_cfg=7.5", exist_ok=True)
    
    # Derive paths from parent directory
    metadata_path = os.path.join(args.data_dir, "metadata.jsonl")
    control_images_dir = os.path.join(args.data_dir, "conditioning_images")
    
    # Verify paths exist
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        exit(1)
    if not os.path.exists(control_images_dir):
        print(f"Control images directory not found: {control_images_dir}")
        exit(1)
    
    # Load metadata and control images
    print(f"Loading metadata from {metadata_path}...")
    metadata_list = []
    for line in open(metadata_path, 'r'):
        data = json.loads(line)
        metadata_list.append(data)
    
    print(f"Found {len(metadata_list)} images\n")
    
    try:
        # Load controlnet from checkpoint once
        print("Loading ControlNet...")
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
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        
        print("Generating images...\n")
        
        # Generate images for each metadata entry
        for idx, data in enumerate(metadata_list):
            try:
                prompt = data['text'][0] if isinstance(data['text'], list) else data['text']
                control_image_path = os.path.join(control_images_dir, data['conditioning_image_file_name'].split('/')[-1])
                image_filename = Path(data['image_file_name']).stem
                
                if not os.path.exists(control_image_path):
                    print(f"[{idx+1}/{len(metadata_list)}] Control image not found: {control_image_path}, skipping...")
                    continue
                
                print(f"[{idx+1}/{len(metadata_list)}] Generating: {prompt[:60]}...")
                
                # Load control image
                control_image = load_image(control_image_path).convert("RGB").resize((512, 512))
                
                # generate image
                generator = torch.manual_seed(args.seed + idx)
                image = pipe(
                    prompt, num_inference_steps=args.steps, generator=generator, image=control_image, guidance_scale=7.5, controlnet_conditioning_scale=args.scale
                ).images[0]
                
                # Save image
                output_path = f"inference_outputs/{checkpoint_name}_cfg=7.5/{image_filename}.png"
                image.save(output_path)
                print(f"  ✓ Saved to {output_path}\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue
        
        # Clean up to free memory
        pipe.to("cpu")
        del pipe
        del controlnet
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}\n")
        return

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a single ControlNet checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="datasets/coco/depth_val", help="Path to data folder containing metadata.jsonl and conditioning_images/")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    main(args)