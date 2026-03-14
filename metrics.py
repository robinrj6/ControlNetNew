#!/usr/bin/env python3
"""
Compute FID (with pytorch-fid) and CLIP score for generated images.

Edit the path variables in the CONFIG section, then run:
	python Quality_Metrics/qualityMetrics.py

NOTE: pytorch-fid requires Inception V3 weights. If you get connection errors,
pre-download them by running this once on a machine with internet:
	python -c "import torch; torch.hub.load_state_dict_from_url('https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth', progress=True)"

Then copy ~/.cache/torch/hub/checkpoints/ to your cluster's cache directory.
Or set TORCH_HOME environment variable to a shared cache location.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Set HuggingFace cache BEFORE importing transformers
HF_CACHE_DIR = Path.cwd() / "models" / "huggingface_cache"
if HF_CACHE_DIR.exists():
	os.environ["HF_HOME"] = str(HF_CACHE_DIR)

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPModel, CLIPProcessor


def log(msg: str) -> None:
	"""Log with timestamp and flush to ensure visibility in real-time logs."""
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"[{timestamp}] {msg}", flush=True)
	sys.stdout.flush()


# =========================
# CONFIG: EDIT THESE PATHS
# =========================
REAL_IMAGES_DIR = Path("datasets/coco/depth_val/images/")
CANNY_IMAGES_DIR = Path("datasets/coco/depth_val/conditioning_images/")
CONTROLNET_IMAGES_DIR = Path("inference_outputs/checkpoint-29000/")
SD15_IMAGES_DIR = Path("inference_outputs/sd_only/")
METADATA_JSONL_PATH = Path("datasets/coco/depth_val/metadata.jsonl")
REPORT_FILE = Path("results/metrics_report.txt")

# Optional settings
CLIP_MODEL_ID = "models/huggingface_cache/hub/models--openai--clip-vit-base-patch32/"
FID_BATCH_SIZE = 32
FID_DIMS = 2048
FID_NUM_WORKERS = 0  # Set to 0 to avoid DataLoader issues with variable image sizes
CLIP_BATCH_SIZE = 16

# Set this to a directory containing pre-downloaded Inception weights if needed
# Expected file: pt_inception-2015-12-05-6726825d.pth in $TORCH_HOME/hub/checkpoints/
TORCH_HOME_OVERRIDE = (Path.cwd() / "models" / "torch_cache").resolve()
FID_WEIGHTS_FILENAME = "pt_inception-2015-12-05-6726825d.pth"

# HuggingFace cache for CLIP model
HUGGINGFACE_CACHE = (Path.cwd() / "models" / "huggingface_cache").resolve()


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Directory not found: {folder}")
	return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def load_metadata_prompts(metadata_jsonl: Path) -> Dict[str, str]:
	"""
	Build a mapping from image stem -> prompt.
	Supports both:
	  - file_name: images/000000123456.jpg
	  - conditioning_file_name: edges/000000123456.png
	"""
	if not metadata_jsonl.exists():
		raise FileNotFoundError(f"Metadata file not found: {metadata_jsonl}")

	stem_to_prompt: Dict[str, str] = {}
	with metadata_jsonl.open("r", encoding="utf-8") as f:
		for line_idx, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON at line {line_idx} in {metadata_jsonl}: {exc}") from exc

			text = row.get("text", "")
			if not isinstance(text, str) or not text.strip():
				continue
			text = text.strip()

			for key in ("file_name", "conditioning_file_name"):
				value = row.get(key)
				if isinstance(value, str) and value.strip():
					stem = Path(value).stem
					stem_to_prompt[stem] = text

	return stem_to_prompt


def _resize_images_in_dir(image_dir: Path, target_size: int = 299) -> Path:
	"""
	Create a temporary directory with resized images for FID computation.
	Inception V3 expects 299x299 input, but we resize with aspect ratio preservation.
	"""
	import tempfile
	import shutil
	
	temp_dir = Path(tempfile.mkdtemp(prefix="fid_resized_"))
	images = list_images(image_dir)
	
	log(f"  Resizing {len(images)} images to {target_size}x{target_size}...")
	for i, img_path in enumerate(images):
		try:
			img = Image.open(img_path).convert("RGB")
			# Resize with aspect ratio preservation, then pad/crop to target size
			img.thumbnail((target_size, target_size), Image.LANCZOS)
			# Create square canvas and paste image (center it)
			canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
			offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
			canvas.paste(img, offset)
			
			output_path = temp_dir / img_path.name
			canvas.save(output_path, "JPEG", quality=95)
		except Exception as e:
			log(f"    Warning: Failed to resize {img_path.name}: {e}")
			continue
		
		if (i + 1) % 50 == 0:
			log(f"    Resized {i + 1}/{len(images)} images")
	
	return temp_dir


def fid_score(real_dir: Path, generated_dir: Path, device: torch.device) -> float:
	"""
	Compute FID while handling variable-sized images by resizing them first.
	"""
	import tempfile
	import shutil
	
	log(f"Starting FID calculation: {real_dir} vs {generated_dir}")
	
	# Resize images to consistent size
	real_resized = _resize_images_in_dir(real_dir, target_size=299)
	gen_resized = _resize_images_in_dir(generated_dir, target_size=299)
	
	try:
		log("Computing FID on resized images...")
		fid_value = float(
			calculate_fid_given_paths(
				[str(real_resized), str(gen_resized)],
				batch_size=FID_BATCH_SIZE,
				device=device,
				dims=FID_DIMS,
				num_workers=FID_NUM_WORKERS,
			)
		)
		log(f"FID calculation complete: {fid_value:.4f}")
		return fid_value
	finally:
		# Clean up temporary directories
		shutil.rmtree(real_resized, ignore_errors=True)
		shutil.rmtree(gen_resized, ignore_errors=True)


def _batched(items, batch_size: int):
	for i in range(0, len(items), batch_size):
		yield items[i : i + batch_size]


@torch.inference_mode()
def clip_score_for_folder(
	image_dir: Path,
	stem_to_prompt: Dict[str, str],
	model: CLIPModel,
	processor: CLIPProcessor,
	device: torch.device,
) -> Tuple[float, int, int]:
	"""
	Returns (mean_clip_score, used_images, skipped_images).

	Score per pair is cosine similarity between normalized image/text embeddings.
	"""
	all_images = list_images(image_dir)
	log(f"  Found {len(all_images)} images in {image_dir}")

	pairs: List[Tuple[Path, str]] = []
	for img_path in all_images:
		prompt = stem_to_prompt.get(img_path.stem)
		if prompt:
			pairs.append((img_path, prompt))

	if not pairs:
		log(f"  No matching prompts found! Skipping {len(all_images)} images.")
		return 0.0, 0, len(all_images)

	log(f"  Processing {len(pairs)} image-prompt pairs in batches of {CLIP_BATCH_SIZE}...")
	total_score = 0.0
	total_count = 0
	batch_num = 0

	for batch in _batched(pairs, CLIP_BATCH_SIZE):
		batch_num += 1
		pil_images = [Image.open(img_path).convert("RGB") for img_path, _ in batch]
		texts = [prompt for _, prompt in batch]

		inputs = processor(text=texts, images=pil_images, return_tensors="pt", padding=True, truncation=True)
		inputs = {k: v.to(device) for k, v in inputs.items()}

		image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
		text_features = model.get_text_features(
			input_ids=inputs["input_ids"],
			attention_mask=inputs["attention_mask"],
		)

		image_features = image_features / image_features.norm(dim=-1, keepdim=True)
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)

		# cosine per matched pair (diagonal)
		scores = (image_features * text_features).sum(dim=-1)

		total_score += scores.sum().item()
		total_count += scores.numel()
		
		log(f"    Batch {batch_num}: processed {len(batch)} pairs")

	mean_score = total_score / max(total_count, 1)
	skipped = len(all_images) - len(pairs)
	return float(mean_score), int(total_count), int(skipped)

def clip_aesthetic_score(
    image_dir: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> Tuple[float, int]:
    """
    Compute CLIP aesthetic score using aesthetic descriptors.
    Based on: https://github.com/christophschuhmann/improved-aesthetic-predictor
    """
    all_images = list_images(image_dir)
    log(f"  Found {len(all_images)} images in {image_dir}")
    
    # Aesthetic prompts
    aesthetic_prompts = [
        "a photo of good composition",
        "a photo of high quality",
        "a professional photograph",
    ]
    
    negative_prompts = [
        "a photo of poor quality",
        "a badly composed photo",
        "an amateur photo",
    ]
    
    total_score = 0.0
    count = 0
    batch_num = 0
    
    log(f"  Processing {len(all_images)} images in batches of {CLIP_BATCH_SIZE}...")
    
    for batch in _batched(all_images, CLIP_BATCH_SIZE):
        batch_num += 1
        pil_images = [Image.open(img_path).convert("RGB") for img_path in batch]
        
        # Process positive prompts
        pos_inputs = processor(
            text=aesthetic_prompts, 
            images=pil_images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
        
        pos_image_features = model.get_image_features(pixel_values=pos_inputs["pixel_values"])
        pos_text_features = model.get_text_features(
            input_ids=pos_inputs["input_ids"],
            attention_mask=pos_inputs["attention_mask"],
        )
        
        # Process negative prompts
        neg_inputs = processor(
            text=negative_prompts, 
            images=pil_images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
        
        neg_image_features = model.get_image_features(pixel_values=neg_inputs["pixel_values"])
        neg_text_features = model.get_text_features(
            input_ids=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
        )
        
        # Normalize
        pos_image_features = pos_image_features / pos_image_features.norm(dim=-1, keepdim=True)
        pos_text_features = pos_text_features / pos_text_features.norm(dim=-1, keepdim=True)
        
        neg_image_features = neg_image_features / neg_image_features.norm(dim=-1, keepdim=True)
        neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
        
        # Aesthetic score = positive alignment - negative alignment
        pos_scores = (pos_image_features * pos_text_features).mean(dim=1)
        neg_scores = (neg_image_features * neg_text_features).mean(dim=1)
        
        aesthetic = (pos_scores - neg_scores) / 2.0  # Normalize to roughly 0-1
        
        total_score += aesthetic.sum().item()
        count += len(batch)
        
        log(f"    Batch {batch_num}: processed {len(batch)} images (total: {count}/{len(all_images)})")
    
    mean_aesthetic = total_score / max(count, 1)
    return float(mean_aesthetic), int(count)


@torch.inference_mode()
def conditioning_fidelity(
    conditioning_dir: Path,
    generated_dir: Path,
    device: torch.device,
) -> Tuple[float, int]:
	"""
	Compute conditioning fidelity using SSIM between conditioning and generated images.
	Measures how well the generated images preserve the structure of conditioning images.
	Returns (mean_fidelity, num_pairs).
	"""
	conditioning_images = list_images(conditioning_dir)
	log(f"  Found {len(conditioning_images)} conditioning images in {conditioning_dir}")
	
	fidelity_scores = []
	matched_pairs = 0
	
	for cond_path in conditioning_images:
		# Try to find corresponding generated image
		gen_path = (Path(generated_dir) / cond_path.name)
		
		if not gen_path.exists():
			continue
		
		try:
			# Load images
			cond_img = Image.open(cond_path).convert("RGB")
			gen_img = Image.open(gen_path).convert("RGB")
			
			# Ensure same size
			if cond_img.size != gen_img.size:
				gen_img = gen_img.resize(cond_img.size, Image.LANCZOS)
			
			# Convert to numpy for SSIM
			import numpy as np
			cond_array = np.array(cond_img).astype(np.float32) / 255.0
			gen_array = np.array(gen_img).astype(np.float32) / 255.0
			
			# Compute SSIM (higher is better, range 0-1)
			score = ssim(cond_array, gen_array, channel_axis=2, data_range=1.0)
			fidelity_scores.append(score)
			matched_pairs += 1
			
		except Exception as e:
			log(f"    Warning: Could not process {cond_path.name}: {e}")
			continue
	
	mean_fidelity = sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0.0
	return float(mean_fidelity), int(matched_pairs)

def main() -> None:
	log("Initializing metrics computation...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	log(f"Device: {device}")
	
	# Set HuggingFace cache
	if HUGGINGFACE_CACHE.exists():
		os.environ["HF_HOME"] = str(HUGGINGFACE_CACHE)
		log(f"Using HF_HOME: {HUGGINGFACE_CACHE}")
	
	# Override torch hub cache location if specified
	if TORCH_HOME_OVERRIDE is not None:
		os.environ["TORCH_HOME"] = str(TORCH_HOME_OVERRIDE)
		weights_path = TORCH_HOME_OVERRIDE / "hub" / "checkpoints" / FID_WEIGHTS_FILENAME
		log(f"Using TORCH_HOME: {TORCH_HOME_OVERRIDE}")
		log(f"Expected FID weights: {weights_path}")
		if not weights_path.exists():
			log(f"ERROR: FID weights not found!")
			raise FileNotFoundError(
				"FID weights not found in shared folder cache. "
				f"Expected: {weights_path}"
			)
		log(f"✓ FID weights found")

	# Validate required dirs/files
	log("Validating input directories...")
	for p in [REAL_IMAGES_DIR, CANNY_IMAGES_DIR, CONTROLNET_IMAGES_DIR, SD15_IMAGES_DIR]:
		if not p.exists() or not p.is_dir():
			log(f"ERROR: Missing directory {p}")
			raise FileNotFoundError(f"Missing directory: {p}")
	if not METADATA_JSONL_PATH.exists():
		log(f"ERROR: Missing metadata file {METADATA_JSONL_PATH}")
		raise FileNotFoundError(f"Missing metadata file: {METADATA_JSONL_PATH}")
	log("✓ All directories and files validated")

	# Quick visibility on dataset sizes
	log("Counting images...")
	n_real = len(list_images(REAL_IMAGES_DIR))
	n_canny = len(list_images(CANNY_IMAGES_DIR))
	n_controlnet = len(list_images(CONTROLNET_IMAGES_DIR))
	n_sd15 = len(list_images(SD15_IMAGES_DIR))

	log("\n" + "="*60)
	log("INPUT SUMMARY")
	log("="*60)
	log(f"Real images      : {n_real} -> {REAL_IMAGES_DIR}")
	log(f"Canny images     : {n_canny} -> {CANNY_IMAGES_DIR}")
	log(f"ControlNet images: {n_controlnet} -> {CONTROLNET_IMAGES_DIR}")
	log(f"SD1.5 images     : {n_sd15} -> {SD15_IMAGES_DIR}")
	log(f"Metadata         : {METADATA_JSONL_PATH}")
	log(f"Device           : {device}")
	log("="*60 + "\n")

	# 1) FID with pytorch-fid
	log("="*60)
	log("STAGE 1/3: Computing FID with pytorch-fid...")
	log("="*60)
	fid_controlnet = fid_score(REAL_IMAGES_DIR, CONTROLNET_IMAGES_DIR, device)
	log(f"✓ ControlNet FID: {fid_controlnet:.4f}")
	
	fid_sd15 = fid_score(REAL_IMAGES_DIR, SD15_IMAGES_DIR, device)
	log(f"✓ SD1.5 FID: {fid_sd15:.4f}")

	# 2) CLIP score (image-prompt alignment)
	log("="*60)
	log("STAGE 2/3: Loading CLIP model and computing scores...")
	log("="*60)
	log("Loading CLIP model...")
	log(f"  Loading CLIPProcessor from: {CLIP_MODEL_ID}")
	processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
	log(f"  ✓ CLIPProcessor loaded")
	
	log(f"  Loading CLIPModel from: {CLIP_MODEL_ID}")
	model = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True).to(device).eval()
	log(f"  ✓ CLIPModel loaded and moved to {device}")
	log("✓ CLIP model fully loaded")

	log("Loading metadata prompts...")
	stem_to_prompt = load_metadata_prompts(METADATA_JSONL_PATH)
	log(f"✓ Loaded {len(stem_to_prompt)} prompts")

	log("Computing CLIP scores for ControlNet...")
	clip_controlnet, used_cn, skipped_cn = clip_score_for_folder(
		CONTROLNET_IMAGES_DIR,
		stem_to_prompt,
		model,
		processor,
		device,
	)
	log(f"✓ ControlNet CLIP: {clip_controlnet:.4f} [used={used_cn}, skipped={skipped_cn}]")
	
	log("Computing CLIP scores for SD1.5...")
	clip_sd15, used_sd, skipped_sd = clip_score_for_folder(
		SD15_IMAGES_DIR,
		stem_to_prompt,
		model,
		processor,
		device,
	)
	log(f"✓ SD1.5 CLIP: {clip_sd15:.4f} [used={used_sd}, skipped={skipped_sd}]")

	# 3) CLIP aesthetic scores
	log("="*60)
	log("STAGE 3/3: Computing CLIP aesthetic scores...")
	log("="*60)
	log("Computing aesthetic scores for ControlNet...")
	aes_controlnet, aes_cn_count = clip_aesthetic_score(
    	CONTROLNET_IMAGES_DIR,
    	model,
    	processor,
    	device,
	)
	log(f"✓ ControlNet Aesthetic: {aes_controlnet:.4f} [processed={aes_cn_count}]")
	
	log("Computing aesthetic scores for SD1.5...")
	aes_sd15, aes_sd_count = clip_aesthetic_score(
    	SD15_IMAGES_DIR,
    	model,
    	processor,
    	device,
	)
	log(f"✓ SD1.5 Aesthetic: {aes_sd15:.4f} [processed={aes_sd_count}]")

	# 4) Conditioning fidelity (optional - requires conditioning images)
	log("="*60)
	log("STAGE 4/4: Computing conditioning fidelity...")
	log("="*60)
	log("Computing conditioning fidelity for ControlNet...")
	cond_fidelity_cn, cond_pairs_cn = conditioning_fidelity(
		CANNY_IMAGES_DIR,
		CONTROLNET_IMAGES_DIR,
		device,
	)
	log(f"✓ ControlNet Conditioning Fidelity: {cond_fidelity_cn:.4f} [pairs={cond_pairs_cn}]")
	
	# Generate report
	log("="*60)
	log("FINAL RESULTS")
	log("="*60)
	report_lines = []
	report_lines.append("="*60)
	report_lines.append("METRICS REPORT")
	report_lines.append("="*60)
	report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	report_lines.append("")
	report_lines.append("Dataset Summary:")
	report_lines.append(f"  Real images      : {n_real}")
	report_lines.append(f"  Canny images     : {n_canny}")
	report_lines.append(f"  ControlNet images: {n_controlnet}")
	report_lines.append(f"  SD1.5 images     : {n_sd15}")
	report_lines.append("")
	report_lines.append("Results:")
	report_lines.append(f"  FID (Real vs ControlNet): {fid_controlnet:.4f}")
	report_lines.append(f"  FID (Real vs SD1.5)    : {fid_sd15:.4f}")
	report_lines.append(f"  CLIP (ControlNet)      : {clip_controlnet:.4f}  [used={used_cn}, skipped={skipped_cn}]")
	report_lines.append(f"  CLIP (SD1.5)           : {clip_sd15:.4f}  [used={used_sd}, skipped={skipped_sd}]")
	report_lines.append(f"  Aesthetic (ControlNet) : {aes_controlnet:.4f}")
	report_lines.append(f"  Aesthetic (SD1.5)      : {aes_sd15:.4f}")
	report_lines.append(f"  Conditioning Fidelity  : {cond_fidelity_cn:.4f}  [pairs={cond_pairs_cn}]")
	report_lines.append("="*60)
	
	# Print to console
	for line in report_lines:
		log(line)
	
	# Save to file
	with REPORT_FILE.open("w", encoding="utf-8") as f:
		f.write("\n".join(report_lines))
	log(f"✓ Report saved to {REPORT_FILE}")

if __name__ == "__main__":
	main()