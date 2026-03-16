import os
import random
import cv2 as cv
from tqdm import tqdm

original = "datasets/coco/depth/images/"
copy = "datasets/coco/depth/conditioning_images/"

original_files = set(os.path.splitext(f)[0] for f in os.listdir(original) if f.endswith('.jpg'))
copy_files = set(os.path.splitext(f)[0] for f in os.listdir(copy) if f.endswith('.png'))

missing = original_files - copy_files
print(f"Missing {len(missing)} files:")
for f in sorted(missing):
    print(f"  {f}.jpg")

extra = copy_files - original_files
if extra:
    print(f"\nExtra files in copy (no original): {len(extra)}")
    for f in sorted(extra):
        print(f"  {f}.png")

print(f"\nSummary: {len(original_files)} originals, {len(copy_files)} canny images, {len(missing)} missing")

# Generate canny for missing images
if missing:
    print(f"\nGenerating canny edges for {len(missing)} missing images...")
    os.makedirs(copy, exist_ok=True)
    success = 0
    for name in tqdm(sorted(missing), desc="Canny edges"):
        img_path = os.path.join(original, name + ".jpg")
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: Could not read {img_path}")
            continue
        lower = random.randint(50, 200)
        upper = random.randint(max(lower + 50, 200), 400)
        edges = cv.Canny(img, lower, upper, apertureSize=3)
        out_path = os.path.join(copy, name + ".png")
        cv.imwrite(out_path, edges)
        success += 1
    print(f"Done. Generated {success}/{len(missing)} canny images.")
else:
    print("\nNo missing images — nothing to generate.")
