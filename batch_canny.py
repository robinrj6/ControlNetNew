import numpy as np
import cv2 as cv
import os
import random
from tqdm import tqdm

def canny_edge_detection(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Random thresholds for variation
    lower_threshold = random.randint(50, 200)
    upper_threshold = random.randint(max(lower_threshold + 50, 200), 400)
    
    edges = cv.Canny(img, lower_threshold, upper_threshold, apertureSize=3)
    return edges, lower_threshold, upper_threshold

def main():
    image_path = 'datasets/coco/depth_val/images/'
    output_path = 'datasets/coco/depth_val/canny_conditioning_images/'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process\n")
    
    # Process with progress bar
    image_count = 0
    for filename in tqdm(image_files, desc="Processing Canny edges"):
        result = canny_edge_detection(os.path.join(image_path, filename))
        if result is not None:
            edges, lower, upper = result
            output_filename = os.path.splitext(filename)[0] + '.png'
            cv.imwrite(os.path.join(output_path, output_filename), edges)
            image_count += 1
    
    print(f"\nTotal images processed: {image_count}/{total_images}")

if __name__ == "__main__":
    main()