# create metadat file for coco dataset like this:
# {"text": "aqua circle with slate blue background", "image_file_name": "images/7.png", "conditioning_image_file_name": "conditioning_images/7.png"} including only the one of the five captions for each image in coco dataset.


import json
import random
import os

def create_metadata_file(coco_images_dir, captions_path, output_file):
    # captions_path should point to a directory containing a json file with caption.json file which contains the captions for the coco dataset.
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)
    
    # Build a lookup dictionary: image_id -> list of captions (O(n) instead of O(n*m))
    print("Building caption lookup dictionary...")
    captions_dict = {}
    for item in captions_data['annotations']:
        img_id = item['image_id']
        if img_id not in captions_dict:
            captions_dict[img_id] = []
        captions_dict[img_id].append(item['caption'])
    print(f"  Created lookup for {len(captions_dict)} unique images")
        
    metadata = []
    
    image_files = sorted([f for f in os.listdir(coco_images_dir) if f.endswith('.jpg')])
    total_images = len(image_files)
    print(f"Processing {total_images} images...")
    
    for idx, image_file in enumerate(image_files, 1):
        image_id = int(os.path.splitext(image_file)[0])
        # Fast lookup using dictionary
        image_captions = captions_dict.get(image_id, [])
        if image_captions:
            # Use only the first caption for each image
            random_caption = random.randint(1, 5)  # random number between 1 and 5 inclusive
            metadata.append({
                "text": image_captions[random_caption - 1],  # get the random caption
                "image_file_name": os.path.join(coco_images_dir, image_file),
                "conditioning_image_file_name": os.path.join('conditioning_images', image_file.replace('.jpg', '.png'))
            })
        
        # Print progress every 1000 images
        if idx % 1000 == 0 or idx == total_images:
            print(f"  Processed {idx}/{total_images} images ({100*idx//total_images}%)")

    with open(output_file, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"Metadata file created: {output_file}")
    print(f"Total entries: {len(metadata)}")

if __name__ == "__main__":
    coco_images_dir = 'datasets/coco/depth/images'  # replace with the path to your coco images directory
    captions_path = 'datasets/coco/captions_train2017.json'  # replace with the path to your captions json file
    output_file = 'datasets/coco/depth/metadata.jsonl'  # output metadata file
    create_metadata_file(coco_images_dir, captions_path, output_file)