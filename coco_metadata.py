# create metadat file for coco dataset like this:
# {"text": "aqua circle with slate blue background", "image_file_name": "images/7.png", "conditioning_image_file_name": "conditioning_images/7.png"} including only the one of the five captions for each image in coco dataset.


import json
import random
import os

def create_metadata_file(coco_images_dir, captions_path, output_file):
    # captions_path should point to a directory containing a json file with caption.json file which contains the captions for the coco dataset.
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)
        
    metadata = []
    
    for image_file in os.listdir(coco_images_dir):
        if image_file.endswith('.jpg'):
            image_id = os.path.splitext(image_file)[0]
            # Find the corresponding captions for this image
            image_captions = [item['caption'] for item in captions_data['annotations'] if item['image_id'] == int(image_id)]
            if image_captions:
                # Use only the first caption for each image
                random_caption = random.randint(1, 5)  # random number between 1 and 5 inclusive
                metadata.append({
                    "text": image_captions[random_caption - 1],  # get the random caption
                    "image_file_name": os.path.join(coco_images_dir, image_file),
                    "conditioning_image_file_name": os.path.join('conditioning_images', image_file.replace('.jpg', '.png'))
                })

    with open(output_file, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    coco_images_dir = 'datasets/coco/depth/images'  # replace with the path to your coco images directory
    captions_path = 'datasets/coco/captions_train2017.json'  # replace with the path to your captions json file
    output_file = 'datasets/coco/depth/metadata.jsonl'  # output metadata file
    create_metadata_file(coco_images_dir, captions_path, output_file)