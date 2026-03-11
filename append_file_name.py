# append file_name to dataset metadata
# ie; {"text": "aqua circle with slate blue background", "image": "images/7.png", "conditioning_image": "conditioning_images/7.png"}
# should look like {"text": "aqua circle with slate blue background", "image_file_name": "images/7.png", "conditioning_image_file_name": "conditioning_images/7.png"}

import json

def append_file_name_to_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = [json.loads(line) for line in f]

    for item in metadata:
        item['image_file_name'] = item.pop('image')
        item['conditioning_image_file_name'] = item.pop('conditioning_image')

    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
if __name__ == "__main__":
    metadata_path = "datasets/fill50k/metadata.jsonl"
    append_file_name_to_metadata(metadata_path)