# code for renaming `prompt` to `text`, `source` to `conditioning_file_name` and `target` to `file_name` in the metadata file of fill50k dataset

import json

with open('fill50k/metadata.jsonl', 'r') as f:
    metadata = [json.loads(line) for line in f]
for item in metadata:
    item['text'] = item.pop('prompt')
    item['conditioning_file_name'] = item.pop('source')
    item['file_name'] = item.pop('target')
with open('fill50k/metadata.jsonl', 'w') as f:
    for item in metadata:
        f.write(json.dumps(item) + '\n')