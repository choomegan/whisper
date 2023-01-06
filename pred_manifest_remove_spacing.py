import os
import json
from tqdm import tqdm

# need to strip off the spacing before and after the sentence

PATH = '/whisper/datasets/jtubespeech/ms_3/annotated_data_whisper_ms/pred_manifest_updated.json'
PATH_OUTPUT = '/whisper/datasets/jtubespeech/ms_3/annotated_data_whisper_ms/pred_manifest.json'

# load the manifest
with open(PATH, 'r+', encoding='utf-8') as fr:
    lines = fr.readlines()

data_dict = [json.loads(line.strip('\r\n')) for line in lines]

for entry in tqdm(data_dict):
    entry['text'] = ' '.join(entry['text'].strip().split())

# export the new json file
with open(PATH_OUTPUT, 'w+', encoding='utf-8') as fw:
    for entry in data_dict:
        fw.write(json.dumps(entry)+'\n')