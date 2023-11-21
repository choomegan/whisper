import os
import json
import tqdm
import whisper
import argparse
import numpy as np
from typing import Any, List, Dict

def detect_language_from_path(path: str, model: Any) -> str:
    
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    lang = max(probs, key=probs.get)
    confidence = probs[lang]

    return lang, confidence, probs

def detect_language_from_manifest(manifest_path: str, model: Any) -> List[Dict[str, str]]:

    with open(manifest_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    items = [json.loads(line.strip('\r\n')) for line in lines]
    base_dir = os.path.dirname(manifest_path)

    for item in tqdm.tqdm(items):
        audio_filepath = os.path.join(base_dir, item['audio_filepath'])
        lang, confidence, probs = detect_language_from_path(audio_filepath, model)
        item['language_pred'] = lang
        item['confidence'] = round(confidence, 5)
        item['confidence_tl'] = round(probs['tl'], 5)

    return items
    
def save_manifest(items: List[Dict[str, str]], save_path: str = 'pred_manifest.json') -> None:
    with open(save_path, 'w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')

if __name__ == '__main__':

    model_path = '/models/whisper/medium.pt'
    batch = 'tlwiki50_split_batch_01'
    manifest_path = f'/datasets/MALTESE/scraping/{batch}/manifest.json'
    save_path = f'/datasets/MALTESE/scraping/{batch}/pred_manifest_whisper.json'
    
    model = whisper.load_model(model_path, download_root='/whisper')

    pred_items = detect_language_from_manifest(manifest_path, model)
    save_manifest(pred_items, save_path)
