import os
import json
import tqdm
import whisper
import argparse
from typing import Any, List, Dict

def detect_language_from_path(path: str, model: Any) -> str:

    # load audio and pad/trim it to fit 30 seconds
    #print(f'dl1: {path} -- type: {type(path)}')
    audio = whisper.load_audio(path)
    #print(f'dl2: {audio} -- type: {type(audio)}')
    audio = whisper.pad_or_trim(audio)
    #print(f'dl3: {audio} -- type: {type(audio)}')

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    #print(f'dl4: {mel} -- type: {type(mel)} -- size: {mel.size()}')

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    print(f"[{path}] Detected language: {lang}")

    return lang

def detect_language_from_manifest(manifest_path: str, model: Any) -> List[Dict[str, str]]:

    with open(manifest_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    items = [json.loads(line.strip('\r\n')) for line in lines]
    base_dir = os.path.dirname(manifest_path)

    for item in tqdm.tqdm(items):
        audio_filepath = os.path.join(base_dir, item['audio_filepath'])
        lang = detect_language_from_path(audio_filepath, model)
        item['language_pred'] = lang

    return items
    
def save_manifest(items: List[Dict[str, str]], save_path: str = 'pred_manifest.json') -> None:
    with open(save_path, 'w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run filewise language detection using Whisper.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest file')
    args = parser.parse_args()

    # python3 lid_inference.py --model /whisper/whisper-large-model/large.pt --manifest /whisper/datasets/mms/data_to_i2r/mms_set_1/manifest.json
    # python3 lid_inference.py --model /whisper/whisper-large-model/large.pt --manifest /whisper/datasets/jtubespeech/ms_3/annotated_data_whisper_ms/manifest_4.json

    model = whisper.load_model(args.model, download_root='/whisper/whisper-large-model')

    # check the model state dict
    # print(f'model state dict - {model.state_dict}')

    pred_items = detect_language_from_manifest(args.manifest, model)
    save_manifest(pred_items, '/whisper/datasets/jtubespeech/ms_3/annotated_data_whisper_ms/pred_manifest.json')
