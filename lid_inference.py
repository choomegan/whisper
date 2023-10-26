import os
import json
import tqdm
import whisper
import argparse
import numpy as np
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

    # temp measure to sift out the targeted languages
    # lang = max(probs, key=probs.get)
    target_dict = {
        'en': probs['en'],
        'ms': probs['ms'],
        'id': probs['id']
    }

    lang = max(target_dict, key=probs.get)

    # print(f"Probs: {probs}")
    # print([probs['en'], probs['ms'], probs['id']])


    '''
        Probs: {'en': 0.9970285296440125, 'zh': 3.9868755266070366e-05, 'de': 2.232918268418871e-05, 'es': 0.00034065687214024365, 'ru': 7.159143569879234e-05, 'ko': 3.5296776331961155e-05, 'fr': 6.79704244248569e-05, 'ja': 0.00010635217040544376, 'pt': 6.719942757626995e-05, 'tr': 5.842810423928313e-05, 'pl': 2.713300818868447e-05, 'ca': 1.2277753569378547e-07, 'nl': 9.12607902137097e-06, 'ar': 4.6735665819142014e-05, 'sv': 2.8863537409051787e-06, 'it': 4.2257037421222776e-05, 'id': 1.1772028301493265e-05, 'hi': 1.2039626199111808e-05, 'fi': 5.930111456109444e-06, 'vi': 6.3538586800859775e-06, 'iw': 5.135241281095659e-06, 'uk': 7.154112608986907e-07, 'el': 1.7025484339683317e-05, 'ms': 1.4020484741195105e-05, 'cs': 1.8390102241028217e-06, 'ro': 4.740271833725274e-06, 'da': 1.373282543681853e-06, 'hu': 3.46476099366555e-06, 'ta': 1.4094077869231114e-06, 'no': 3.623018471898831e-07, 'th': 3.1782856240170076e-06, 'ur': 3.5076559470326174e-06, 'hr': 3.0438636144936027e-07, 'bg': 2.3240782809352822e-07, 'lt': 9.606798023753527e-09, 'la': 0.00011903973791049793, 'mi': 1.8944512703455985e-05, 'ml': 2.0862933070020517e-06, 'cy': 0.0013500405475497246, 'sk': 4.1996219124484924e-07, 'te': 4.259388788341312e-06, 'fa': 1.0266967365168966e-06, 'lv': 5.4335874466460155e-08, 'bn': 9.876494004856795e-06, 'sr': 1.7879072666815432e-09, 'az': 1.7767780136068723e-08, 'sl': 6.253039828152396e-07, 'kn': 9.260308075909052e-09, 'et': 9.421488478267293e-09, 'mk': 2.9264060064093655e-09, 'br': 5.0988364819204435e-06, 'eu': 1.2464975895909447e-07, 'is': 1.4622133903685608e-07, 'hy': 3.7192911861438915e-08, 'ne': 2.4028576461887496e-08, 'mn': 4.4296859869064065e-07, 'bs': 1.754736302927995e-07, 'kk': 3.295120176716182e-09, 'sq': 1.8655347489016094e-08, 'sw': 3.970867965108482e-06, 'gl': 3.946922220166016e-07, 'mr': 7.086571685022136e-08, 'pa': 5.426586113799203e-08, 'si': 1.7392263771398575e-06, 'km': 1.815683936001733e-05, 'sn': 2.040075560216792e-05, 'yo': 7.494747023883974e-06, 'so': 1.2587683118070458e-09, 'af': 1.1259744496783242e-06, 'oc': 2.3009549465768941e-07, 'ka': 4.055503044497755e-09, 'be': 1.96202325497552e-08, 'tg': 3.595490571939308e-10, 'sd': 8.813327667667181e-08, 'gu': 2.2882256089928887e-09, 'am': 7.547239277982953e-09, 'yi': 3.4046732366732613e-07, 'lo': 4.6185029134448996e-08, 'uz': 2.139453978561301e-12, 'fo': 8.707791465667469e-08, 'ht': 5.063764092483325e-07, 'ps': 4.1246767779057336e-08, 'tk': 2.0536711220486836e-11, 'nn': 0.0002798727073241025, 'mt': 1.4938077441684072e-08, 'sa': 2.244268216600176e-06, 'lb': 9.543087847729836e-11, 'my': 1.9657325367461453e-07, 'bo': 4.2230126950926206e-07, 'tl': 2.276051054650452e-05, 'mg': 3.40018811717524e-11, 'as': 2.6395463592621127e-08, 'tt': 2.161007900403078e-10, 'haw': 5.8318993978900835e-05, 'ln': 5.11422282301055e-09, 'ha': 1.3116435715332386e-09, 'ba': 2.7621937723210088e-11, 'jw': 8.903967682272196e-06, 'su': 6.349253900417295e-10}
    '''

    confidence = np.exp(target_dict[lang])/sum(np.exp(list(target_dict.values())))

    print(f"[{path}] Detected language: {lang} - Confidence: {confidence:.5f}")

    confidence_generic = {
        'en': np.exp(target_dict['en'])/sum(np.exp(list(target_dict.values()))),
        'ms': np.exp(target_dict['ms'])/sum(np.exp(list(target_dict.values()))),
        'id': np.exp(target_dict['id'])/sum(np.exp(list(target_dict.values()))),
    }


    return lang, confidence, confidence_generic

def detect_language_from_manifest(manifest_path: str, model: Any) -> List[Dict[str, str]]:

    with open(manifest_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    items = [json.loads(line.strip('\r\n')) for line in lines]
    base_dir = os.path.dirname(manifest_path)

    for item in tqdm.tqdm(items):
        audio_filepath = os.path.join(base_dir, item['audio_filepath'])
        lang, confidence, confidence_generic = detect_language_from_path(audio_filepath, model)
        item['language_pred'] = lang
        item['confidence'] = round(confidence, 5)
        item['confidence_en'] = round(confidence_generic['en'], 5)
        item['confidence_ms'] = round(confidence_generic['ms'], 5)
        item['confidence_id'] = round(confidence_generic['id'], 5)

    return items
    
def save_manifest(items: List[Dict[str, str]], save_path: str = 'pred_manifest.json') -> None:
    with open(save_path, 'w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run filewise language detection using Whisper.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save predicted manifest file')
    args = parser.parse_args()

    # python3 lid_inference.py --model /whisper/whisper-large-model/small.pt --manifest /whisper/datasets/mms/data_to_i2r/mms_set_1/manifest.json
    # python3 lid_inference.py --model /whisper/small.pt --manifest /whisper/datasets/mms/mms_silence_removed/mms_batch_14/manifest.json --save_path /whisper/datasets/mms/mms_silence_removed/mms_batch_14/pred_manifest.json
    model = whisper.load_model(args.model, download_root='/whisper')

    # check the model state dict
    # print(f'model state dict - {model.state_dict}')

    pred_items = detect_language_from_manifest(args.manifest, model)
    # save_manifest(pred_items, '/whisper/datasets/jtubespeech/ms_3/annotated_data_whisper_ms/pred_manifest.json')
    save_manifest(pred_items, args.save_path)
