'''
NOTE: TO DUPLICATE ONE ROOT FOLDER IF YOU WANT TO KEEP THE ORIGINAL COPY, AS IT WILL DELETE THE UNWANTED AUDIO INPLACE
'''

import os
import json
from tqdm import tqdm
from typing import Dict, List



class GetActualLanguageAudio:
    '''
        loads the manifest, checks for the predicted language key, if it matches the target, retain. Else delete the audio, updates the manifest
    '''

    def __init__(self, root_dir: str, input_manifest_dir: str, output_manifest_dir: str, target_language_list: List[str]) -> None:
        '''
            root_dir: the root dir of the audio directory
            input_manifest_dir: the directory of the manifest with all the audio info
            output_manifest_dir: the directory of the manifest with the selected language id
            target_language_list: the list of language id to keep
        '''

        self.root_dir = root_dir
        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir = output_manifest_dir
        self.target_language_list = target_language_list

    def delete_unwanted_audio(self) -> None:

        # load the manifest
        with open(self.input_manifest_dir, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()

        items = [json.loads(line.strip('\r\n')) for line in lines]

        selected_manifest_list = []

        # iterate the dictionaries in the json file
        for entry in tqdm(items):
            if entry['language_pred'] in self.target_language_list:
                selected_manifest_list.append(entry)

            else:
                os.remove(os.path.join(self.root_dir, entry['audio_filepath']))

        # export the new json file
        with open(self.output_manifest_dir, 'w', encoding='utf-8') as fw:
            for item in selected_manifest_list:
                fw.write(json.dumps(item)+'\n')

    def __call__(self):
        return self.delete_unwanted_audio()

if __name__ == '__main__':
    d = GetActualLanguageAudio(root_dir='/whisper/datasets/jtubespeech/ms_1/annotated_data_whisper_ms/', 
                               input_manifest_dir='/whisper/datasets/jtubespeech/ms_1/annotated_data_whisper_ms/pred_manifest.json', 
                               output_manifest_dir='/whisper/datasets/jtubespeech/ms_1/annotated_data_whisper_ms/pred_manifest_updated.json', 
                               target_language_list=['ms', 'id']) 