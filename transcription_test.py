import whisper
import os
from tqdm import tqdm

model = whisper.load_model(name='/whisper/whisper-large-model/large.pt', download_root='/whisper/whisper-large-model')

ROOT_FOLDER = '/whisper/datasets/mms/mms_silence_removed/mms_batch_train/mms_20220417/CH 10/'
LANGUAGE = 'ms'
PRED_FILENAME = 'pred_annotation.txt'

pred_list = []

for audio in tqdm(os.listdir(os.path.join(ROOT_FOLDER, LANGUAGE))):
    if audio.endswith('.wav'):
        print(f'audio file - {audio}')
        # result = model.transcribe(f'/whisper/datasets/mms/mms_silence_removed/mms_batch_train/mms_20220417/CH 10/ms/{audio}')
        result = model.transcribe(os.path.join(ROOT_FOLDER, LANGUAGE, audio))
        print(result["text"])

        pred_list.append(audio)
        pred_list.append(result["text"])

# export to a text file
print('Writing to text file now')
with open(os.path.join(ROOT_FOLDER, LANGUAGE, PRED_FILENAME), 'w+') as f:
    for line in tqdm(pred_list):
        f.write(f'{line}\n')