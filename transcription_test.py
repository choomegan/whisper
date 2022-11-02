import whisper
import os
from tqdm import tqdm

model = whisper.load_model(name='/whisper/whisper-large-model/large.pt', download_root='/whisper/whisper-large-model')

for audio in tqdm(os.listdir('/whisper/datasets/mms/mms_silence_removed/mms_batch_train/mms_20220417/CH 73/ms/')):
    if audio.endswith('.wav'):
        print(f'audio file - {audio}')
        result = model.transcribe(f'/whisper/datasets/mms/mms_silence_removed/mms_batch_train/mms_20220417/CH 73/ms/{audio}')
        print(result["text"])

