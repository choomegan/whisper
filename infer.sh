# BATCH=22

# python3 lid_inference.py \
#     --model /models/small.pt \
#     --manifest /datasets/mms/mms_silence_removed/mms_batch_${BATCH}/manifest.json \
#     --save_path /datasets/mms/mms_silence_removed/mms_batch_${BATCH}/pred_manifest.json

BATCH=22

python3 lid_inference.py \
    --model /models/large-v2.pt \
    --manifest /datasets/mms/mms_silence_removed/mms_batch_${BATCH}/manifest_0.4_0.5.json \
    --save_path /datasets/mms/mms_silence_removed/mms_batch_${BATCH}/pred_manifest_0.4_0.5.json