python -m open_clip_train.main \
    --device mps \
    --dataset-type webdataset \
    --train-data "./cc3m/00000.tar" \
    --train-num-samples 66 \
    --workers 0 \
    --batch-size 2 \
    --epochs 1