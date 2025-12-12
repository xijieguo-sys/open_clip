# python -m open_clip_train.main \
#     --model ViT-B-32 \
#     --pretrained openai \
#     --imagenet-v2 /Users/xijieguo/Desktop/ \
#     --zeroshot-frequency 50

# python -m open_clip_train.main \
#     --model ViT-B-32 \
#     --pretrained /Users/xijieguo/Desktop/epoch_32.pt \
#     --imagenet-r /Users/xijieguo/Desktop/imagenet-r \
#     --zeroshot-frequency 1 \
#     --seed 0

python -m open_clip_train.main \
    --model ViT-B-32 \
    --pretrained /Users/xijieguo/Desktop/epoch_32.pt \
    --imagenet-c /Users/xijieguo/Desktop/blur \
    --imagenet-c-corruption glass_blur \
    --imagenet-c-severity 1 \
    --zeroshot-frequency 1 \
    --seed 0