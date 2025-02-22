torchrun --standalone --nproc_per_node=8 convnext_train.py \
    --data-dir data/imagenet \
    --model "convnext_base.fb_in1k" \
    --pruned-model "output/pruned/convnext_8.5G.pth" \
    --epochs 300 \
    --batch-size 128 \
    --opt adamw \
    --lr 2e-3 \
    --grad-accum-steps 2 \
    --weight-decay 0.05 \
    --sched cosine \
    --amp \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 \
    --drop-path 0.4 \
    --drop 0.1 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --output output/convnext_8.5G \
    --model-ema \
    --model-ema-decay 0.9999 \
    --color-jitter 0.4 \
    --interpolation bicubic \
    --train-interpolation bicubic \
    