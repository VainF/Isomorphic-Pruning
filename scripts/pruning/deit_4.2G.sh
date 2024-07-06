python prune.py --data-path data/imagenet \
    --model deit_base_distilled_patch16_224.fb_in1k \
    --pruning-type taylor \
    --pruning-ratio 0.5 \
    --head-pruning-ratio 0.5 \
    --head-dim-pruning-ratio 0.25 \
    --global-pruning \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --taylor-batchs 50 \
    --save-as output/pruned/deit_4.2G.pth \
    --val-resize 256