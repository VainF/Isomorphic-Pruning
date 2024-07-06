python prune.py --data-path data/imagenet \
    --model resnet50.tv_in1k \
    --pruning-type taylor \
    --pruning-ratio 0.26 \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --global-pruning \
    --taylor-batchs 50 \
    --save-as output/pruned/resnet50_2.0G.pth  \
    --val-resize 256 \
    --interpolation bicubic \
    --round-to 2 \
    --drop 0.1 