torchrun --nproc_per_node=4 train.py \
    --data-path data/imagenet \
    --model output/pruned/resnet101_4.5G.pth \
    --epochs 100 \
    --batch-size 256 \
    --opt sgd \
    --lr-scheduler steplr \
    --lr-step-size 30 \
    --lr 0.04 \
    --weight-decay 1e-4 \
    --amp \
    --output output/finetuned/resnet101_4.5G \
    --val-resize 256 \
    --interpolation bilinear \
