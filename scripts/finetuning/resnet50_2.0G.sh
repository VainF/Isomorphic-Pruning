torchrun --nproc_per_node=4 --master_port 24552 train.py \
    --data-path data/imagenet \
    --model output/pruned/resnet152_4.0G.pth \
    --epochs 100 \
    --batch-size 64 \
    --opt sgd \
    --lr-scheduler steplr \
    --lr-step-size 30 \
    --lr 0.04 \
    --weight-decay 1e-4 \
    --amp \
    --output output/finetuned/resnet152_4.0G \
    --val-resize 256 \
    --interpolation bilinear \

    