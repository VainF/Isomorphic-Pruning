torchrun --nproc_per_node=4 --master_port=24358 train.py\
    --model "output/pruned/mobilenet_v2_0.15G.pth" \
    --epochs 300 --batch-size 512 --wd 0.00002 --lr=0.036 \
    --lr-scheduler=cosineannealinglr \
    --lr-warmup-epochs=0 \
    --output-dir output/finetuned/mobilenet_v2_0.15G \
    --interpolation bilinear \
    --amp \