#!/bin/bash

DTU_PATH="/data/yfu314/dtu"

DATASET="$DTU_PATH/DTU_TRAIN"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--num_src_view 4 --lr 0.001 --lr_feature 0.001 --cosine_lr --min_lr 3e-7 \
--max_iters 200000 --batch_size 1 --weight_rgb 1.0 --weight_depth 1.0 \
--train_ray_num 1024 --volume_reso 144 --root_dir=$DATASET --logdir=./outputs/pretrain
