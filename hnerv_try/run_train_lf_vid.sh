#!/bin/sh

dataset_name=$1


CUDA_VISIBLE_DEVICES=0 python train_nerv_lightfield_vid.py --outf 0417 \
   --data_path /data/ysj/dataset/Neulf_video_e1k/${dataset_name} \
   --val_path /data/ysj/dataset/Neulf_video_e1k/${dataset_name} \
   --vid ${dataset_name} \
   --conv_type convnext pshuffel --act swish --norm none --crop_list 432_864  \
   --resize_list -1 --loss L2 \
   --enc_strds 4 3 3 3 2 --enc_dim 4_2 --dec_strds 4 3 3 3 2 --ks 0_1_5 --reduce 1.2 --num_blks 1_1 \
   --modelsize 3  -e 1600 --eval_freq 30 --lower_width 12 -b 1 --lr 0.001 \
   --eval_fps --use_lightfield_val --grid_size 9


# /data/ysj/teacher_output/Exp_knights_8_256
# ./data/knights
# bracelet 4 4 2 2 1 / 5_8 / 320_512