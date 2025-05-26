#!/bin/sh

dataset_name=$1
mlp_depth=$2
mlp_width=$3

CUDA_VISIBLE_DEVICES=1 python train_nerv_lightfield.py --outf 0417 \
   --data_path /data/ysj/teacher_output/Exp_${dataset_name}_${mlp_depth}_${mlp_width} \
   --val_path /data/ysj/dataset/stanford_half/${dataset_name}/images \
   --vid ${dataset_name}_teacher_${mlp_depth}_${mlp_width} \
   --conv_type convnext pshuffel --act swish --norm none --crop_list 512_512  \
   --resize_list -1 --loss L2  --fc_hw 2_2 \
   --dec_strds 4 4 4 2 2 --ks 0_1_5 --reduce 1.2 --num_blks 1_1 \
   --modelsize 3  -e 1600 --eval_freq 30 --lower_width 32 -b 1 --lr 0.001 \
   --embed pe_2_16 \
   --eval_fps --use_lightfield_val --grid_size 17


# /data/ysj/teacher_output/Exp_knights_8_256
# ./data/knights
# bracelet 4 4 2 2 1 / 5_8 / 320_512