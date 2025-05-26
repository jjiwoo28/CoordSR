CUDA_VISIBLE_DEVICES=1 python train_nerv_all.py --outf 0417_vid --data_path /data/ysj/dataset/Neulf_video_e1k/ambushfight_6 --vid ambushfight_6_vid \
   --conv_type convnext pshuffel --act gelu --norm none --crop_list 432_864  \
   --resize_list -1 --loss L2  --fc_hw 2_4 \
   --dec_strds 4 3 3 3 2 --ks 0_1_5 --reduce 1.2 --num_blks 1_1 \
   --modelsize 3  -e 1600 --eval_freq 30 --lower_width 12 -b 2 --lr 0.001 \
   --embed pe_2_16 \
   --eval_fps