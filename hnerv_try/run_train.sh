CUDA_VISIBLE_DEVICES=1 python train_nerv_all.py --outf 0403 --data_path /data/ysj/dataset/bunny --vid bunny \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 1280_720  \
   --resize_list -1 --loss L2  --enc_strds 5 2 2 2 2 --enc_dim 16_9 \
   --dec_strds 5 2 2 2 2 --ks 0_1_5 --reduce 1.2 \
   --modelsize 6  -e 1000 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001 \
   --eval_fps --eval_only