CUDA_VISIBLE_DEVICES=0 python efficient_nvloader_lf.py \
--decoder output/0331/knights_teacher_8_256/1_1_1_pe_2_16_Dim64_16_FC2_2_KS0_1_5_RED1.2_low32_blk1_1_e1600_b1_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext__DEC_pshuffel_4,4,4,2,2_relu1_1/img_decoder.pth \
--dump_dir visualize/knights \
--pe True \
--grid_size 16 --lbase 2.0 --levels 16