stanford_path="/data/NeuLF_rgb/stanford_half" #?��?��?��?�� 경로


test_day="250303_CoordSR_use_combine" #?��?��결과 ?��?�� (json?��?�� ) 중간 ?��름을 결정?��?��?��.
result_path="/data/result/${test_day}" #?��?�� 결과 ?��?�� ????�� 경로 , 기본?��?��로는 최상?�� data ?��?��?�� ?��?�� ?��?���? �??��?��?��?��?��?��?��.

teacher_header="/data/result/250227_teacher_/250227_teacher__relu_d8_w256_cd2_cd256_R2_8192_decom_dim_us_lr0.0005_"
teacher_footer="_skip_connection_1"

coordx_header="/data/result/250227_CoordX_down_scale/250227_CoordX_down_scale_relu_d0_w128_cd8_cd256_R1_8192_decom_dim_us_lr0.0005_"

depths=("0")
widths=("128") 
#coord_depths=("8" "12" )
coord_depths=("8")
#coord_widths=("256" "512")
coord_widths=("256")
epoch="500"

#datasets=( "knights" )
datasets=( "tarot"   "chess"  )
#datasets=( "bracelet" )
#batch_sizes=("8192" "4096")
batch_sizes=("2")
Rs=("1")

decom_dims=("us")
#lrs=("0.0005" "0.00025" )
lrs=("0.0005" )

nomlin=("relu" )
skip_connections=("1" )


res_depths=("4")


res_widths=("64")

scales=("8")
#scales=("2")
#scales=("8" "4" "2")
#scales=("8" "4" "2")

after_network_types=("rgb" "feature")
after_network_types=("feature")
for R in "${Rs[@]}"; do
    for depth in "${depths[@]}"; do
        for width in "${widths[@]}"; do
            for coord_depth in "${coord_depths[@]}"; do
                for coord_width in "${coord_widths[@]}"; do
                    for dataset in "${datasets[@]}"; do
                        for nonlin in "${nomlin[@]}"; do  
                            for lr in "${lrs[@]}"; do  
                                for batch_size in "${batch_sizes[@]}"; do
                                    for decom_dim in "${decom_dims[@]}"; do
                                        for skip_connection in "${skip_connections[@]}"; do
                                            for res_depth in "${res_depths[@]}"; do
                                                for res_width in "${res_widths[@]}"; do
                                                    for after_network_type in "${after_network_types[@]}"; do
                                                        for scale in "${scales[@]}"; do
                                                            echo "Processing $dataset , $nonlin , $depth , $width , $lr "
                                                            echo "Processing before : $coord_depth , $coord_width ,$decom_dim"
                                                            echo "Processing after : $res_depth , $res_width"
                                                            python run_CoordSR_combine.py \
                                                                --data_set "${dataset}" \
                                                                --data_dir "${stanford_path}/${dataset}" \
                                                                --pseudo_data_path "${teacher_header}${dataset}${teacher_footer}/pseudo_data" \
                                                                --coordx_model_path "${coordx_header}${dataset}_scale_${scale}" \
                                                                --exp_dir "${result_path}/${test_day}_${nonlin}_d${depth}_w${width}_cd${coord_depth}_cd${coord_width}_R${R}_${batch_size}_decom_dim_${decom_dim}_lr${lr}_${dataset}_cnn_depth_${res_depth}_cnn_width_${res_width}_after_network_type_${after_network_type}_cnn_type_sr_pixel_shuffle_scale_${scale}" \
                                                                --depth $depth \
                                                                --width $coord_width \
                                                                --coord_depth $coord_depth \
                                                                --coord_width $coord_width \
                                                                --whole_epoch $epoch \
                                                                --test_freq 10 \
                                                                --nonlin $nonlin \
                                                                --lr $lr \
                                                                --benchmark \
                                                                --batch_size $batch_size \
                                                                --gpu 1 \
                                                                --decom_dim $decom_dim \
                                                                --R $R \
                                                                --skip_connection $skip_connection \
                                                                --save_ckpt_path 100 \
                                                                --loadcheckpoint \
                                                                --res_depth $res_depth \
                                                                --res_width $res_width \
                                                                --after_network_type $after_network_type \
                                                                --cnn_type sr_pixel_shuffle \
                                                                --sr_scale $scale

                                                            python asem_json.py "/data/result/${test_day}" "result_json/${test_day}"
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done 
done
