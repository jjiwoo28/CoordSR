stanford_path="/data/NeuLF_rgb/stanford_half" #?��?��?��?�� 경로


test_day="250317_MLP_debug" #?��?��결과 ?��?�� (json?��?�� ) 중간 ?��름을 결정?��?��?��.
result_path="/data/result/${test_day}" #?��?�� 결과 ?��?�� ????�� 경로 , 기본?��?��로는 최상?�� data ?��?��?�� ?��?�� ?��?���? �??��?��?��?��?��?��?��.


depths=("8" )
widths=("256" ) 
coord_depths=("0")
coord_widths=("0")
epoch="300"

#datasets=( "gem" "knights" "bracelet" "bunny" "tarot")

datasets=( "bunny"  "bracelet"  "knights" "tarot")
datasets=( "knights" )

batch_sizes=("8192" )
Rs=(  "2" )

decom_dims=("us")
lrs=("0.0005"  )

nomlin=("relu" )
skip_connections=("1" )

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
                                            echo "Processing $dataset , $nonlin , $depth , $width , $lr "
                                            echo "Processing  $coord_depth , $coord_width ,$decom_dim"
                                            python run_MLP.py \
                                                --data_dir "${stanford_path}/${dataset}" \
                                                --exp_dir "${result_path}/${test_day}_${nonlin}_d${depth}_w${width}_cd${coord_depth}_cd${coord_width}_R${R}_${batch_size}_decom_dim_${decom_dim}_lr${lr}_${dataset}_skip_connection_${skip_connection}" \
                                                --depth $depth \
                                                --width $coord_width \
                                                --coord_depth $coord_depth \
                                                --coord_width $coord_width \
                                                --whole_epoch $epoch \
                                                --test_freq 1 \
                                                --nonlin $nonlin \
                                                --lr $lr \
                                                --benchmark \
                                                --batch_size $batch_size \
                                                --gpu 0 \
                                                --decom_dim $decom_dim \
                                                --R $R \
                                                --skip_connection $skip_connection \
                                                --save_ckpt_path 10 \
                                                --render_only

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
