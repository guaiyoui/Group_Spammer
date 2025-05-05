#!/bin/bash
echo "\n####### running time is $(date) #######\n" >> ./logs/Active_Spammer.txt

# 定义各个权重参数的取值数组（请根据实际需求调整数值）
weight_loss_subgraph_vals=(0.1 0.5 1.0)
weight_loss_reconstruction_vals=(0.1 0.5 1.0)
weight_kl_loss_vals=(0.5 0.1 0.05)
weight_t_loss_vals=(1.0 0.1 0.01)

# 遍历所有权重参数组合
for wls in "${weight_loss_subgraph_vals[@]}"; do
    for wlr in "${weight_loss_reconstruction_vals[@]}"; do
        for wkl in "${weight_kl_loss_vals[@]}"; do
            for wt in "${weight_t_loss_vals[@]}"; do
                
                echo "Running baseline with weights: subgraph=$wls, reconstruction=$wlr, kl=$wkl, t=$wt, with global sampling"
                CUDA_VISIBLE_DEVICES=4 nohup python run_ComGA.py \
                    --dataset he_amazon \
                    --model GCN \
                    --epoch 300 \
                    --strategy uncertainty \
                    --file_io 1 \
                    --lr 0.01 \
                    --test_percents 5percent \
                    --hidden 64 \
                    --data_path ../datasets/he_amazon/ \
                    --sample_global \
                    --weight_loss_subgraph $wls \
                    --weight_loss_reconstruction $wlr \
                    --weight_kl_loss $wkl \
                    --weight_t_loss $wt >> ./logs/ComGA.txt 2>&1
            done
        done
    done
done &
