for i in {0..200}
do
    CUDA_VISIBLE_DEVICES=5 nohup python -u run_baselines.py \
        --dataset he_amazon \
        --model GCN_update \
        --epoch 300 \
        --strategy uncertainty \
        --file_io 1 \
        --lr 0.001 \
        --hidden 64 \
        --test_percents 5percent \
        --data_path ../datasets/he_amazon/ \
        --sample_global >> nohup2.out  2>&1
done &
