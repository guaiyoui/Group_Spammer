



CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --test_percents 30percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy largest_degrees --file_io 1 --lr 0.01 --hidden 16 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy coreset_greedy --file_io 1 --lr 0.01 --hidden 16 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/


# compare results
CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/amazon_cn/

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset he_amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/amazon_cn/

CUDA_VISIBLE_DEVICES=3 python clustering_al.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/amazon_cn/


CUDA_VISIBLE_DEVICES=4 python clustering_al.py --dataset amazon --epoch 300 --strategy LSCALE --file_io 1 --reweight 0 --lr 0.001 --hidden 64  --feature cat --adaptive 1 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

# CUDA_VISIBLE_DEVICES=0 python LSCALE.py --dataset $i --epoch 300 --strategy LSCALE --file_io 1 --reweight 0 --hidden 100 --feature cat --adaptive 1 --weight_decay 0.000005



echo "Running baseline for $test_percent, no global sampling"
CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --test_percents 5percent --data_path ../he_amazon/data/network/

echo "Running baseline for $test_percent, with global sampling"
CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 64 --test_percents 5percent --data_path ../he_amazon/data/network/ --sample_global

CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 32 --test_percents 10percent --data_path ../he_amazon/data/network/ --sample_global

CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 200 --strategy coreset_greedy --file_io 1 --lr 0.001 --hidden 16 --test_percents 10percent --data_path ../he_amazon/data/network/ --sample_global

CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 32 --test_percents 50percent --data_path ../datasets/he_amazon/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset he_amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global

CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global

CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global

CUDA_VISIBLE_DEVICES=5 python run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global

CUDA_VISIBLE_DEVICES=5 nohup python -u run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global >> nohup.out  2>&1

CUDA_VISIBLE_DEVICES=3 nohup python -u run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global >> nohup1.out  2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u run_baselines.py --dataset he_amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../datasets/he_amazon/ --sample_global