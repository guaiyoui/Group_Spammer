python -u mlp.py --feature_path ../datasets/he_amazon/UserFeature.txt --train_csv ../datasets/he_amazon/Training_Testing/5percent/train_4.csv --test_csv ../datasets/he_amazon/Training_Testing/5percent/test_4.csv

python -u mlp.py --feature_path ../datasets/he_amazon/UserFeature_noID.txt --train_csv ../datasets/he_amazon/Training_Testing/5percent/train_4.csv --test_csv ../datasets/he_amazon/Training_Testing/5percent/test_4.csv


python -u mlp.py --feature_path ../datasets/ali/UserFeature.txt --train_csv ../datasets/ali/Training_Testing/5percent/train_4.csv --test_csv ../datasets/ali/Training_Testing/5percent/test_4.csv


python -u mlp.py --feature_path ../datasets/ali/UserFeature.txt --train_csv ../datasets/ali/Training_Testing/10percent/train_4.csv --test_csv ../datasets/ali/Training_Testing/10percent/test_4.csv

python -u mlp.py --feature_path ../datasets/ali/UserFeature.txt --train_csv ../datasets/ali/Training_Testing/30percent/train_4.csv --test_csv ../datasets/ali/Training_Testing/30percent/test_4.csv

python -u mlp.py --feature_path ../datasets/ali/UserFeature.txt --train_csv ../datasets/ali/Training_Testing/50percent/train_4.csv --test_csv ../datasets/ali/Training_Testing/50percent/test_4.csv