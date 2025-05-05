nohup python -u lr.py --feature_path ../datasets/amazon_cn/UserFeature.txt --train_csv ../datasets/amazon_cn/Training_Testing/5percent/train_5.csv --test_csv ../datasets/amazon_cn/Training_Testing/5percent/test_5.csv >> logs/lr.log 2>&1 &


python -u lr.py --feature_path ../datasets/Yelp/UserFeature.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u lr.py --feature_path ../datasets/Yelp/UserFeature_ori.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u lr.py --feature_path ../datasets/Yelp/UserFeature_node2vec.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u lr.py --feature_path ../datasets/Yelp/UserFeature_node2vec_combined.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv


python -u lr.py --feature_path ../he_amazon/data/network/product_features.txt --train_csv ../he_amazon/data/network/Training_Testing/5percent/train_4.csv --test_csv ../he_amazon/data/network/Training_Testing/5percent/test_4.csv

python -u lr.py --feature_path ../datasets/he_amazon/UserFeature.txt --train_csv ../datasets/he_amazon/Training_Testing/5percent/train_4.csv --test_csv ../datasets/he_amazon/Training_Testing/5percent/test_4.csv

python -u lr.py --feature_path ../datasets/he_amazon/UserFeature_noID.txt --train_csv ../datasets/he_amazon/Training_Testing/5percent/train_4.csv --test_csv ../datasets/he_amazon/Training_Testing/5percent/test_4.csv