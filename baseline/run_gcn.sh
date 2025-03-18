nohup python -u gcn.py --feature_path ../datasets/amazon_cn/UserFeature.txt --edge_list ../datasets/amazon_cn/J01Network.txt --train_csv ../datasets/amazon_cn/Training_Testing/5percent/train_5.csv --test_csv ../datasets/amazon_cn/Training_Testing/5percent/test_5.csv >> logs/lr.log 2>&1 &


python -u gcn.py --feature_path ../datasets/Yelp/UserFeature.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u gcn.py --feature_path ../datasets/Yelp/UserFeature_ori.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv


python -u gcn.py --feature_path ../datasets/Yelp/UserFeature_emb.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv


python -u gcn.py --feature_path ../datasets/Yelp/UserFeature_node2vec_combined.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv


python -u gcn.py --feature_path ../he_amazon/data/network/product_features.txt --edge_list ../he_amazon/data/network/product_features.txt --train_csv ../he_amazon/data/network/Training_Testing/5percent/train_4.csv --test_csv ../he_amazon/data/network/Training_Testing/5percent/test_4.csv