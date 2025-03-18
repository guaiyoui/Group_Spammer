nohup python -u gcn_mrf.py --feature_path ../datasets/amazon_cn/UserFeature.txt --edge_list ../datasets/amazon_cn/J01Network.txt --train_csv ../datasets/amazon_cn/Training_Testing/5percent/train_5.csv --test_csv ../datasets/amazon_cn/Training_Testing/5percent/test_5.csv >> ./logs/gcn_mrf.txt 2>&1 &


python -u gcn_mrf.py --feature_path ../datasets/Yelp/UserFeature.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv


python -u gcn_mrf.py --feature_path ../datasets/Yelp/UserFeature_ori.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u gcn_mrf.py --feature_path ../datasets/Yelp/UserFeature_emb.txt --edge_list ../datasets/Yelp/J01Network.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv