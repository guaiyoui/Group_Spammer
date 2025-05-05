nohup python -u xgb.py --feature_path ../datasets/amazon_cn/UserFeature.txt --train_csv ../datasets/amazon_cn/Training_Testing/5percent/train_5.csv --test_csv ../datasets/amazon_cn/Training_Testing/5percent/test_5.csv >> logs/xgb.log 2>&1 &


python -u xgb.py --feature_path ../datasets/Yelp/UserFeature.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u xgb.py --feature_path ../datasets/Yelp/UserFeature_Unnormalized.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u xgb.py --feature_path ../datasets/Yelp/UserFeature.txt --train_csv ../datasets/Yelp/Training_Testing/50percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/50percent/test_4.csv

python -u xgb.py --feature_path ../datasets/Yelp/UserFeature_ori.txt --train_csv ../datasets/Yelp/Training_Testing/5percent/train_4.csv --test_csv ../datasets/Yelp/Training_Testing/5percent/test_4.csv

python -u xgb.py --feature_path ../datasets/he_amazon/UserFeature.txt --train_csv ../datasets/he_amazon/Training_Testing/5percent/train_4.csv --test_csv ../datasets/he_amazon/Training_Testing/5percent/test_4.csv

python -u xgb.py --feature_path ../datasets/he_amazon/UserFeature_noID.txt --train_csv ../datasets/he_amazon/Training_Testing/5percent/train_4.csv --test_csv ../datasets/he_amazon/Training_Testing/5percent/test_4.csv