#!/bin/bash

# 顺序执行不同百分比的XGBoost训练
# 定义百分比数组
percentages=(1 2 3 4 5 10 15 20 25 30 35 40 45 50)

# 基础路径
FEATURE_PATH="../datasets/he_amazon/UserFeature.txt"
BASE_PATH="../datasets/he_amazon/Training_Testing"

echo "开始顺序执行ISR训练..."
echo "总共需要执行 ${#percentages[@]} 个百分比"

# 循环执行每个百分比
for percent in "${percentages[@]}"
do
    echo "========================================="
    echo "开始执行 ${percent}percent 训练..."
    echo "时间: $(date)"
    echo "========================================="
    
    TRAIN_CSV="${BASE_PATH}/${percent}percent/train_4.csv"
    TEST_CSV="${BASE_PATH}/${percent}percent/test_4.csv"
    
    # 检查文件是否存在
    if [[ ! -f "$TRAIN_CSV" ]]; then
        echo "错误: 训练文件不存在: $TRAIN_CSV"
        continue
    fi
    
    if [[ ! -f "$TEST_CSV" ]]; then
        echo "错误: 测试文件不存在: $TEST_CSV"
        continue
    fi

    # 执行训练命令
    python -u main_he_amazon.py \
        --edge_list ../datasets/he_amazon/J01Network.txt \
        --feature_path "$FEATURE_PATH" \
        --train_csv "$TRAIN_CSV" \
        --test_csv "$TEST_CSV"
    
    # 检查执行结果
    if [[ $? -eq 0 ]]; then
        echo "${percent}percent 训练完成成功!"
    else
        echo "错误: ${percent}percent 训练失败!"
        echo "是否继续执行下个百分比? (y/n)"
        read -r response
        if [[ "$response" != "y" && "$response" != "Y" ]]; then
            echo "用户选择停止执行"
            exit 1
        fi
    fi
    
    echo ""
done

echo "========================================="
echo "所有百分比训练执行完毕!"
echo "结束时间: $(date)"
echo "========================================="