import pandas as pd
import numpy as np


for i in range(5):
    training_file_name = 'train_'+str(i+1)+'.csv'
    testing_file_name = 'test_'+str(i+1)+'.csv'
    # Load the datasets
    train_df = pd.read_csv('./5percent/'+training_file_name, header=None)
    test_df = pd.read_csv('./5percent/'+testing_file_name, header=None)

    # Randomly select 20% of the rows from train_1.csv
    train_update_df = train_df.sample(frac=0.2, random_state=42)

    # Save the selected 20% rows to train_1_update.csv
    train_update_df.to_csv('./1percent/'+training_file_name, index=False)

    # Get the remaining 80% of the rows from train_1.csv
    remaining_train_df = train_df.drop(train_update_df.index)

    # Ensure both DataFrames have the same columns in the same order
    test_df = test_df[remaining_train_df.columns]

    # Combine the remaining 80% of train_1.csv with test_1.csv (vertically)
    test_update_df = pd.concat([remaining_train_df, test_df], axis=0, ignore_index=True)
    print(test_update_df)
    # Save the combined dataset without trailing commas
    test_update_df.to_csv('./1percent/' + testing_file_name, index=False, lineterminator='\n')