# -*- utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file):
    ##reading data
    credit_data = pd.read_csv(file, encoding="utf8")

    # print(credit_data)

    X = credit_data.drop(labels='Class', axis=1)  # Features
    y = credit_data.loc[:, 'Class']  # Response

    scaler_time = StandardScaler()
    scaler_amount = StandardScaler()
    # scaling time
    scaled_time = scaler_time.fit_transform(credit_data[['Time']])
    flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
    scaled_time = pd.Series(flat_list1)

    # scaling amount
    scaled_amount = scaler_amount.fit_transform(credit_data[['Amount']])
    flat_list1 = [item for sublist in scaled_amount.tolist() for item in sublist]
    scaled_amount = pd.Series(flat_list1)

    # concatenating newly created columns w original credit_data
    credit_data = pd.concat([credit_data, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')],
                            axis=1)

    # dropping old amount and time columns
    credit_data.drop(['Amount', 'Time'], axis=1, inplace=True)
    # X_final = credit_data.drop(labels='Class', axis=1)  # Features
    # y_final = credit_data.loc[:, 'Class']  # Response
    mask = np.random.rand(len(credit_data)) < 0.9
    train = credit_data[mask]
    test = credit_data[~mask]
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    # how many random samples from normal transactions do we need?
    no_of_frauds = train.Class.value_counts()[1]
    # print(train.Class.value_counts()[1])
    # print(train.Class.value_counts())

    #print('There are {} fraudulent transactions in the train data.'.format(no_of_frauds))
    # randomly selecting 442 random non-fraudulent transactions
    non_fraud = train[train['Class'] == 0]
    fraud = train[train['Class'] == 1]
    selected_1 = fraud.sample(350)
    selected_2 = fraud.sample(350)
    selected_3 = fraud.sample(350)
    selected_4 = fraud.sample(350)
    selected_5 = fraud.sample(350)
    selected_6 = fraud.sample(350)
    selected_7 = fraud.sample(350)
    selected_8 = fraud.sample(350)
    selected_9 = fraud.sample(350)
    selected_10 = fraud.sample(350)
    selected_11 = fraud.sample(350)
    selected_12 = fraud.sample(350)
    selected_13 = fraud.sample(350)
    selected_14 = fraud.sample(350)
    selected = non_fraud.sample(20000)
    selected_dev = non_fraud.sample(2000)
    selected_test = non_fraud.sample(2000)

    # selected = non_fraud.sample(no_of_frauds)
    #
    # # concatenating both into a subsample data set with equal class distribution
    selected_1.reset_index(drop=True, inplace=True)
    selected_2.reset_index(drop=True, inplace=True)
    selected_3.reset_index(drop=True, inplace=True)
    selected_4.reset_index(drop=True, inplace=True)
    selected_5.reset_index(drop=True, inplace=True)
    selected_6.reset_index(drop=True, inplace=True)
    selected_7.reset_index(drop=True, inplace=True)
    selected_8.reset_index(drop=True, inplace=True)
    selected_9.reset_index(drop=True, inplace=True)
    selected_10.reset_index(drop=True, inplace=True)
    selected_11.reset_index(drop=True, inplace=True)
    selected_12.reset_index(drop=True, inplace=True)
    selected_13.reset_index(drop=True, inplace=True)
    selected_14.reset_index(drop=True, inplace=True)



    # fraud.reset_index(drop=True, inplace=True)
    subsample = pd.concat([selected_1,selected_2,selected_3,selected_4,selected_5,selected_6,
                           selected_7,selected_8,selected_9,selected_10,
                           selected_11,selected_12,selected_13,selected_14,
                           selected])


    sumsample_dev = pd.concat([ selected_3,selected_2,selected_dev])
    sumsample_test = pd.concat([selected_7, selected_6, selected_test])

    subsample = subsample.sample(frac=1).reset_index(drop=True)
    sumsample_dev = sumsample_dev.sample(frac=1).reset_index(drop=True)
    sumsample_test = sumsample_test.sample(frac=1).reset_index(drop=True)

    X_final = subsample.drop(labels='Class', axis=1)  # Features
    y_final = subsample.loc[:, 'Class']  # Response

    X_final_dev = sumsample_dev.drop(labels='Class', axis=1)  # Features
    y_final_dev = sumsample_dev.loc[:, 'Class']  # Response

    X_final_test = sumsample_test.drop(labels='Class', axis=1)  # Features
    y_final_test = sumsample_test.loc[:, 'Class']  # Response

    return X_final,y_final, X_final_dev,y_final_dev, X_final_test, y_final_test


# print(load_data("./data/creditcard.csv"))