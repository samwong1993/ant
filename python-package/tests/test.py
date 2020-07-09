import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset

# feature_names = 'age, marital, address, ed, employ, gender, reside, tollfree, equip, callcard, wireless, multline'.split(', ')
feature_names = ['rank', 'review', 'review_increase',
       'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 'price_6',
       'price_7', 'price_8', 'price_9', 'price_10', 'price_11', 'price_12',
       'sales_1', 'sales_2', 'sales_3', 'sales_4', 'sales_5', 'sales_6',
       'sales_7', 'sales_8', 'sales_9', 'sales_10', 'sales_11', 'sales_12']
label_name = ['label']

df_test = pd.read_csv('D:/OneDrive - HV/code/AIFT code/ant/测试集评分卡.csv')
df_train = pd.read_csv('D:/OneDrive - HV/code/AIFT code/ant/训练集评分卡.csv')

df_train_x = df_train[feature_names]
train_y = df_train[label_name].values.T.tolist()[0]

df_test_x = df_test[feature_names]
test_y = df_test[label_name].values.T.tolist()[0]

train_dataset = Dataset(df_train_x, label=train_y, free_raw_data=False)
valid_dataset = Dataset(df_test_x, label=test_y, free_raw_data=False)

params = {
    'objective': 'binary',
    'boosting': 'rf',
    'metric': 'auc',
    'num_iteration': 1000,
    'learning_rate': 0.01,
    'num_leaves': 20,
    # 'verbose': -1,
    'seed': 7,
    'mode': 'sequential_covering',
    'min_data_in_leaf': 500,

    'lambda_l2': 0.1,
    'lambda_l1': 0.1,
    'max_bin': 255,
    'max_depth': 5,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'min_data_in_bin': 1,
    'rule_top_n_each_tree': 2,
    'num_stop_threshold': 10000,
    'categorical_feature': '0,1'
}


booster, rules = lgb.train(params, train_dataset)
# booster = lgb.train(params, train_dataset)

dmodel = booster.dump_model()
str_model = booster.model_to_string()

pred = booster.predict(train_dataset.get_data(), pred_leaf=True)
booster.save_model('model.txt')
paths = booster.get_leaf_path(missing_value=True)
dataframe = pd.DataFrame(rules)
dataframe.to_csv('output.csv')
print(pred)

TP = 0
FP = 0
TN = 0
FN = 0
num0 = []
num1 = []
for i in range(len(rules)):
    num0.append(rules[i]['rule_info'][0][0])
    num1.append(rules[i]['rule_info'][0][1])
for i in range(len(rules)-1):
    TP = TP + rules[i]['rule_info'][0][1]
    FP = FP + rules[i]['rule_info'][0][0]
    TN = sum(num0[i+1:])
    FN = sum(num1[i+1:])
    train_precision = TP / (TP + FP + 1e-10)
    train_recall = TP / (TP + FN + 1e-10)
    train_accuracy = (TP + TN)/(TP + FN + FP + TN)
    print(f"Number of rules: {i}")
    print('TP:{} FP:{} TN:{} FN:{}'.format(TP,FP,TN,FN))
    print('Train set: Positive:{} Negative:{}'.format(sum(num1),sum(num0)))
    print('train_precision:{:.2f}% train_recall:{:.2f}% train_accuracy:{:.2f}%'.format(100*train_precision,100*train_recall,100*train_accuracy))

import os
os.environ["PATH"] += os.pathsep + 'd:/Graphviz/bin'
graph = lgb.create_tree_digraph(booster, tree_index=0)
graph.render(view=True)
