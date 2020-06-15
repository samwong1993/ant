import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset


# feature_names = 'age, marital, address, ed, employ, gender, reside, tollfree, equip, callcard, wireless, multline'.split(', ')
feature_names = 'reside_null, region, logtoll'.split(', ')
label_name = ['churn']
df_all = pd.read_csv('test.csv')

df_test = df_all[feature_names + label_name].iloc[20000:, :]
df_train = df_all[feature_names + label_name].iloc[:20000, :]

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
    'num_iteration': 40,
    'learning_rate': 0.2,
    'num_leaves': 7,
    # 'verbose': -1,
    'seed': 7,
    'mode': 'sequential_covering',
    'min_data_in_leaf': 200,

    'lambda_l2': 1.0,
    'lambda_l1': 1.0,
    'max_bin': 255,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'min_data_in_bin': 1,
    'rule_top_n_each_tree': 7,
    'num_stop_threshold': 10000,
    'categorical_feature': '0,1'
}


booster, rules = lgb.train(params, train_dataset)
# booster = lgb.train(params, train_dataset)

dmodel = booster.dump_model()
str_model = booster.model_to_string()

pred = booster.predict(train_dataset.get_data(), pred_leaf=True)

paths = booster.get_leaf_path(missing_value=True)

print(pred)

