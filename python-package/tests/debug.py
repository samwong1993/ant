import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset


# feature_names = 'age, marital, address, ed, employ, gender, reside, tollfree, equip, callcard, wireless, multline'.split(', ')
feature_names = 'region, logtoll'.split(', ')
label_name = ['churn']
df_all = pd.read_csv('/Users/lsw/WorkRoot/LocalNotebook/run_python/interpretML/dataset/pai_telco_demo_data/pai_telco_demo_data_all.csv')

df_test = df_all[feature_names + label_name].iloc[:10000, :]
df_train = df_all[feature_names + label_name].iloc[10000:, :]

df_train_x = df_train[feature_names]
train_y = df_train[label_name].values.T.tolist()[0]

df_test_x = df_test[feature_names]
test_y = df_test[label_name].values.T.tolist()[0]

train_dataset = Dataset(df_train_x, label=train_y)
valid_dataset = Dataset(df_test_x, label=test_y)


params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    'num_iteration': 40,
    'learning_rate': 0.2,
    'num_leaves': 7,
    # 'verbose': -1,
    'seed': 7,
    # 'gam_mode': True,
    'boost_from_average': True,

    'lambda_l2': 1.0,
    'lambda_l1': 1.0,
    'max_bin': 255,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'min_data_in_bin': 1,
    # 'categorical_feature': '0'
}


mock_data = {
    'feat': [5.0] * 100
}
mock_data = pd.DataFrame(mock_data)
mock_label = [1] * 50 + [0] * 50
mock_data = Dataset(mock_data, label=mock_label)

booster = lgb.train(params, train_dataset)

paths = booster.get_leaf_path()

print(paths)

from lightgbm.basic import global_sync_up_by_sum, global_sync_up_by_max, global_sync_up_by_min

v = global_sync_up_by_max(10)

print(v)

