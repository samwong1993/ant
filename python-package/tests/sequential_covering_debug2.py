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
dataframe = pd.DataFrame(rules)
dataframe.to_csv('output.csv')
print(pred)
thres = 5
pos = []
neg = []
label = []
TP = 0
FP = 0
TN = 0
FN = 0
num0 = []
num1 = []
for i in range(len(rules)):
    if rules[i]['rule_info'][0][0]>=thres*rules[i]['rule_info'][0][1]:
        label.append(0)
        num0.append(rules[i]['rule_info'][0][0] + rules[i]['rule_info'][0][1])
        num1.append(0)
    else:
        label.append(1)
        num1.append(rules[i]['rule_info'][0][0] + rules[i]['rule_info'][0][1])
        num0.append(0)
    if label[i] == 1:
        TP = TP + rules[i]['rule_info'][0][1]
        FP = FP + rules[i]['rule_info'][0][0]
    elif label[i] == 0:
        TN = TN + rules[i]['rule_info'][0][0]
        FN = FN + rules[i]['rule_info'][0][1]
train_precision = TP / (TP + FP + 1e-10)
train_recall = TP / (TP + FN + 1e-10)
train_accuracy = (TP + TN)/(TP + FN + FP + TN)
print(num0)
print(num1)
print('TP:{} FP:{} TN:{} FN:{}'.format(TP,FP,TN,FN))
print('Train set: Positive:{} Negative:{}'.format(sum(num1),sum(num0)))
print('train_precision:{:.2f}% train_recall:{:.2f}% train_accuracy:{:.2f}%'.format(100*train_precision,100*train_recall,100*train_accuracy))
TP = 0
FP = 0
TN = 0
FN = 0
Validation_pos = []
Validation_neg = []
for i in range(len(rules)):
    num0[i] = rules[i]['rule_info'][1][0]
    num1[i] = rules[i]['rule_info'][1][1]
    if label[i] == 1:
        TP = TP + num1[i]
        FP = FP + num0[i]
        Validation_pos.append(num0[i]+num1[i])
    elif label[i] == 0:
        TN = TN + num0[i]
        FN = FN + num1[i]
        Validation_neg.append(num0[i]+num1[i])
Validation_precision = TP / (TP + FP + 1e-10)
Validation_recall = TP / (TP + FN + 1e-10)
Validation_accuracy = (TP + TN)/(TP + FN + FP + TN)
#precision = pos / (pos + neg)

print('Validation set: Positive:{} Negative:{}'.format(sum(Validation_pos),sum(Validation_neg)))
print('Validation_precision:{:.2f}% Validation_recall:{:.2f}% Validation_accuracy:{:.2f}%'.format(100*Validation_precision,100*Validation_recall,100*Validation_accuracy))
'''
pre_pre.append(100*Validation_precision)
pre_rec.append(100*Validation_recall)
print(sum(pre_pre)/100)
print(sum(pre_rec)/100)
'''
#print(pred)
#{'rule_info': [{0: 52, 1: 14, 'recall': 0.00307152259763054, 'precision': 0.21212121212121213, 'accuracy': 0.21212121212121213, 'pred': 1, 'score': 0.21212121212121213, 'pos_count': 14.0, 'neg_count': 52.0, 'F': 0.006055363321799307}, {0: 313, 1: 14, 'recall': 0.006737247353224254, 'precision': 0.04281345565749235, 'accuracy': 0.04281345565749235, 'pred': 1, 'score': 0.04281345565749235, 'pos_count': 14.0, 'neg_count': 313.0, 'F': 0.011642411642411643}], 'rule_link': '0->1->2->4->6->14', 'rule': 'case when BILL_AMT1 > 418.5 and (BILL_AMT1 <= 34084.5 or BILL_AMT1 is null) and PAY_AMT3 > 7506.0 and PAY_AMT3 > 12000.5 and PAY_AMT1 > 2062.5 then 0.212121', 'tree_id': 9, 'rule_id': '9_7'}
#print(rules[i]['rule_info'][1]['precision'])

