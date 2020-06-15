import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset
from odps import ODPS
import os
import random


class ODPSWriter(object):
    def __init__(self):
        self._o = ODPS('*****',
                       '*****',
                       'sre_mpi_algo_dev',
                       'http://service-corp.odps.aliyun-inc.com/api')

    def open_table(self, table_name, out_cols, drop_if_exists=True):
        if drop_if_exists and self._o.exist_table(table_name):
            table = self._o.get_table(table_name)
            table.drop()
        tmp_table = self._o.create_table(
            table_name, out_cols, if_not_exists=False
        )
        self._odps_table = self._o.get_table(table_name)
        self._writer = self._odps_table.open_writer()

    # batchWrite Records
    def batch_write(self, records):
        self._writer.write(records)

    # Flush & Close table
    def close(self):
        self._writer.close()

    # get table record
    def gen_record(self):
        return self._odps_table.new_record()



# def upload_rule_list(rule_list, table_name, lifecycle=28):
#     if table_name is None:
#         return
#     schema = 'treeid bigint, ruleid string, rulelink string, sql_rules string, ' \
#              'accuracy double, pos_count bigint, neg_count bigint'
#     rule_records = []
#     writer = ODPSWriter()
#     writer.open_table(table_name, schema)
#     for rule in rule_list:
#         record = writer.gen_record()
#         record['treeid'] = rule['tree_id']
#         record['ruleid'] = rule['rule_id']
#         record['rulelink'] = rule['rule_link']
#         record['sql_rules'] = rule['rule']
#         record['accuracy'] = rule['rule_info']['accuracy']
#         record['pos_count'] = rule['rule_info'][1]
#         record['neg_count'] = rule['rule_info'][0]
#         rule_records.append(record)
#     writer.batch_write(rule_records)
#     writer.close()


def upload_rule_list(rule_list, table_name, lifecycle=28):
    if table_name is None:
        return
    schema = 'treeid bigint, ruleid string, rulelink string, sql_rules string, ' \
             'train_accuracy double, valid_accuracy double, train_precision double, valid_precision double, ' \
             'train_recall double, valid_recall double, train_pos_count bigint, valid_pos_count bigint, ' \
             'train_neg_count bigint, valid_neg_count bigint'
    rule_records = []
    writer = ODPSWriter()
    writer.open_table(table_name, schema)
    for rule in rule_list:
        record = writer.gen_record()
        record['treeid'] = rule['tree_id']
        record['ruleid'] = rule['rule_id']
        record['rulelink'] = rule['rule_link']
        record['sql_rules'] = rule['rule']
        record['train_accuracy'] = rule['rule_info'][0]['accuracy']
        record['valid_accuracy'] = rule['rule_info'][1]['accuracy']
        record['train_precision'] = rule['rule_info'][0]['precision']
        record['valid_precision'] = rule['rule_info'][1]['precision']
        record['train_recall'] = rule['rule_info'][0]['recall']
        record['valid_recall'] = rule['rule_info'][1]['recall']
        record['train_pos_count'] = rule['rule_info'][0]['pos_count']
        record['valid_pos_count'] = rule['rule_info'][1]['pos_count']
        record['train_neg_count'] = rule['rule_info'][0]['neg_count']
        record['valid_neg_count'] = rule['rule_info'][1]['neg_count']
        rule_records.append(record)
    writer.batch_write(rule_records)
    writer.close()



# feature_names = 'region, logtoll, logwire'.split(', ')
# feature_names = 'age, marital, address, ed, employ, gender, reside, tollfree, equip, callcard, wireless, multline'.split(', ')
feature_names = 'LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6'.split(',')
label_name = ['label']
# df_all = pd.read_csv('pai_telco_demo_data_append_id.csv')
df_all = pd.read_csv('newdata1.csv')
'''
#iris
df_all = pd.read_csv('iris.csv')
index = [i for i in range(100)]
random.seed(10)
slice = random.sample(index, 50)
diff = list(set(index) - set(slice))
feature_names = 'sepal_length,sepal_width,petal_length,petal_width'.split(',')
df_train = df_all[feature_names + label_name].iloc[:, :]
df_Validation = df_all[feature_names + label_name].iloc[diff, :]
print(df_train)
df_train_x = df_train[feature_names]
train_y = df_train[label_name].values.T.tolist()[0]

df_Validation_x = df_Validation[feature_names]
Validation_y = df_Validation[label_name].values.T.tolist()[0]
'''
'''
pre_pre = []
pre_rec = []
for _ in range(100):
'''
index = [i for i in range(len(df_all))]
random.seed(1)
thres = 5
slice = random.sample(index, 20000)
diff = list(set(index) - set(slice))
df_train = df_all[feature_names + label_name].iloc[slice, :]
df_Validation = df_all[feature_names + label_name].iloc[diff, :]
# df_train = df_all[feature_names + label_name]

df_train_x = df_train[feature_names]
train_y = df_train[label_name].values.T.tolist()[0]

df_Validation_x = df_Validation[feature_names]
Validation_y = df_Validation[label_name].values.T.tolist()[0]


train_dataset = Dataset(df_train_x, label=train_y, free_raw_data=False)
valid_dataset = Dataset(df_Validation_x, label=Validation_y, free_raw_data=False)

#print(valid_dataset.data)
#print(valid_dataset.data.iloc[0, :])

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    'num_iteration': 20,
    'learning_rate': 0.2,
    'num_leaves': 15,
    'max_depth': 5,
    # 'verbose': -1,
    'seed': 7,
    'mode': 'sequential_covering',
    'min_data_in_leaf': 50,

    'lambda_l2': 1.0,
    'lambda_l1': 1.0,
    'max_bin': 255,
    # 'bagging_fraction': 0.9,
    # 'bagging_freq': 1,
    'min_data_in_bin': 1,
    'rule_top_n_each_tree': 5,
    'num_stop_threshold': 10000,
    # 'rule_pos_rate_threshold': 0.5,
    # 'categorical_feature': '0,1'
}


booster, rules = lgb.train(params, train_dataset, valid_sets=valid_dataset)
# booster, rules = lgb.train(params, train_dataset)
dataframe = pd.DataFrame(rules)
dataframe.to_csv('output.csv')
#leaf_preds = booster.predict(train_dataset.data, pred_leaf=True)

#upload_rule_list(rules, table_name='default_credit_card_rules')

#dmodel = booster.dump_model()
#str_model = booster.model_to_string()
booster.save_model('model.txt')
# paths = booster.get_leaf_path(missing_value=True)
#{0: 15442.0, 1: 4558.0}
#{0: 7922.0, 1: 2078.0}
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
