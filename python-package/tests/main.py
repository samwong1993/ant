import pandas as pd
from lightgbm import Dataset
import random
import pre_train
import backfit
import test_backfit
df_all = pd.read_csv('newdata1.csv')
feature_names = 'LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6'.split(
    ',')
label_name = ['label']
num = 2
N = 30
random.seed(2)
index = [i for i in range(len(df_all))]
slice = random.sample(index, 20000)
diff = list(set(index) - set(slice))
df_train = df_all[feature_names + label_name].iloc[slice, :]
df_Validation = df_all[feature_names + label_name].iloc[diff, :]
df_train_x = df_train[feature_names]
train_y = df_train[label_name].values.T.tolist()[0]
df_Validation_x = df_Validation[feature_names]
Validation_y = df_Validation[label_name].values.T.tolist()[0]
train_dataset = Dataset(df_train_x, label=train_y, free_raw_data=False)
valid_dataset = Dataset(df_Validation_x, label=Validation_y, free_raw_data=False)
rules = pre_train.pre_train(df_train_x,train_y,df_Validation_x,Validation_y)
dataframe = pd.DataFrame(rules)
dataframe.to_csv('0.csv')
index = [1*i for i in range(1,N+1)]
#test original rules
for i in index:
    test_backfit.test_backfit(df_all,dataframe,i)
#test backfitting rules
for iter in range(1,int(N/2+1)):
    rules0, s = backfit.backfit(df_all, slice, rules, dataframe, (iter) * num)
    df_train = df_all[feature_names + label_name].iloc[s, :]
    df_Validation = df_all[feature_names + label_name].iloc[diff, :]
    df_train_x = df_train[feature_names]
    train_y = df_train[label_name].values.T.tolist()[0]
    df_Validation_x = df_Validation[feature_names]
    Validation_y = df_Validation[label_name].values.T.tolist()[0]
    train_dataset = Dataset(df_train_x, label=train_y, free_raw_data=False)
    valid_dataset = Dataset(df_Validation_x, label=Validation_y, free_raw_data=False)
    rules1 = pre_train.pre_train(df_train_x, train_y, df_Validation_x, Validation_y)
    #previous rules plus backfitting rules
    rules = rules0 + rules1
    dataframe = pd.DataFrame(rules)
    dataframe.to_csv(str(iter)+'.csv')
    test_backfit.test_backfit(df_all, dataframe, (iter) * num - 1)
    test_backfit.test_backfit(df_all, dataframe, (iter) * num)
