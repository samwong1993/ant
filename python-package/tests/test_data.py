#test rules set
import pandas as pd
import test_backfit
rules_all = pd.read_csv('output.csv')
rules = rules_all.iloc[:,:]['rule']
data_all = pd.read_csv('newdata1.csv')
for num in range(1,len(rules)):
    test_backfit.test_backfit(data_all,rules_all,num)

