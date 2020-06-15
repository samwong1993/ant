import rule as rule_dec
import sklearn.metrics
import copy
import random
def backfit(df_all,slice,rules_old,dataframe,num):
    rules_all = dataframe
    rules = rules_all.iloc[:, :]['rule']
    m = len(df_all.iloc[:, :])
    label = df_all.iloc[slice, :]['label']
    results = []
    index = [[] for i in range(len(rules))]
    for i in slice:
        data = df_all.iloc[i, :]
        for j in range(len(rules)):
            out, result = rule_dec.rule_decide(rules[j], data)
            if out == True:
                index[j].append(i)
                results.append(result)
                break
            if j == len(rules):
                print('error')
    results = list(map(float, results))
    label = list(label)
    F = []
    for i in range(len(rules_all)):
        #a = eval(rules_all.iloc[i, :]['rule_info'])
        a = rules_all.iloc[i, :]['rule_info']
        F.append(a[1]['precision'])
    t = copy.deepcopy(F)
    # 求m个最大的数值及其索引
    max_number = []
    max_index = []
    for _ in range(num):#round(len(F) / 2)
        number = max(t)
        t_index = t.index(number)
        t[t_index] = 0
        max_number.append(number)
        max_index.append(t_index)
    t = []
    #按precision的backfitting
    del_index = [i for i in range(len(index))]
    half_index = list(set(del_index) - set(max_index))
    #随机化backfitting
    '''
    del_index = [i for i in range(num-2+1,len(index))]
    max_index = [i for i in range(num-2)]
    max_index = max_index + random.sample(del_index, 2)
    del_index = [i for i in range(len(index))]
    half_index = list(set(del_index) - set(max_index))
    #del_index = [i for i in range(num-2,len(index))]
    #half_index = random.sample(del_index, 2)
    '''
    del_index = [index[i] for i in half_index]
    s = []
    for i in range(len(del_index)):
        s = del_index[i] + s
    rules0 = [rules_old[max_index[i]] for i in range(len(max_index))]
    #rule0为选取的规则，s为剩余数据
    return rules0,s
