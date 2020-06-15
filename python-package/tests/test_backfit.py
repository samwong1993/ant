import rule as rule_dec
import AUC
import sklearn.metrics
def test_backfit(df_all,rules_all,num):
    rules = rules_all.iloc[:, :]['rule']
    data_all = df_all
    m = len(data_all.iloc[:, :])
    label = data_all.iloc[20000:, :]['label']
    num0 = [0] * len(rules)
    num1 = [0] * len(rules)
    results = [0 for i in range(len(data_all))]
    index = [[] for i in range(len(rules))]
    for i in range(20000, len(data_all)):
        data = data_all.iloc[i, :]
        for j in range(len(rules)):
            out, result = rule_dec.rule_decide(rules[j], data)
            if out == True:
                index[j].append(i)
                results[i] = result
                if j<num:
                    num1[j] = num1[j] + 1
                    results[i] = 1
                else:
                    num0[j] = num0[j] + 1
                    results[i] = 0
                break
            if j == len(rules):
                print('error')
    results = list(map(float, results))
    acc_index = []
    for i in range(len(index)):
        acc_index = acc_index + index[i]
    test_lab = []
    test_pre = []
    for i in acc_index:
        test_lab.append(label[i])
        test_pre.append(results[i])
    test_auc = sklearn.metrics.roc_auc_score(test_lab, test_pre)
    print('AUC:',test_auc)
    sum = len(label)
    pos = 0
    neg = 0
    for i in acc_index:
        if label[i] == 1:
            pos = pos + 1
        else:
            neg = neg + 1
    TP, FP, TN, FN, precision, recall, accuracy = AUC.cal(label, results, acc_index, sum, neg, pos)
    # print('Number of samples:{},Positive:{},Negative:{}'.format(sum, pos, neg))
    print('TP:{} FP:{} TN:{} FN:{}'.format(TP, FP, TN, FN))
    print('precision:{:.2f}% recall:{:.2f}% accuracy:{:.2f}%'.format(100 * precision, 100 * recall, 100 * accuracy))
    with open('results.txt', 'a') as file_handle:
        #save results to results.txt
        file_handle.write('AUC:'+str(test_auc)[0:6])
        file_handle.write('\n')
        file_handle.write('TP:'+str(TP)+' FP:'+str(FP)+' TN:'+str(TN)+' FN:'+str(FN))
        file_handle.write('\n')
        file_handle.write('precision:'+str(100*precision)[0:5]+' recall:'+str(100*recall)[0:5]+' accuracy:'+str(100 * accuracy)[0:5])
        file_handle.write('\n')
