def cal(label,results,index,sum,neg,pos):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in index:
        if label[i] == 1 and results[i] == 1:
            TP = TP + 1
        elif label[i] == 0 and results[i] == 1:
            FP = FP + 1
        elif label[i] == 0 and results[i] == 0:
            TN = TN + 1
        elif label[i] == 1 and results[i] == 0:
            FN = FN + 1
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (pos + 1e-10)
    accuracy = (TP + TN)/(TP + FN + FP + TN + 1e-10)
    return TP,FP,TN,FN,precision,recall,accuracy
