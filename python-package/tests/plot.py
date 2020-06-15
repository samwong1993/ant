#rearrange data from txt to csv
import pandas as pd
f = open("./results.txt")
line = f.readline()
AUC = []
precision = []
recall = []
AUC.append(float(line[4:]))
while line:
    print(line)
    line = f.readline()
    if line[0:3] == 'AUC':
        AUC.append(float(line[4:]))
    elif line[0:3] == 'pre':
        precision.append(float(line[10: 15]))
        recall.append(float(line[23:28]))
f.close()
name = ['AUC','precision','recall']
lists = [AUC,precision,recall]
print(lists)
df = pd.DataFrame(dict(zip(name, lists)))
print(df)
df.to_csv('./data.csv',encoding='gbk')
