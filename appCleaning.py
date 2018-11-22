from __future__ import division, print_function
import pandas as pd
import csv


# 功能：对字典中的values进行排序并获取排序后的前n个keys
def order_dict(dicts, n):
    result = []
    result1 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
    return result1


# 本程序中用来对app特征进行提取

# 尝试的第一种方式：特征太多，无法通过stack的方式将数据特征进行行列转换
# dataAppPD = pd.read_csv('/data2/lookalike/AppPD_DROP_DUP.csv')
# dataAppsPD = pd.get_dummies(dataAppPD.reset_index('tdid').stack()).sum(level=0)
# dataAppPD.to_csv('/data2/lookalike/AppPD_DROP_DUP_DUMMY.csv')
# dataAppPD.loc['Row_sum'] = dataAppPD.apply(lambda x: x.sum())
# dataAppPD['Row_sum']['tdid'] = 0
# dataAppPD.to_csv('/home/litianan/lta/lookalike/AppPD_DROP_DUP_DUMMY_SUM.csv')


# 尝试的第二种方式：换一种用字典的方式提取APP的特征
dic = {}
originData = []
with open('src/DataCleaning/AppPD_DROP_DUP.csv', 'r') as f:
    rows = csv.reader(f)
    row_index = 0
    for row in rows:
        originData.append(row)
        row_index = row_index + 1
        if row_index == 1:
            continue
        column_index = 0
        for data in row:
            column_index = column_index + 1
            if column_index == 1:
                continue
            if data != '':
                dic[data] = 1 + dic.get(data, 0)
keys = order_dict(dic, 300)
print(len(keys))
# dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
# print(dic)
dataAppFeatures = []
for rowNum in range(len(originData)):
    dataAppFeaturesItem = []
    originRow = originData[rowNum]
    # print(originRow)
    if rowNum == 0:  # 构造features的标题行
        dataAppFeaturesItem.append(originRow[0])
        for key in keys:
            dataAppFeaturesItem.append(key)
    else:
        dataAppFeaturesItem.append(originRow[0])
        for key in keys:
            if key in originRow:
                dataAppFeaturesItem.append(1)
            else:
                dataAppFeaturesItem.append(0)
    dataAppFeatures.append(dataAppFeaturesItem)
dataAppPDFeatures = pd.DataFrame(data=dataAppFeatures)
print(len(dataAppPDFeatures))
dataAppPDFeatures.to_csv('src/DataCleaning/AppFeatures.csv', encoding='utf-8_sig')
# for row in range
