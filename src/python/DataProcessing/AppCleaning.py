from __future__ import division, print_function

import operator
import pandas as pd
import csv
import os
from filecmp import cmp

from src.python.Config.Config import FILE_ROOT_DIRECTORY


# 功能：对字典中的values进行排序并获取排序后的前n个keys
def order_dict(dicts, n):
    result = []
    # result1 = []
    p = sorted(dicts.items(), key=operator.itemgetter(1))
    p.reverse()
    for i in range(n):
        result.append(p[i][0])
    return result


# 本程序中用来对app特征进行提取

# 尝试的第一种方式：特征太多，无法通过stack的方式将数据特征进行行列转换
# dataAppPD = pd.read_csv('/data2/lookalike/AppPD_DROP_DUP.csv')
# dataAppsPD = pd.get_dummies(dataAppPD.reset_index('tdid').stack()).sum(level=0)
# dataAppPD.to_csv('/data2/lookalike/AppPD_DROP_DUP_DUMMY.csv')
# dataAppPD.loc['Row_sum'] = dataAppPD.apply(lambda x: x.sum())
# dataAppPD['Row_sum']['tdid'] = 0
# dataAppPD.to_csv('/home/litianan/lta/lookalike/AppPD_DROP_DUP_DUMMY_SUM.csv')


# 尝试的第二种方式：换一种用字典的方式提取APP的特征
# d = {"Pierre": 42, "Anne": 33, "Zoe": 24}
# sorted_d = sorted(d.items(), key=lambda x: x[1])
# sorted_d.reverse()
# a = sorted_d[0][0]
dic = {}
originData = []
with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/App.csv'), 'r') as f:
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
tdidInFeature = []  # 用来记录在Feature中该tdid的位置
for rowNum in range(len(originData)):
    if rowNum % 10000 == 0:
        print(rowNum)
    dataAppFeaturesItem = []
    originRow = originData[rowNum]
    # print(originRow)
    if rowNum == 0:  # 构造features的标题行
        dataAppFeaturesItem.append(originRow[0])
        for key in keys:
            dataAppFeaturesItem.append(key)
        dataAppFeatures.append(dataAppFeaturesItem)
    else:
        if originRow[0] in tdidInFeature:  # 如果该tdid已经出现过
            tdid_index = tdidInFeature.index(originRow[0])
            dataAppFeaturesItem = dataAppFeatures[tdid_index + 1]
            for keyId in range(len(keys)):  # 检查每个key是否存在
                if keys[keyId] in originRow:
                    dataAppFeaturesItem[keyId + 1] += 1
            # # 替换掉相应位置处的tdid的数据
            # dataAppFeatures[tdid_index + 1] = dataAppFeaturesItem
        else:  # 如果该tdid没出现过,则添加该tdid
            dataAppFeaturesItem.append(originRow[0])
            tdidInFeature.append(originRow[0])
            for key in keys:
                if key in originRow:
                    dataAppFeaturesItem.append(1)
                else:
                    dataAppFeaturesItem.append(0)
            dataAppFeatures.append(dataAppFeaturesItem)
dataAppPDFeatures = pd.DataFrame(data=dataAppFeatures)
print(len(dataAppPDFeatures))
dataAppPDFeatures.to_csv(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/AppFeatures.csv'),
                         encoding='utf-8_sig')
# for row in range
