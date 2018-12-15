from __future__ import division, print_function
from src.python.Config.Config import FILE_ROOT_DIRECTORY
import pandas as pd
import csv
import os

# 本程序的目的是将三个提取出来的特征整合到一起，首先进行排序，并比较其中的tdid的数量

# dataAppPD = pd.read_csv('src/DataCleaning/AppFeatures.csv')
# print(dataAppPD)
# dataDevPD = pd.read_csv('src/DataCleaning/DevFeatures.csv')
# print(dataDevPD)
# dataAoiPD = pd.read_csv('src/DataCleaning/AoiFeatures.csv')
# print(dataAoiPD)

all_tdid = []
with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/AppFeaturesSorted.csv'), 'r') as f:
    rows = csv.reader(f)
    row_index = 0
    for row in rows:
        row_index += 1
        if row_index == 1:
            continue
        all_tdid.append(row[0])
print(len(all_tdid))

# 进行判断App和Dev中是否存在不一致的tdid， 测试结果，完全相同，总数为84572
# with open('src/DataCleaning/DevFeaturesSorted.csv', 'r') as f:
#     rows = csv.reader(f)
#     row_index = 0
#     for row in rows:
#         row_index += 1
#         # print(row)
#         if row_index == 1:
#             continue
#         if row[0] not in all_tdid:
#             print(row[0])

# 进行判断Aoi和Dev中是否存在不一致的tdid， 测试结果，完全相同，总数为78477
with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/AoiFeaturesSorted.csv'), 'r') as f:
    rows = csv.reader(f)
    row_index = 0
    for row in rows:
        row_index += 1
        # print(row)
        if row_index == 1:
            continue
        if row[0] not in all_tdid:
            print(row[0])
    print(row_index)

# 接下来将Aoi的tdid进行补全
AoiFeaturesAppend = []
tdid_Aoi = []
with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/AoiFeatures.csv'), 'r') as f:
    rows = csv.reader(f)
    row_index = 0
    for row in rows:
        row_index += 1
        if row_index != 1:
            tdid_Aoi.append(row[0])
        AoiFeaturesAppend.append(row)
print(len(AoiFeaturesAppend[0]))
for tdid in all_tdid:
    if tdid not in tdid_Aoi:
        append_line = []
        append_line.append(tdid)
        for i in range(len(AoiFeaturesAppend[0]) - 1):
            append_line.append(0)
        AoiFeaturesAppend.append(append_line)
dataAppPDFeaturesAppend = pd.DataFrame(data=AoiFeaturesAppend)
print(len(dataAppPDFeaturesAppend))
dataAppPDFeaturesAppend.to_csv(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/AoiFeaturesAppend.csv'),
                               encoding='utf-8_sig')
