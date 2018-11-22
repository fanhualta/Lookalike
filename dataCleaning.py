from __future__ import division, print_function
import os
import pandas as pd

# 在本程序中对Dev和Aoi的信息进行处理并成功提取特征，分别为DevPosition和AoiPosition，另一个App信息的处理在appCleaning.py中
dir = 'src/LookalikeOrigin'  # 原始数据的路径
dataDirs = [os.path.join(dir, x) for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]
dataDev = []
dataAoi = []
dataApp = []
for dataDir in dataDirs:
    listDimensions = [os.path.join(dataDir, x) for x in os.listdir(dataDir)]
    for dimension in listDimensions:  # 三个维度的数据信息
        if dimension.endswith('dev'):  # 设备维度的信息
            aoi = [x for x in os.listdir(dimension) if x == 'part-00000']  # 设备维度的文件
            # print(os.path.join(dimension, dev[0]))
            with open(os.path.join(dimension, aoi[0]), 'r') as f:
                data = f.readlines()  # 将txt中所有字符串读入data
                lines = []
                for i in range(len(data)):
                    data[i] = data[i].replace('\n', '')
                    items = data[i].split('\t')
                    line = []
                    for item in items:
                        if item != ' ':
                            line.append(item)
                    lines.append(line)
                # print(len(lines))
                # print(len(dataDev))
                dataDev.extend(lines)
                # print(len(dataDev))
        elif dimension.endswith('aoi'):  # aoi位置信息
            aoi = [x for x in os.listdir(dimension) if x == 'part-00000']  # aoi位置信息的文件
            # print(os.path.join(dimension, dev[0]))
            with open(os.path.join(dimension, aoi[0]), 'r') as f:
                data = f.readlines()  # 将txt中所有字符串读入data
                lines = []
                for i in range(len(data)):
                    data[i] = data[i].replace('\n', '')
                    items = data[i].split('\t')
                    line = []
                    # for item in items:
                    #     if item != ' ':
                    #         line.append(item)
                    # lines.append(line)
                    line.append(items[0])
                    locations = []  # 表示的是所有的地点位置
                    item_locations = items[1].split(' ')  # 将aoi中的位置信息分离开来
                    for item in item_locations:
                        if item != ' ':
                            line.append(item)
                    lines.append(line)
                # print(len(lines))
                # print(len(dataDev))
                dataAoi.extend(lines)
        elif dimension.endswith('app'):  # app信息
            app = [x for x in os.listdir(dimension) if x == 'part-00000']  # app信息文件
            # print(os.path.join(dimension, dev[0]))
            with open(os.path.join(dimension, app[0]), 'r') as f:
                data = f.readlines()  # 将txt中所有字符串读入data
                lines = []
                for i in range(len(data)):
                    data[i] = data[i].replace('\n', '')
                    items = data[i].split('\t')
                    line = []
                    line.append(items[0])
                    item_apps = items[1].split(' ')  # 将app中的信息分离开来
                    for item in item_apps:
                        if item != ' ':
                            line.append(item)
                    lines.append(line)
                dataApp.extend(lines)
# 已将所有的数据加载进来

columnsNameDev = ['tdid', 'device']

# 用pandas 载入这些数据
dataDevPD = pd.DataFrame(columns=columnsNameDev, data=dataDev)
dataAoiPD = pd.DataFrame(data=dataAoi)
dataAoiPD.rename(columns={dataAoiPD.columns[0]: 'tdid'}, inplace=True)
dataAppPD = pd.DataFrame(data=dataApp)
dataAppPD.rename(columns={dataAppPD.columns[0]: 'tdid'}, inplace=True)
dataDevPD.to_csv('src/DataCleaning/Dev.csv', encoding='utf-8_sig')
dataAoiPD.to_csv('src/DataCleaning/Aoi.csv', encoding='utf-8_sig')
dataAppPD.to_csv('src/DataCleaning/App.csv', encoding='utf-8_sig')

# 开始处理机型信息
dataDevPD['IOS'] = dataDevPD.device.apply(lambda x: 1 if 'iPhone' in x or 'iPad' in x or 'iPod' in x else 0)  # 加入ios列
dataDevPD['Android'] = dataDevPD.device.apply(
    lambda x: 0 if 'iPhone' in x or 'iPad' in x or 'iPod' in x else 1)  # 加入Android列
dataDevPD.pop('device')  # 删除device列
dataDevPD.set_index('tdid')
dataDevPD = dataDevPD.drop_duplicates('tdid')
dataDevPD.sort_index

# 开始处理位置信息
# print(dataAoiPD)
# print(dataAoiPD.set_index('tdid').stack())  # 将所有的位置作为特征进行平铺展开
# print(pd.get_dummies(dataAoiPD.set_index('tdid').stack()))
# print(pd.get_dummies(dataAoiPD.set_index('tdid').stack()).sum(level=0))
dataAoisPD = pd.get_dummies(dataAoiPD.set_index('tdid').stack()).sum(level=0)
dataAoisPD.pop('null')  # 删除null列
dataAoiPD.set_index('tdid')
dataAoiPD.sort_index

dataDevPD.to_csv('src/DataCleaning/DevFeatures.csv', encoding='utf-8_sig')
dataAoisPD.to_csv('src/DataCleaning/AoiFeatures.csv', encoding='utf-8_sig')

# 开始处理App信息
print(len(dataAppPD))
dataAppPD.set_index('tdid')
dataAppPD = dataAppPD.drop_duplicates('tdid')  # 去重处理
print(len(dataAppPD))
dataAppPD.to_csv('src/DataCleaning/AppPD_DROP_DUP.csv', encoding='utf-8_sig')
# print(len(pd.get_dummies(dataAppPD.set_index('tdid').stack())))
# dataAppPD = pd.get_dummies(dataAppPD.set_index('tdid').stack()).sum(level=0)
# dataAppPD.loc['Row_sum'] = dataAppPD.apply(lambda x: x.sum())
# dataAppPD['Row_sum']['tdid']=0
# dataAppPD=dataAppPD.stack()
# dataAppPD=dataAppPD.sort
# print(dataAppPD)
# dataAppPD.pop('null')
# print(len(dataAoisPD))
