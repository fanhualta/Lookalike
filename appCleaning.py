from __future__ import division, print_function
import pandas as pd

# 本程序中用来对app特征进行提取
dataAppPD = pd.read_csv('/home/litianan/lta/lookalike/app.csv')
dataAppsPD = pd.get_dummies(dataAppPD.set_index('tdid').stack()).sum(level=0)
print(dataAppPD)
dataAppPD.loc['Row_sum'] = dataAppPD.apply(lambda x: x.sum())
dataAppPD['Row_sum']['tdid'] = 0
dataAppPD.to_csv('/home/litianan/lta/lookalike/appPD.csv')
dataAppPD = dataAppPD.stack()
dataAppPD = dataAppPD.sort
print(dataAppPD)
dataAppPD.pop('null')
dataAppPD.to_csv('/home/litianan/lta/lookalike/appFeatures.csv')
