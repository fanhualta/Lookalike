from __future__ import division, print_function
import pandas as pd
import csv

# 提取的总特征数362， 行数84572，FinalFeaturesSorted.csv表为84572 rows * 326 columns的形式，其中依次是AppFeatures:326个,AoiFeatures:35个,DevFeatures:2个
all_tdid = []
with open('src/DataCleaning/FinalFeaturesSorted.csv', 'r') as f:
    rows = csv.reader(f)
    row_index = 0
    for row in rows:
        row_index += 1
        if row_index == 1:
            print(len(row))
            continue
        all_tdid.append(row[0])
print(len(all_tdid))
