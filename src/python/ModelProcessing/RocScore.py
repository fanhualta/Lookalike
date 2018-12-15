from sklearn.metrics import roc_auc_score
import csv
import os
import sys

sys.path.append(os.path.join(sys.path[0], '..', 'Config'))

from Config import FILE_ROOT_DIRECTORY


# 本程序被LookalikeModel的model_training_and_predict函数调用，作为最后计算roc准确性的步骤
# 本函数用于预测模型后计算roc，输入的两个参数分别是：正确结果的文件路径、预测结果的文件路径


def roc_score(y_true_path, y_score_path):
    all_tdid = []
    with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/FinalFeaturesSorted.csv'), 'r',
              encoding='utf-8_sig') as f:
        rows = csv.reader(f)
        row_index = 0
        for row in rows:
            row_index += 1
            if row_index == 1:
                # print(len(row))
                continue
            all_tdid.append(row[0])
    all_P = []
    with open(y_true_path, 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            all_P.append(int(row[0]))
    y_true = []
    # print(len(all_P))
    # print(all_P)
    count = 0
    for row_id in range(len(all_tdid)):
        # row.append(int(all_tdid[row_id]))
        if int(all_tdid[row_id]) in all_P:
            count += 1
            y_true.append(1.0)
        else:
            y_true.append(0.0)
    # print('1.txt中有%d个有效数据' % count)
    # print(y_true)
    # np.savetxt("src/PredictProbability/y_true.csv", y_true, delimiter=",")
    y_scores = []
    with open(y_score_path, 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            y_scores.append(float(row[0]))
    return roc_auc_score(y_true, y_scores)


if __name__ == '__main__':
    result = roc_score(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/PredictProbability/1.txt'),
                       os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/PredictProbability/predict_proba.csv'))
    print(result)
