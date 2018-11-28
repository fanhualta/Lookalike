from __future__ import division, print_function
import numpy as np
import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import sys

sys.path.append(os.path.join(sys.path[0], '..', 'Config'))
from RocScore import roc_score
from Config import FILE_ROOT_DIRECTORY


# 本函数用于模型训练和预测，以及通过调用RocScore中的roc_score函数计算出最后的准确率
# 提取的总特征数362， 行数84572，FinalFeaturesSorted.csv表为84572 rows * 326 columns的形式，其中依次是AppFeatures:326个,AoiFeatures:35个,DevFeatures:2个
# 函数参数分别为：种子人群文件路径、 训练次数、正确结果的文件路径


def model_training_and_predict(seeds_path, t, y_true_path, model):
    # 1. 获取所有的数据特征
    all_features = []
    with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/FinalFeaturesSorted.csv'), 'r',
              encoding='utf-8_sig') as f:
        rows = csv.reader(f)
        row_index = 0
        print(f)
        for row in rows:
            row_index += 1
            if row_index == 1:
                # print(len(row))
                continue
            all_features.append(row)
    # print(len(all_tdid))

    # 2. 获取所有的正样本
    positive_group = []  # 表示正样本的tdid
    with open(seeds_path, 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            positive_group.append(row[0])
    # print(len(all_features))
    # print(len(positive_group))
    # print(all_features)

    # 3. 构成全量数据的label
    label = []  # 表示每个tdid的label，0表示的是负样本，1表示的是正样本
    count = 0
    for row_id in range(len(all_features)):
        # print(all_tdid[row_id][0])
        if all_features[row_id][0] in positive_group:
            label.append(1)
            count += 1
        else:
            label.append(0)
    # print(len(label))
    # print(count)

    # 4. 转化为numpy数组用于数据处理
    X = np.asarray(all_features)  # 转化为np
    y = np.asarray(label)  # 转化为np
    data_P = X[y == 1]  # 正样本
    data_P = np.delete(data_P, 0, axis=1)  # 去掉tdid列
    data_U = X[y == 0]  # 无标签样本
    data_U = np.delete(data_U, 0, axis=1)  # 去掉tdid列
    data_A = np.delete(X, 0, axis=1)  # 待预测的全部数据
    print("Amount of labeled samples: %d" % (data_P.shape[0]))
    print("Amount of unlabeled samples: %d" % (data_U.shape[0]))
    print("Amount of all samples: %d" % (data_A.shape[0]))

    NP = data_P.shape[0]
    NU = data_U.shape[0]
    NA = data_A.shape[0]
    # print(data_A)

    # 5. 模型训练和结果预测
    # T = 100  # 训练次数：100次
    K = NP  # 正样本数
    train_label = np.zeros(shape=(NP + K,))  # 训练样本的数量为正样本的数量乘2
    shape = (NP + K,)
    train_label[:NP] = 1.0
    n_oob = np.zeros(shape=(NA,))  # 训练中被预测的次数
    f_oob = np.zeros(shape=(NA, 2))  # 预测的结果
    for i in range(t):
        print("这是第%d次训练" % (i + 1))
        # Bootstrap resample
        bootstrap_sample = np.random.choice(np.arange(NU), replace=True, size=K)  # 从非样本实例中随机选取一定数量的样本，并取完放回，长度为NP
        # 获取训练集，训练集的构成为： Positive set + bootstrapped unlabeled set
        data_bootstrap = np.concatenate((data_P, data_U[bootstrap_sample, :]), axis=0)
        # 训练模型
        # 模型训练
        model.fit(data_bootstrap, train_label)
        # 结果预测
        idx_oob = range(NA)
        # 预测结果的累加
        f_oob[idx_oob] += model.predict_proba(data_A[idx_oob])  # 每一个样本的数量
        # 被预测的次数
        n_oob[idx_oob] += 1

    # print(f_oob)
    # 预测结果
    predict_proba = f_oob[:, 1] / n_oob
    # 将结果进行保存
    np.savetxt(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/PredictProbability/predict_proba.csv'), predict_proba,
               delimiter=',', fmt='%f')
    return roc_score(y_true_path,
                     os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/PredictProbability/predict_proba.csv'))
    # predict_proba_tdid = []
    # row = []
    # row.append("tdid")
    # row.append("predict_proba")
    # predict_proba_tdid.append(row)
    # for row_id in range(NA):
    #     row = []
    #     # print(int(X[row_id][0]))
    #     # print(predict_proba[row_id])
    #     row.append(int(X[row_id][0]))
    #     row.append(predict_proba[row_id])
    #     predict_proba_tdid.append(row)
    # print(predict_proba_tdid)
    # np.savetxt('src/PredictProbability/predict_proba_tdid.csv', predict_proba_tdid, delimiter=',', fmt='%d,%f')


if __name__ == '__main__':
    # 训练模型
    # model = GradientBoostingClassifier(max_features=None, max_depth=None, min_samples_split=8, min_samples_leaf=3, n_estimators=1200, learning_rate=0.05, subsample=0.95)
    # model = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='gini', class_weight='balanced')
    # model = AdaBoostClassifier()
    train_times = [1, 10, 100]
    for i in range(len(train_times)):
        result = model_training_and_predict(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/seeds_1'),
                                            train_times[i],
                                            os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/PredictProbability/1.txt'),
                                            model)
        print(result)
