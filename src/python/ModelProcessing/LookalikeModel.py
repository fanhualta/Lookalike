from __future__ import division, print_function
import numpy as np
import pandas as pd
import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier
from src.python.ModelProcessing.RocScore import roc_score
from src.python.Config.Config import FILE_ROOT_DIRECTORY


# 本函数用于模型训练和预测，以及通过调用RocScore中的roc_score函数计算出最后的准确率
# 提取的总特征数337， 行数84572，FinalFeaturesSorted.csv表为84572 rows * 337 columns的形式，其中依次是AppFeatures:300个,AoiFeatures:35个,DevFeatures:2个
# 函数参数分别为：种子人群文件路径、 训练次数、正确结果的文件路径


def model_training_and_predict(seeds_path, t, y_true_path):
    # 1. 获取所有的数据特征
    all_features = []
    with open(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/FinalFeaturesSorted.csv'), 'r') as f:
        rows = csv.reader(f)
        row_index = 0
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
    K = int(NP)  # 正样本数
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
        # data_bootstrap = pd.concat((data_P, data_U.iloc[bootstrap_sample, :]), axis=0)
        # 训练模型
        # model = svm.SVC()  # SVM
        model = XGBClassifier(n_jobs=4)  # XGBoost
        # model = XGBClassifier(silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        #                       # nthread=4,# cpu 线程数 默认最大
        #                       learning_rate=0.3,  # 如同学习率
        #                       min_child_weight=1,
        #                       # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        #                       # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        #                       # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        #                       max_depth=6,  # 构建树的深度，越大越容易过拟合
        #                       gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        #                       subsample=1,  # 随机采样训练样本 训练实例的子采样比
        #                       max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        #                       colsample_bytree=1,  # 生成树时进行的列采样
        #                       reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        #                       # reg_alpha=0, # L1 正则项参数
        #                       # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        #                       # objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        #                       # num_class=10, # 类别数，多分类与 multisoftmax 并用
        #                       n_estimators=100,  # 树的个数
        #                       seed=1000  # 随机种子
        #                       # eval_metric= 'auc'
        #                       )  # XGBoost
        # model = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', class_weight='balanced')  # 决策树
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
    # 对于种子人群的预测概率，直接置为1.0
    for row_id in range(len(all_features)):
        if all_features[row_id][0] in positive_group:
            predict_proba[row_id] = 1.0
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


result = model_training_and_predict(os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/DataCleaning/seeds_1'), 100,
                                    os.path.join(FILE_ROOT_DIRECTORY, 'src/resource/PredictProbability/1.txt'))
print(result)
