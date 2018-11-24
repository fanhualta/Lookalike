from __future__ import division, print_function
import numpy as np
import matplotlib.pylab as plt

'exec(%matplotlib inline)'
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve

# 本程序为PU learning的一个示例学习程序，并非Lookalike项目中用到的py程序

N = 60
known_labels_ratio = 0.1
X, y = make_moons(n_samples=N, noise=0.1, shuffle=True)  # 随机生成一些样例数据，其中X表示的是坐标，y表示打的标签
rp = np.random.permutation(int(N / 2))  # 随机生成一个序列并打乱顺序
data_P = X[y == 1][rp[:int(len(rp) * known_labels_ratio)]]  # 在所有的样例数据中标签为1的数据抽取一定数据量的样例作为正样本
print(y)
print(rp)
print(X[y == 1])
print(rp[:int(len(rp) * known_labels_ratio)])
print(data_P)
data_U = np.concatenate((X[y == 1][rp[int(len(rp) * known_labels_ratio):]], X[y == 0]), axis=0)  # 将正样本以外的所有数据作为无标签数据
# print("Amount of labeled samples: %d" % (data_P.shape[0]))
# print(X[y==1][rp[int(len(rp)*known_labels_ratio):]])
# print((X[y==1][rp[int(len(rp)*known_labels_ratio):]], X[y==0]))
plt.figure(figsize=(8, 4.5))
plt.scatter(data_U[:, 0], data_U[:, 1], c='k', marker='.', linewidth=1, s=1, alpha=0.5, label='Unlabeled')
plt.scatter(data_P[:, 0], data_P[:, 1], c='b', marker='o', linewidth=0, s=20, alpha=0.5, label='Positive')
plt.grid()
plt.legend()
NP = data_P.shape[0]
NU = data_U.shape[0]

T = 1000  # 训练次数
K = NP  # 正样本数
train_label = np.zeros(shape=(NP + K,))  # 训练样本的数量为正样本的数量乘2
shape = (NP + K,)
# print(shape)
# print(train_label)
train_label[:NP] = 1.0
n_oob = np.zeros(shape=(NU,))
# print(n_oob)
f_oob = np.zeros(shape=(NU, 2))
# print(f_oob)
for i in range(T):
    # Bootstrap resample
    bootstrap_sample = np.random.choice(np.arange(NU), replace=True, size=K)  # 从非样本实例中随机选取一定数量的样本，并取完放回，长度为NP
    # Positive set + bootstrapped unlabeled set
    data_bootstrap = np.concatenate((data_P, data_U[bootstrap_sample, :]), axis=0)
    # Train model
    model = DecisionTreeClassifier(max_depth=None, max_features=None,
                                   criterion='gini', class_weight='balanced')
    model.fit(data_bootstrap, train_label)
    # Index for the out of the bag (oob) samples
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    # Transductive learning of oob samples
    f_oob[idx_oob] += model.predict_proba(data_U[idx_oob])  # 每一个样本的数量
    # print(f_oob)
    n_oob[idx_oob] += 1
# print(f_oob)
predict_proba = f_oob[:, 1] / n_oob
print(predict_proba)
# Plot the class probabilities for the unlabeled samples
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
sp = ax1.scatter(data_U[:, 0], data_U[:, 1], c=predict_proba, linewidth=0, s=5, alpha=0.5, cmap=plt.cm.plasma,
                 label='unlabeled')
plt.grid()
plt.colorbar(sp, label='Class probability on Unlabeled set')

true_labels = np.zeros(shape=(data_U.shape[0]))
true_labels[:int(len(rp) * (1.0 - known_labels_ratio))] = 1.0
precision, recall, th = precision_recall_curve(true_labels, predict_proba)
ax2 = fig.add_subplot(1, 2, 2)
f1s = precision[:-1] * recall[:-1]
ax2.plot(th, f1s, linewidth=2, alpha=0.5)
best_th = np.argmax(f1s)
ax2.plot(th[best_th], f1s[best_th], c='r', marker='o')
ax2.plot([th[best_th], th[best_th]], [0.0, f1s[best_th]], 'r--')
ax2.plot([0.0, th[best_th]], [f1s[best_th], f1s[best_th]], 'r--')
ax2.annotate('Pre: %0.3f, Rec: %0.3f' % (precision[best_th], recall[best_th]),
             xy=(th[best_th] + 0.01, f1s[best_th] - 0.05))
ax2.set_ylabel('F1 score')
ax2.set_xlabel('Probability threshold')
plt.grid()
plt.show()
