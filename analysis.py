import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import common
import service
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 随机森林
# 导入所需要的包
import pickle
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
# 评估报告
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
# 交叉验证
from sklearn.model_selection import cross_val_score
# 网格搜索
from sklearn.model_selection import GridSearchCV
# 归一化，标准化
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# 忽略警告
import warnings

warnings.filterwarnings("ignore")


class Analysis(object):

    # 训练
    def do(self):

        for i in range(35):
            x, y = self.get_fit_data(i, -1)
            self.to_fit_model(x, y, "b", i + 1)
        # for i in range(12):
        #     self.to_fit_model(a_x, a_y.T[i], "a", i + 1)

    # 进行分项训练
    def to_fit_model(self, x, y, place: str, number: int):

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=90)

        scorel = []
        for i in range(30, 200, 10):
            rfc = RandomForestClassifier(max_depth=8, n_estimators=i, n_jobs=-1, random_state=90)
            rfc.fit(x_train, preprocessing.LabelEncoder().fit_transform(y_train))
            score = rfc.score(x_test, y_test)
            scorel.append(score)

        n_estimators = ([*range(30, 200, 10)][scorel.index(max(scorel))])

        scorel = []
        for i in range(3, 30):
            rfc = RandomForestClassifier(max_depth=i, n_estimators=n_estimators, n_jobs=-1, random_state=90)
            rfc.fit(x_train, preprocessing.LabelEncoder().fit_transform(y_train))
            score = rfc.score(x_test, y_test)
            scorel.append(score)

        max_depth = ([*range(3, 30)][scorel.index(max(scorel))])

        scorel = []
        for i in range(1, 20):
            rfc = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, min_samples_leaf=i, n_jobs=-1, random_state=90)
            rfc.fit(x_train, preprocessing.LabelEncoder().fit_transform(y_train))
            score = rfc.score(x_test, y_test)
            scorel.append(score)

        min_samples_leaf = ([*range(1, 20)][scorel.index(max(scorel))])

        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=7,
                                       max_features='sqrt', criterion='entropy')
        # 训练模型
        model.fit(x_train, preprocessing.LabelEncoder().fit_transform(y_train))
        y_pred = model.predict(x_test)
        '''
        评估指标
        '''
        # 求出预测和真实一样的数目
        true = np.sum(y_pred == y_test)
        print('预测对的结果数目为：', true)
        print('预测错的的结果数目为：', y_test.shape[0] - true)
        # 评估指标
        print('预测数据的准确率为： {:.4}%'.format(accuracy_score(y_test, y_pred) * 100))
        print('预测数据的精确率为：{:.4}%'.format(
            precision_score(y_test, y_pred, average='macro') * 100))
        print('预测数据的召回率为：{:.4}%'.format(
            recall_score(y_test, y_pred, average='macro') * 100))
        # print("训练数据的F1值为：", f1score_train)
        print('预测数据的F1值为：',
              f1_score(y_test, y_pred, average='macro'))
        print('预测数据的Cohen’s Kappa系数为：',
              cohen_kappa_score(y_test, y_pred))
        # 打印分类报告
        print('预测数据的分类报告为：', '\n',
              classification_report(y_test, y_pred))

        if not os.path.exists(os.path.dirname("./data/model/")):
            os.makedirs(os.path.dirname("./data/model/"), mode=7777)

        with open("./data/model/" + place + "_" + str(number) + ".pkl", 'wb') as f:
            pickle.dump(model, f)


        return

    # 预测
    def to_predit(self, before):

        res = []
        keys = []
        for i in range(28):
            b_x, a_x = self.get_fit_data(i, -1, "23070", "23080")
            with open("./data/model/b_" + str(i + 1) + ".pkl", 'rb') as f:
                model = pickle.load(f)
                y_pred = model.predict([b_x.T["23080"]])
                if y_pred[0] > 0:
                    keys.append(str(i + 1))
                    res.append(y_pred[0])

        print(pd.DataFrame(res, keys))
        return

    # 获取训练数据
    def get_fit_data(self, b_index, a_index, begin_index: str = "07077", end_index: str = "23036"):

        s = service.Service()

        m, step, n, draw, dot_count = 8, 1, 15, False, 500
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        b_single_count1, a_single_count1 = s.get_single_count()
        b_single_prob1, a_single_prob1 = s.get_single_prob()
        b_single_prob_ema1, a_single_prob_ema1 = s.get_single_prob_ema()
        b_single_prob_ema_speed1, a_single_prob_ema_speed1 = s.get_single_prob_ema_speed()

        b_piece_count1, a_piece_count1, b_piece_index1, a_piece_index1 = s.get_piece_count()
        b_piece_prob1, a_piece_prob1 = s.get_piece_prob()
        b_piece_prob_ema1, a_piece_prob_ema1 = s.get_piece_prob_ema()
        b_piece_prob_ema_speed1, a_piece_prob_ema_speed1 = s.get_piece_prob_ema_speed()

        m, step, n, draw, dot_count = 16, 1, 15, False, 500
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        b_single_count2, a_single_count2 = s.get_single_count()
        b_single_prob2, a_single_prob2 = s.get_single_prob()
        b_single_prob_ema2, a_single_prob_ema2 = s.get_single_prob_ema()
        b_single_prob_ema_speed2, a_single_prob_ema_speed2 = s.get_single_prob_ema_speed()

        b_piece_count2, a_piece_count2, b_piece_index2, a_piece_index2 = s.get_piece_count()
        b_piece_prob2, a_piece_prob2 = s.get_piece_prob()
        b_piece_prob_ema2, a_piece_prob_ema2 = s.get_piece_prob_ema()
        b_piece_prob_ema_speed2, a_piece_prob_ema_speed2 = s.get_piece_prob_ema_speed()

        m, step, n, draw, dot_count = 32, 1, 15, False, 500
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        b_single_count3, a_single_count3 = s.get_single_count()
        b_single_prob3, a_single_prob3 = s.get_single_prob()
        b_single_prob_ema3, a_single_prob_ema3 = s.get_single_prob_ema()
        b_single_prob_ema_speed3, a_single_prob_ema_speed3 = s.get_single_prob_ema_speed()

        b_piece_count3, a_piece_count3, b_piece_index3, a_piece_index3 = s.get_piece_count()
        b_piece_prob3, a_piece_prob3 = s.get_piece_prob()
        b_piece_prob_ema3, a_piece_prob_ema3 = s.get_piece_prob_ema()
        b_piece_prob_ema_speed3, a_piece_prob_ema_speed3 = s.get_piece_prob_ema_speed()

        m, step, n, draw, dot_count = 56, 1, 15, False, 500
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        b_single_count5, a_single_count5 = s.get_single_count()
        b_single_prob5, a_single_prob5 = s.get_single_prob()
        b_single_prob_ema5, a_single_prob_ema5 = s.get_single_prob_ema()
        b_single_prob_ema_speed5, a_single_prob_ema_speed5 = s.get_single_prob_ema_speed()

        b_piece_count5, a_piece_count5, b_piece_index5, a_piece_index5 = s.get_piece_count()
        b_piece_prob5, a_piece_prob5 = s.get_piece_prob()
        b_piece_prob_ema5, a_piece_prob_ema5 = s.get_piece_prob_ema()
        b_piece_prob_ema_speed5, a_piece_prob_ema_speed5 = s.get_piece_prob_ema_speed()

        b_y, a_y = s.get_y_data()
        x = pd.DataFrame([], b_single_count3.iloc[b_index][begin_index: end_index].index).T

        if b_index != -1:
            y = b_y.iloc[b_index][begin_index: end_index]
            x.loc["b_single_count1"] = b_single_count1.iloc[b_index][begin_index: end_index]
            x.loc["b_single_count2"] = b_single_count2.iloc[b_index][begin_index: end_index]
            x.loc["b_single_count3"] = b_single_count3.iloc[b_index][begin_index: end_index]
            x.loc["b_single_count5"] = b_single_count5.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob1"] = b_single_prob1.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob2"] = b_single_prob2.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob3"] = b_single_prob3.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob5"] = b_single_prob5.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema1"] = b_single_prob_ema1.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema2"] = b_single_prob_ema2.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema3"] = b_single_prob_ema3.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema5"] = b_single_prob_ema5.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema_speed1"] = b_single_prob_ema_speed1.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema_speed2"] = b_single_prob_ema_speed2.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema_speed3"] = b_single_prob_ema_speed3.iloc[b_index][begin_index: end_index]
            x.loc["b_single_prob_ema_speed5"] = b_single_prob_ema_speed5.iloc[b_index][begin_index: end_index]

            piece_index = 0
            for i0 in range(len(b_piece_index5)):
                if b_index in b_piece_index5.iloc[i0].astype('int').to_numpy().tolist():
                    piece_index = i0
                    break
            print(piece_index)
            x.loc["b_piece_count1"] = b_piece_count1.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_count2"] = b_piece_count2.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_count3"] = b_piece_count3.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_count5"] = b_piece_count5.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob1"] = b_piece_prob1.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob2"] = b_piece_prob2.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob3"] = b_piece_prob3.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob5"] = b_piece_prob5.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob_ema1"] = b_piece_prob_ema1.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob_ema2"] = b_piece_prob_ema2.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob_ema3"] = b_piece_prob_ema3.iloc[piece_index][begin_index: end_index]
            x.loc["b_piece_prob_ema5"] = b_piece_prob_ema5.iloc[piece_index][begin_index: end_index]
            # x.loc["b_piece_prob_ema_speed1"] = b_piece_prob_ema_speed1.iloc[piece_index][begin_index: end_index]
            # x.loc["b_piece_prob_ema_speed2"] = b_piece_prob_ema_speed2.iloc[piece_index][begin_index: end_index]
            # x.loc["b_piece_prob_ema_speed3"] = b_piece_prob_ema_speed3.iloc[piece_index][begin_index: end_index]
            # x.loc["b_piece_prob_ema_speed5"] = b_piece_prob_ema_speed5.iloc[piece_index][begin_index: end_index]

        else:
            y = a_y.iloc[a_index][begin_index: end_index]
            x.loc["a_single_count1"] = a_single_count1.iloc[a_index][begin_index: end_index]
            x.loc["a_single_count2"] = a_single_count2.iloc[a_index][begin_index: end_index]
            x.loc["a_single_count3"] = a_single_count3.iloc[a_index][begin_index: end_index]
            x.loc["a_single_count5"] = a_single_count5.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob1"] = a_single_prob1.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob2"] = a_single_prob2.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob3"] = a_single_prob3.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob5"] = a_single_prob5.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema1"] = a_single_prob_ema1.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema2"] = a_single_prob_ema2.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema3"] = a_single_prob_ema3.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema5"] = a_single_prob_ema5.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema_speed1"] = a_single_prob_ema_speed1.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema_speed2"] = a_single_prob_ema_speed2.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema_speed3"] = a_single_prob_ema_speed3.iloc[a_index][begin_index: end_index]
            x.loc["a_single_prob_ema_speed5"] = a_single_prob_ema_speed5.iloc[a_index][begin_index: end_index]

            x.loc["a_piece_count1"] = a_piece_count1.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_count2"] = a_piece_count2.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_count3"] = a_piece_count3.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_count5"] = a_piece_count5.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob1"] = a_piece_prob1.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob2"] = a_piece_prob2.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob3"] = a_piece_prob3.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob5"] = a_piece_prob5.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema1"] = a_piece_prob_ema1.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema2"] = a_piece_prob_ema2.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema3"] = a_piece_prob_ema3.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema5"] = a_piece_prob_ema5.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema_speed1"] = a_piece_prob_ema_speed1.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema_speed2"] = a_piece_prob_ema_speed2.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema_speed3"] = a_piece_prob_ema_speed3.iloc[b_index][begin_index: end_index]
            x.loc["a_piece_prob_ema_speed5"] = a_piece_prob_ema_speed5.iloc[b_index][begin_index: end_index]

        x[:][np.isinf(x[:])] = 10000
        x[:][np.isneginf(x[:])] = -10000
        x = x.fillna(0)

        return x.T, y.T

