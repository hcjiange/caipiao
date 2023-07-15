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

    def do(self):

        m, step, n, draw, dot_count = 30, 1, 15, False, 200

        s = service.Service()
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)

        b_single_count, a_single_count = s.get_single_count()
        b_single_prob, a_single_prob = s.get_single_prob()
        b_single_prob_ema, a_single_prob_ema = s.get_single_prob_ema()
        b_single_prob_ema_speed, a_single_prob_ema_speed = s.get_single_prob_ema_speed()
        b_single_prob = b_single_prob

        org_data = common.read_json("./data/data.json")
        b_data = pd.DataFrame(org_data[30:], b_single_prob.T.index).loc[:, "lotteryDrawResult"].apply(
            lambda x: pd.Series(str(x).split(" ")[:-2]).astype('int').to_numpy().tolist())
        a_data = pd.DataFrame(org_data).loc[:, "lotteryDrawResult"].apply(
            lambda x: pd.Series(str(x).split(" ")[-2:]).astype('int').to_numpy().tolist())

        # 复制一个同大小阵列
        b_y = pd.DataFrame(b_single_prob.T.to_numpy().tolist(), b_single_prob.T.index).T
        b_y.loc[:, :] = 0
        b_y_include_num = 2
        for i in range(0, len(b_single_prob.T) - b_y_include_num):
            for i0 in range(len(b_y[b_single_prob.T.index[i]])):
                if i0 + 1 in np.array(b_data[b_single_prob.T.index[i + 1]]).tolist() or i0 + 1 in np.array(b_data[b_single_prob.T.index[i + 2]]).tolist():
                    b_y[b_single_prob.T.index[i]][i0] = 1
                else:
                    b_y[b_single_prob.T.index[i]][i0] = 0

        b_y.T.to_csv("./data/b_y.csv")
        # a_y.T.to_csv("./data/a_y.csv")

        b_x = b_single_prob
        a_x = a_single_prob

        print("begin:")
        # i = 0
        # self.to_fit_model(b_x.T, b_y.T[i], "b", i + 1)
        for i in range(35):
            self.to_fit_model(b_x.T, b_y.T[i], "b", i + 1)
        # for i in range(12):
        #     self.to_fit_model(a_x, a_y.T[i], "a", i + 1)

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

    def get_data(self, m: int, step: int, n: int, draw: bool, dot_count: int):

        s = service.Service()

        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)

        # 获取基础数据
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = s.get_base_data()
        before_speed_data, after_speed_data = s.get_single_data()
        before_parity_data, after_parity_data, before_parity_speed_data, after_parity_speed_data = s.get_parity_data()
        before_pieces_data, after_pieces_data, before_pieces_speed_data, after_pieces_speed_data = s.get_piece_data(
            b_n=7, a_n=4)

        single_data_before = pd.DataFrame(single_data_before).T - all_before_average
        single_data_after = pd.DataFrame(single_data_after).T - all_after_average

        before_speed_data = pd.DataFrame(before_speed_data).T
        after_speed_data = pd.DataFrame(after_speed_data).T

        return single_data_before, single_data_after, date_index, before_speed_data, after_speed_data

    def to_predit(self, before):

        m, step, n, draw, dot_count = 30, 1, 15, False, 200
        s = service.Service()
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        b_single_count, a_single_count = s.get_single_count()
        b_single_prob, a_single_prob = s.get_single_prob()
        b_single_prob_ema, a_single_prob_ema = s.get_single_prob_ema()
        b_single_prob_ema_speed, a_single_prob_ema_speed = s.get_single_prob_ema_speed()

        b_x = b_single_prob["23043"]

        print(b_single_prob)
        print(b_x)
        a_x = a_single_prob
        res = []
        keys = []
        for i in range(35):
            with open("./data/model/b_" + str(i + 1) + ".pkl", 'rb') as f:
                model = pickle.load(f)
                y_pred = model.predict([b_x])
                if y_pred[0] > 0:
                    keys.append(str(i + 1))
                    res.append(y_pred[0])

        print(pd.DataFrame(res, keys))
        return
