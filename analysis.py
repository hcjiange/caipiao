import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import common
import service
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(m: int, step: int, n: int, draw: bool, dot_count: int):

    s = service.Service()

    s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)

    # 获取基础数据
    all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = s.get_base_data()
    before_speed_data, after_speed_data = s.get_single_data()
    before_parity_data, after_parity_data, before_parity_speed_data, after_parity_speed_data = s.get_parity_data()
    before_pieces_data, after_pieces_data, before_pieces_speed_data, after_pieces_speed_data = s.get_piece_data(b_n=7, a_n=4)

    single_data_before = pd.DataFrame(single_data_before).T - all_before_average
    single_data_after = pd.DataFrame(single_data_after).T - all_after_average

    before_data = pd.DataFrame(s.data).loc[:, "lotteryDrawResult"].apply(lambda x: pd.Series(str(x).split(" ")[:-2]).astype('int').to_numpy().tolist())
    after_data = pd.DataFrame(s.data).loc[:, "lotteryDrawResult"].apply(lambda x: pd.Series(str(x).split(" ")[-2:]).astype('int').to_numpy().tolist())

    single_data_before_result = pd.DataFrame(single_data_before.T.to_numpy().tolist())

    # print(before_data[0])
    # exit()
    for i in range(len(date_index) - 1):
        next_i = i + 1
        for i0 in range(len(single_data_before_result[i])):
            single_data_before_result[i][i0] = 1 if i0+1 in np.array(before_data[next_i]).tolist() else 0


    # print(single_data_before[2])
    # exit()
    print(single_data_before_result.T[-20:])
    model = RandomForestClassifier()
    model.fit(single_data_before, preprocessing.LabelEncoder().fit_transform(single_data_before_result.T[1]))
    res = model.predict(single_data_before[-20:])
    print(res)
    # sns_data_before = single_data_before.assign(Y=single_data_before_result.T[0])
    # print(sns_data_before)
    # exit()
    # plt.figure(figsize=(20, 20))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # sns.heatmap(sns_data_before.corr(), cmap="YlGnBu", annot=False)
    # plt.title("相关性分析图")
    # plt.show()

if __name__ == '__main__':

    # m, step, n, draw, dot_count = 50, 3, 15, False, 200
    # get_data(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    #
    # m, step, n, draw, dot_count = 50, 1, 15, False, 200
    # get_data(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    #
    m, step, n, draw, dot_count = 30, 1, 15, False, 200
    get_data(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    #
    # m, step, n, draw, dot_count = 10, 1, 15, False, 200
    # get_data(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
