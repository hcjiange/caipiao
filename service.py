import json
import pandas as pd
import common
import matplotlib.pyplot as plt
import numpy as np
import os


class Service(object):
    data = []
    m = 30
    step = 1
    n = 30
    draw = False
    dot_count = 200
    before = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
              "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
              "35"]
    after = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # 初始化获取数据
    def __init__(self):
        self.data = common.read_json("./data/data.json")

    def init(self, m: int, step: int, n: int, draw: bool, dot_count: int):
        self.m = m
        self.step = step
        self.n = n
        self.draw = draw
        self.dot_count = dot_count

    # 获取数量 pass
    @staticmethod
    def get_count(data):
        single_data_before = {"01": 0, "02": 0, "03": 0, "04": 0, "05": 0, "06": 0, "07": 0, "08": 0, "09": 0, "10": 0,
                              "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0, "20": 0,
                              "21": 0, "22": 0, "23": 0, "24": 0, "25": 0, "26": 0, "27": 0, "28": 0, "29": 0, "30": 0,
                              "31": 0, "32": 0, "33": 0, "34": 0, "35": 0}
        single_data_after = {"01": 0, "02": 0, "03": 0, "04": 0, "05": 0, "06": 0, "07": 0, "08": 0, "09": 0, "10": 0,
                             "11": 0, "12": 0}
        for item in data:
            for date_item in str.split(item['lotteryDrawResult'], " ")[:-2]:
                single_data_before[date_item] += 1
            for date_item in str.split(item['lotteryDrawResult'], " ")[-2:]:
                single_data_after[date_item] += 1
        return single_data_before, single_data_after

    # 获取单号区间内出现的次数 pass
    def get_single_count(self, is_from_file: bool = True):
        data = self.data
        if is_from_file:
            # 从文件里读取数据
            b_single_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_count.csv").iloc[:, 1:]
            a_single_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_count.csv").iloc[:, 1:]
        else:
            b_single_count = pd.DataFrame([])
            a_single_count = pd.DataFrame([])
            # 获取前m期出现次数
            for i in range(len(data)):
                item = data[i]
                if self.m <= i < len(data) and (i % self.step == 0 or i == len(data) - 1):
                    b_single_counts, a_single_counts = self.get_count(data[i+1-self.m:i+1])
                    for i0 in range(len(self.before)):
                        b_single_count.loc[i0, item['lotteryDrawNum']] = b_single_counts[self.before[i0]]
                    for i0 in range(len(self.after)):
                        a_single_count.loc[i0, item['lotteryDrawNum']] = a_single_counts[self.after[i0]]

            # 存储数据
            if not os.path.exists(os.path.dirname("./data/" + str(self.m) + "_" + str(self.step) + "/")):
                os.makedirs(os.path.dirname("./data/" + str(self.m) + "_" + str(self.step) + "/"), mode=7777)
            b_single_count.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_count.csv")
            a_single_count.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_count.csv")
        return b_single_count, a_single_count

    # 获取单号码出现比率 pass
    def get_single_prob(self, is_from_file: bool = True):
        b_single_count, a_single_count = self.get_single_count()

        if is_from_file:
            # 从文件里读取数据
            b_single_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob.csv").iloc[:, 1:]
            a_single_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_single_prob = b_single_count / (self.m * 5)
            a_single_prob = a_single_count / (self.m * 2)
            b_single_prob.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob.csv")
            a_single_prob.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob.csv")
        return b_single_prob, a_single_prob

    # 获取单号码出现比率移动均线 pass
    def get_single_prob_ema(self, is_from_file: bool = True):
        b_single_prob, a_single_prob = self.get_single_prob()

        if is_from_file:
            # 从文件里读取数据
            b_single_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema.csv").iloc[:, 1:]
            a_single_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_single_prob_ema = b_single_prob.T.loc[:].apply(lambda x: x.rolling(window=self.n).mean()).T
            a_single_prob_ema = a_single_prob.T.loc[:].apply(lambda x: x.rolling(window=self.n).mean()).T
            b_single_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema.csv")
            a_single_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema.csv")
        return b_single_prob_ema, a_single_prob_ema

    # 获取单号码出现比率移动均线速率 pass
    def get_single_prob_ema_speed(self, is_from_file: bool = True):

        b_single_prob_ema, a_single_prob_ema = self.get_single_prob_ema()
        if is_from_file:
            # 从文件里读取数据
            b_single_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema_speed.csv").iloc[:, 1:]
            a_single_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema_speed.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_single_prob_ema_speed = b_single_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            a_single_prob_ema_speed = a_single_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            b_single_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema_speed.csv")
            a_single_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema_speed.csv")
        return b_single_prob_ema_speed, a_single_prob_ema_speed

    # 绘制单号曲线 pass
    def draw_single(self):

        dot_count = 500

        b_single_prob, a_single_prob = self.get_single_prob()
        b_single_prob_ema, a_single_prob_ema = self.get_single_prob_ema()
        b_single_prob_ema_speed, a_single_prob_ema_speed = self.get_single_prob_ema_speed()

        # 创建文件夹数据
        if not os.path.exists(os.path.dirname("./images/" + str(self.m) + "_" + str(self.step) + "/single/")):
            os.makedirs(os.path.dirname("./images/" + str(self.m) + "_" + str(self.step) + "/single/"), mode=7777)

        for i in range(len(b_single_prob)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))

            ax1.scatter(range(len(b_single_prob.loc[i][-dot_count:])), b_single_prob.loc[i][-dot_count:],
                        c="#cccccc", linewidths=1)
            ax1.plot(b_single_prob_ema.loc[i][-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(b_single_prob_ema_speed.loc[i][-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot(np.zeros((len(b_single_prob_ema_speed.loc[i][-dot_count:]))), linestyle="-", color="#F52D2D", linewidth=2)

            plt.xlabel("b_single:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/single/b_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

        for i in range(len(a_single_prob)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))

            ax1.scatter(range(len(a_single_prob.loc[i][-dot_count:])), a_single_prob.loc[i][-dot_count:],
                        c="#cccccc", linewidths=1)
            ax1.plot(a_single_prob_ema.loc[i][-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(a_single_prob_ema_speed.loc[i][-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot(np.zeros((len(a_single_prob_ema_speed.loc[i][-dot_count:]))), linestyle="-", color="#F52D2D", linewidth=2)

            plt.xlabel("a_single:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/single/a_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

    #######################
    # 获取区块出现次数 pass
    def get_piece_count(self, is_from_file: bool = True, b_n: int = 7, a_n: int = 4):

        if is_from_file:
            # 从文件里读取数据
            b_piece_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_count.csv").iloc[:, 1:]
            a_piece_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_count.csv").iloc[:, 1:]
            b_index = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_index.csv")
            a_index = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_index.csv")
        else:
            b_single_count, a_single_count = self.get_single_count()
            b_piece_count = pd.DataFrame({0: b_single_count.loc[0]}, b_single_count.loc[0].index).T
            a_piece_count = pd.DataFrame({0: a_single_count.loc[0]}, a_single_count.loc[0].index).T
            b_index = []
            a_index = []
            include = []
            piece_index = 0
            for i in range(1, len(b_single_count.index)):
                if piece_index == int(i / b_n) and i != len(b_single_count.index) - 1:
                    include.append(i)
                else:
                    piece_index = int(i / b_n)
                    include.append(i)
                    b_index.append(','.join(str(i) for i in include))
                    include = [i]

                if len(b_piece_count) > piece_index:
                    data_item = b_piece_count.loc[piece_index] + b_single_count.iloc[i]
                    b_piece_count.loc[piece_index] = data_item
                else:
                    b_piece_count.loc[piece_index] = b_single_count.loc[i-1]

            include = []
            piece_index = 0
            for i in range(1, len(a_single_count.index)):
                if piece_index == int(i / a_n) and i != len(a_single_count.index) - 1:
                    include.append(i)
                else:
                    piece_index = int(i / a_n)
                    include.append(i)
                    a_index.append(','.join(str(i) for i in include))
                    include = [i]

                if len(a_piece_count) > piece_index:
                    data_item = a_piece_count.loc[piece_index] + a_single_count.iloc[i]
                    a_piece_count.loc[piece_index] = data_item
                else:
                    a_piece_count.loc[piece_index] = a_single_count.loc[i][1:]

            b_piece_count.to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_count.csv")
            a_piece_count.to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_count.csv")
            pd.DataFrame(b_index).to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_index.csv")
            pd.DataFrame(a_index).to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_index.csv")

        return b_piece_count, a_piece_count, pd.DataFrame(b_index), pd.DataFrame(a_index)

    # 获取区块出现比率 pass
    def get_piece_prob(self, is_from_file: bool = True, b_n: int = 7, a_n: int = 4):
        b_piece_count, a_piece_count, b_index, a_index = self.get_piece_count()

        if is_from_file:
            # 从文件里读取数据
            b_piece_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_prob.csv").iloc[:, 1:]
            a_piece_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_prob.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_piece_prob = b_piece_count / (self.m * 5 * b_n)
            a_piece_prob = a_piece_count / (self.m * 2 * a_n)
            b_piece_prob.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_prob.csv")
            a_piece_prob.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_prob.csv")
        return b_piece_prob, a_piece_prob

    # 获取区块出现比率移动均线 pass
    def get_piece_prob_ema(self, is_from_file: bool = True):
        b_piece_prob, a_piece_prob = self.get_piece_prob()

        if is_from_file:
            # 从文件里读取数据
            b_piece_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_prob_ema.csv").iloc[:, 1:]
            a_piece_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_prob_ema.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_piece_prob_ema = b_piece_prob.T.loc[:].apply(lambda x: x.rolling(window=self.n).mean()).T
            a_piece_prob_ema = a_piece_prob.T.loc[:].apply(lambda x: x.rolling(window=self.n).mean()).T
            b_piece_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_prob_ema.csv")
            a_piece_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_prob_ema.csv")
        return b_piece_prob_ema, a_piece_prob_ema

    # 获取区块出现比率移动均线速率 pass
    def get_piece_prob_ema_speed(self, is_from_file: bool = True):

        b_piece_prob_ema, a_piece_prob_ema = self.get_piece_prob_ema()
        if is_from_file:
            # 从文件里读取数据
            b_piece_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_prob_ema_speed.csv").iloc[:, 1:]
            a_piece_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_prob_ema_speed.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_piece_prob_ema_speed = b_piece_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            a_piece_prob_ema_speed = a_piece_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            b_piece_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_piece_prob_ema_speed.csv")
            a_piece_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_piece_prob_ema_speed.csv")
        return b_piece_prob_ema_speed, a_piece_prob_ema_speed

    # 绘制单号曲线 pass
    def draw_piece(self):

        dot_count = 500

        b_piece_prob, a_piece_prob = self.get_piece_prob()
        b_piece_prob_ema, a_piece_prob_ema = self.get_piece_prob_ema()
        b_piece_prob_ema_speed, a_piece_prob_ema_speed = self.get_piece_prob_ema_speed()

        # 创建文件夹数据
        if not os.path.exists(os.path.dirname("./images/" + str(self.m) + "_" + str(self.step) + "/piece/")):
            os.makedirs(os.path.dirname("./images/" + str(self.m) + "_" + str(self.step) + "/piece/"), mode=7777)

        for i in range(len(b_piece_prob)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))

            ax1.scatter(range(len(b_piece_prob.loc[i][-dot_count:])), b_piece_prob.loc[i][-dot_count:],
                        c="#cccccc", linewidths=1)
            ax1.plot(b_piece_prob_ema.loc[i][-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(b_piece_prob_ema_speed.loc[i][-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot(np.zeros((len(b_piece_prob_ema_speed.loc[i][-dot_count:]))), linestyle="-", color="#F52D2D", linewidth=2)

            plt.xlabel("b_piece:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/piece/b_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

        for i in range(len(a_piece_prob)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))

            ax1.scatter(range(len(a_piece_prob.loc[i][-dot_count:])), a_piece_prob.loc[i][-dot_count:],
                        c="#cccccc", linewidths=1)
            ax1.plot(a_piece_prob_ema.loc[i][-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(a_piece_prob_ema_speed.loc[i][-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot(np.zeros((len(a_piece_prob_ema_speed.loc[i][-dot_count:]))), linestyle="-", color="#F52D2D", linewidth=2)

            plt.xlabel("a_piece:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/piece/a_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

    ########################
    # 获取区块出现次数 pass
    def get_parity_count(self, is_from_file: bool = True):

        if is_from_file:
            # 从文件里读取数据
            b_parity_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_count.csv").iloc[:, 1:]
            a_parity_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_count.csv").iloc[:, 1:]
        else:
            b_single_count, a_single_count = self.get_single_count()
            b_parity_count = pd.DataFrame({1: b_single_count.loc[0], 2: b_single_count.loc[1]}, b_single_count.loc[0].index).T
            a_parity_count = pd.DataFrame({1: a_single_count.loc[0], 2: a_single_count.loc[1]}, a_single_count.loc[0].index).T

            b_index = [[1], [2]]
            a_index = [[1], [2]]
            for i in range(2, len(b_single_count.index)):
                parity = common.parity(i+1)
                b_index[parity-1].append(i+1)
                data_item = b_parity_count.loc[parity] + b_single_count.iloc[i]
                b_parity_count.loc[parity] = data_item

            for i in range(2, len(a_single_count.index)):
                parity = common.parity(i+1)
                a_index[parity-1].append(i+1)
                data_item = a_parity_count.loc[parity] + a_single_count.iloc[i]
                a_parity_count.loc[parity] = data_item

            b_parity_count.to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_count.csv")
            a_parity_count.to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_count.csv")
            pd.DataFrame(b_index).to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_index.csv")
            pd.DataFrame(a_index).to_csv(
                "./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_index.csv")

        return b_parity_count, a_parity_count

    # 获取区块出现比率 pass
    def get_parity_prob(self, is_from_file: bool = True):
        b_parity_count, a_parity_count = self.get_parity_count()

        if is_from_file:
            # 从文件里读取数据
            b_parity_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_prob.csv").iloc[:, 1:]
            a_parity_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_prob.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_parity_prob = b_parity_count
            b_parity_prob.loc[0] = b_parity_count.loc[0] / (self.m * 5 * 18)
            b_parity_prob.loc[1] = b_parity_count.loc[1] / (self.m * 5 * 17)
            a_parity_prob = a_parity_count / (self.m * 2 * 12)
            b_parity_prob.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_prob.csv")
            a_parity_prob.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_prob.csv")
        return b_parity_prob, a_parity_prob

    # 获取区块出现比率移动均线 pass
    def get_parity_prob_ema(self, is_from_file: bool = True):
        b_parity_prob, a_parity_prob = self.get_parity_prob()

        if is_from_file:
            # 从文件里读取数据
            b_parity_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_prob_ema.csv").iloc[:, 1:]
            a_parity_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_prob_ema.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_parity_prob_ema = b_parity_prob.T.loc[:].apply(lambda x: x.rolling(window=self.n).mean()).T
            a_parity_prob_ema = a_parity_prob.T.loc[:].apply(lambda x: x.rolling(window=self.n).mean()).T
            b_parity_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_prob_ema.csv")
            a_parity_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_prob_ema.csv")
        return b_parity_prob_ema, a_parity_prob_ema

    # 获取区块出现比率移动均线速率 pass
    def get_parity_prob_ema_speed(self, is_from_file: bool = True):

        b_parity_prob_ema, a_parity_prob_ema = self.get_parity_prob_ema()
        if is_from_file:
            # 从文件里读取数据
            b_parity_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_prob_ema_speed.csv").iloc[:, 1:]
            a_parity_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_prob_ema_speed.csv").iloc[:, 1:]
        else:
            # 计算单号码出现比率
            b_parity_prob_ema_speed = b_parity_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            a_parity_prob_ema_speed = a_parity_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            b_parity_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_parity_prob_ema_speed.csv")
            a_parity_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_parity_prob_ema_speed.csv")
        return b_parity_prob_ema_speed, a_parity_prob_ema_speed

    # 绘制单号曲线 pass
    def draw_parity(self):

        dot_count = 500

        b_parity_prob, a_parity_prob = self.get_parity_prob()
        b_parity_prob_ema, a_parity_prob_ema = self.get_parity_prob_ema()
        b_parity_prob_ema_speed, a_parity_prob_ema_speed = self.get_parity_prob_ema_speed()

        # 创建文件夹数据
        if not os.path.exists(os.path.dirname("./images/" + str(self.m) + "_" + str(self.step) + "/parity/")):
            os.makedirs(os.path.dirname("./images/" + str(self.m) + "_" + str(self.step) + "/parity/"), mode=7777)

        for i in range(len(b_parity_prob)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))

            ax1.scatter(range(len(b_parity_prob.loc[i][-dot_count:])), b_parity_prob.loc[i][-dot_count:],
                        c="#cccccc", linewidths=1)
            ax1.plot(b_parity_prob_ema.loc[i][-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(b_parity_prob_ema_speed.loc[i][-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot(np.zeros((len(b_parity_prob_ema_speed.loc[i][-dot_count:]))), linestyle="-", color="#F52D2D", linewidth=2)

            plt.xlabel("b_parity:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/parity/b_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

        for i in range(len(a_parity_prob)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))

            ax1.scatter(range(len(a_parity_prob.loc[i][-dot_count:])), a_parity_prob.loc[i][-dot_count:],
                        c="#cccccc", linewidths=1)
            ax1.plot(a_parity_prob_ema.loc[i][-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(a_parity_prob_ema_speed.loc[i][-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot(np.zeros((len(a_parity_prob_ema_speed.loc[i][-dot_count:]))), linestyle="-", color="#F52D2D", linewidth=2)

            plt.xlabel("a_parity:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/parity/a_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

    def get_y_data(self):
        b_single_count, a_single_count = self.get_single_count()
