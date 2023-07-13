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
            b_single_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_count.csv")
            a_single_count = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_count.csv")
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
            b_single_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob.csv")
            a_single_prob = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob.csv")
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
            b_single_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema.csv")
            a_single_prob_ema = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema.csv")
        else:
            # 计算单号码出现比率
            b_single_prob_ema = b_single_prob.T.loc[:].apply(lambda x: x.rolling(window=30).mean()).T
            a_single_prob_ema = a_single_prob.T.loc[:].apply(lambda x: x.rolling(window=30).mean()).T
            b_single_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema.csv")
            a_single_prob_ema.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema.csv")
        return b_single_prob_ema, a_single_prob_ema

    # 获取单号码出现比率移动均线速率 pass
    def get_single_prob_ema_speed(self, is_from_file: bool = True):

        b_single_prob_ema, a_single_prob_ema = self.get_single_prob_ema()
        if is_from_file:
            # 从文件里读取数据
            b_single_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema_speed.csv")
            a_single_prob_ema_speed = pd.read_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema_speed.csv")
        else:
            # 计算单号码出现比率
            b_single_prob_ema_speed = b_single_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            a_single_prob_ema_speed = a_single_prob_ema.T.loc[:].pct_change(periods=1, fill_method="pad").T
            b_single_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/b_single_prob_ema_speed.csv")
            a_single_prob_ema_speed.to_csv("./data/" + str(self.m) + "_" + str(self.step) + "/a_single_prob_ema_speed.csv")
        return b_single_prob_ema_speed, a_single_prob_ema_speed

    # 生成概率基础数据
    def get_base_data(self, ignore: int = 0):

        m = self.m
        step = self.step

        data = self.data
        data_count = len(data)

        single_data_before = []
        single_data_after = []
        date_stage = []
        date_index = []

        for i in range(len(self.before)):
            single_data_before.append([])
        for i1 in range(len(self.after)):
            single_data_after.append([])

        all_before_average = []
        all_after_average = []
        all_before_count, all_after_count = self.get_count(data)
        for i in range(len(self.before)):
            all_before_average.append(all_before_count[self.before[i]] / (data_count * 5))
        for i1 in range(len(self.after)):
            all_after_average.append(all_after_count[self.after[i1]] / (data_count * 2))

        for i in range(len(data)):
            item = data[i]
            if m <= i < len(data) and (i % step == 0 or i == len(data) - 1):
                single_data_before_item, single_data_after_item = self.get_count(data[i - m:i])
                date_stage.append(item['lotteryDrawNum'])
                date_index.append(i)
                for i0 in range(len(self.before)):
                    single_data_before[i0].append(single_data_before_item[self.before[i0]] / (m * 5))
                for i0 in range(len(self.after)):
                    single_data_after[i0].append(single_data_after_item[self.after[i0]] / (m * 2))

        # 存储数据
        common.save_file("./data/" + str(m) + "_" + str(step) + "/single_all_before_average.json",
                         json.dumps(all_before_average))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/single_all_after_average.json",
                         json.dumps(all_after_average))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/single_data_before.json",
                         json.dumps(single_data_before))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/single_data_after.json",
                         json.dumps(single_data_after))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/single_data_stage.json", json.dumps(date_stage))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/single_data_index.json", json.dumps(date_index))
        return all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index

    # 获取数量
    def _single_data(self):

        m = self.m
        step = self.step

        all_before_average = common.read_json("./data/" + str(m) + "_" + str(step) + "/single_all_before_average.json")
        all_after_average = common.read_json("./data/" + str(m) + "_" + str(step) + "/single_all_after_average.json")
        single_data_before = common.read_json("./data/" + str(m) + "_" + str(step) + "/single_data_before.json")
        single_data_after = common.read_json("./data/" + str(m) + "_" + str(step) + "/single_data_after.json")
        date_stage = common.read_json("./data/" + str(m) + "_" + str(step) + "/single_data_stage.json")
        date_index = common.read_json("./data/" + str(m) + "_" + str(step) + "/single_data_index.json")
        return all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index

    # 绘制单号曲线
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
            plt.show()
            exit()

            # ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
            # ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000",
            #          linewidth=2)

            plt.xlabel("single_before:" + self.before[i])
            plt.savefig("./images/" + str(self.m) + "_" + str(self.step) + "/single/before_" + self.before[i] + ".jpg",
                        format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

    # 绘制奇偶曲线
    def get_parity_data(self):
        m = self.m
        step = self.step
        n = self.n
        dot_count = self.dot_count
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()
        common.save_file("./images/" + str(m) + "_" + str(step) + "/parity/last_stage", date_stage[-1])

        before_lines_data = {"1": np.zeros((len(date_stage),), dtype=float).tolist(),
                             "2": np.zeros((len(date_stage),), dtype=float).tolist()}
        after_lines_data = {"1": np.zeros((len(date_stage),), dtype=float).tolist(),
                            "2": np.zeros((len(date_stage),), dtype=float).tolist()}

        for i in range(len(self.before)):
            parity = common.parity(self.before[i])
            for i0 in range(len(date_stage)):
                before_lines_data[str(parity)][i0] += single_data_before[i][i0]

        for i in range(len(self.after)):
            parity = common.parity(self.after[i])
            for i0 in range(len(date_stage)):
                after_lines_data[str(parity)][i0] += single_data_after[i][i0]

        before_lines_average = {"1": np.average(np.array(before_lines_data["1"])),
                                "2": np.average(np.array(before_lines_data["2"]))}
        after_lines_average = {"1": np.average(np.array(after_lines_data["1"])),
                               "2": np.average(np.array(after_lines_data["2"]))}

        before_speed_data = []
        after_speed_data = []
        for i in ["1", "2"]:
            average_line = []
            for i0 in range(len(date_stage)):
                average_line.append(before_lines_average[i])

            item_ema = common.ema(before_lines_data[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            before_speed_data.append(data_speed)

            if self.draw:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))
                ax1.scatter(range(len(before_lines_data[i][-dot_count:])), before_lines_data[i][-dot_count:],
                            c="#cccccc", linewidths=1)
                ax1.plot(average_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
                ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

                ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
                ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000",
                         linewidth=2)

                plt.xlabel("parity_before:" + i)
                plt.savefig("./images/" + str(m) + "_" + str(step) + "/parity/before_" + i + ".jpg",
                            format="jpg", bbox_inches="tight", pad_inches=0,
                            transparent=True, dpi=64)
                plt.axis("off")
                plt.clf()
                plt.close("all")

            average_line = []
            for i0 in range(len(date_stage)):
                average_line.append(after_lines_average[i])

            item_ema = common.ema(after_lines_data[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            after_speed_data.append(data_speed)

            if self.draw:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))
                ax1.scatter(range(len(after_lines_data[i][-dot_count:])), after_lines_data[i][-dot_count:],
                            c="#cccccc", linewidths=1)
                ax1.plot(average_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
                ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

                ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
                ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000",
                         linewidth=2)

                plt.xlabel("parity_after:" + i)
                plt.savefig("./images/" + str(m) + "_" + str(step) + "/parity/after_" + i + ".jpg",
                            format="jpg", bbox_inches="tight", pad_inches=0,
                            transparent=True, dpi=64)
                plt.axis("off")
                plt.clf()
                plt.close("all")

        # 存储数据
        common.save_file("./data/" + str(m) + "_" + str(step) + "/parity/before_lines_data.json",
                         json.dumps(np.array(before_lines_data).tolist()))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/parity/after_lines_data.json",
                         json.dumps(np.array(after_lines_data).tolist()))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/parity/before_speed_data.json",
                         json.dumps(np.array(before_speed_data).tolist()))
        common.save_file("./data/" + str(m) + "_" + str(step) + "/parity/after_speed_data.json",
                         json.dumps(np.array(after_speed_data).tolist()))
        return before_lines_data, after_lines_data, before_speed_data, after_speed_data

    # 绘制分块曲线
    def get_piece_data(self, b_n: int, a_n: int):
        m = self.m
        step = self.step
        n = self.n
        dot_count = self.dot_count
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()

        before_pieces_data = []
        after_pieces_data = []
        before_pieces = []
        after_pieces = []
        for i in range(len(self.before)):
            piece_index = int(i / b_n)
            if len(before_pieces_data) <= piece_index:
                before_pieces_data.append(single_data_before[i])
                before_pieces.append([i])
            else:
                before_pieces_data[piece_index] = (
                            np.array(before_pieces_data[piece_index]) + np.array(single_data_before[i])).tolist()
                before_pieces[piece_index].append(i)

        for i in range(len(self.after)):
            piece_index = int(i / a_n)
            if len(after_pieces_data) <= piece_index:
                after_pieces_data.append(single_data_before[i])
                after_pieces.append([i])
            else:
                after_pieces_data[piece_index] = (
                            np.array(after_pieces_data[piece_index]) + np.array(single_data_after[i])).tolist()
                after_pieces[piece_index].append(i)

        common.save_file("./images/" + str(m) + "_" + str(step) + "/piece/before_pieces", json.dumps(before_pieces))
        common.save_file("./images/" + str(m) + "_" + str(step) + "/piece/after_pieces", json.dumps(after_pieces))

        before_speed_data = []
        after_speed_data = []
        for i in range(len(before_pieces_data)):
            average_line = []
            before_lines_average = np.average(np.array(before_pieces_data[i]))
            for i0 in range(len(date_stage)):
                average_line.append(before_lines_average)

            item_ema = common.ema(before_pieces_data[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            before_speed_data.append(data_speed)

            if self.draw:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))
                ax1.scatter(range(len(before_pieces_data[i][-dot_count:])), before_pieces_data[i][-dot_count:],
                            c="#cccccc", linewidths=1)
                ax1.plot(average_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
                ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

                ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
                ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000",
                         linewidth=2)

                plt.xlabel("piece_before:" + str(i))
                plt.savefig("./images/" + str(m) + "_" + str(step) + "/piece/before_" + str(i) + ".jpg",
                            format="jpg", bbox_inches="tight", pad_inches=0,
                            transparent=True, dpi=64)
                plt.axis("off")
                plt.clf()
                plt.close("all")

        for i in range(len(after_pieces_data)):
            average_line = []
            after_lines_average = np.average(np.array(after_pieces_data[i]))
            for i0 in range(len(date_stage)):
                average_line.append(after_lines_average)

            item_ema = common.ema(after_pieces_data[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            after_speed_data.append(data_speed)

            if self.draw:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count / 80) * 10, 13))
                ax1.scatter(range(len(after_pieces_data[i][-dot_count:])), after_pieces_data[i][-dot_count:],
                            c="#cccccc", linewidths=1)
                ax1.plot(average_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
                ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

                ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
                ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000",
                         linewidth=2)

                plt.xlabel("piece_after:" + str(i))
                plt.savefig("./images/" + str(m) + "_" + str(step) + "/piece/after_" + str(i) + ".jpg",
                            format="jpg", bbox_inches="tight", pad_inches=0,
                            transparent=True, dpi=64)
                plt.axis("off")
                plt.clf()
                plt.close("all")

        return before_pieces_data, after_pieces_data, before_speed_data, after_speed_data
