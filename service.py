import json
import pandas as pd
import common
import matplotlib.pyplot as plt
import numpy as np
import os


class Service(object):
    data = []
    m=50
    step=1
    before = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
              "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
              "35"]
    after = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # 初始化获取数据
    def __init__(self):
        self.data = common.read_json("./data/data.json")

    def init(self, m: int, step: int):
        self.m = m
        self.step = step

    # 生成概率基础数据
    def get_single_data(self, ignore: int = 0):

        m = self.m
        step = self.step

        data = self.data[:len(self.data) - ignore]
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
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single_all_before_average.json", json.dumps(all_before_average))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single_all_after_average.json", json.dumps(all_after_average))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single_data_before.json", json.dumps(single_data_before))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single_data_after.json", json.dumps(single_data_after))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single_data_stage.json", json.dumps(date_stage))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single_data_index.json", json.dumps(date_index))
        return all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index

    # 获取数量
    def _single_data(self):

        m = self.m
        step = self.step

        all_before_average = common.read_json("./data/"+str(m)+"_"+str(step)+"/single_all_before_average.json")
        all_after_average = common.read_json("./data/"+str(m)+"_"+str(step)+"/single_all_after_average.json")
        single_data_before = common.read_json("./data/"+str(m)+"_"+str(step)+"/single_data_before.json")
        single_data_after = common.read_json("./data/"+str(m)+"_"+str(step)+"/single_data_after.json")
        date_stage = common.read_json("./data/"+str(m)+"_"+str(step)+"/single_data_stage.json")
        date_index = common.read_json("./data/"+str(m)+"_"+str(step)+"/single_data_index.json")
        return all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index

    # 获取数量
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

    # 绘制单号曲线
    def draw_single_lines(self, n: int, dot_count: int = 200):
        m = self.m
        step = self.step
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()
        common.save_file("./images/"+str(m)+"_"+str(step)+"/single/last_stage", date_stage[-1])

        print("single_before")
        before_speed_data = []
        after_speed_data = []
        for i in range(len(self.before)):
            item_ema = common.ema(single_data_before[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            before_speed_data.append(data_speed.to_numpy().tolist())

            all_before_average_item = all_before_average[i]
            average_line = []
            for i0 in range(len(date_stage)):
                average_line.append(all_before_average_item)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count/80) * 10, 13))
            ax1.scatter(range(len(single_data_before[i][-dot_count:])), single_data_before[i][-dot_count:], c="#cccccc", linewidths=1)
            ax1.plot(average_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000", linewidth=2)

            plt.xlabel("single_before:" + self.before[i])
            plt.savefig("./images/"+str(m)+"_"+str(step)+"/single/before_" + self.before[i] + ".jpg", format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

        print("single_after")
        for i in range(len(self.after)):
            item_ema = common.ema(single_data_after[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            after_speed_data.append(data_speed.to_numpy().tolist())

            all_after_average_item = all_after_average[i]
            average_line = []
            for i0 in range(len(date_stage)):
                average_line.append(all_after_average_item)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count/80) * 10, 13))
            ax1.scatter(range(len(single_data_after[i][-dot_count:])), single_data_after[i][-dot_count:], c="#cccccc", linewidths=1)
            ax1.plot(average_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000", linewidth=2)

            plt.xlabel("single_after:" + self.before[i])
            plt.savefig("./images/"+str(m)+"_"+str(step)+"/single/after_" + self.after[i] + ".jpg", format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

        # 存储均值速率数据
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single/single_speed_data.json", json.dumps(before_speed_data))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/single/single_speed_data.json", json.dumps(after_speed_data))

    # 绘制奇偶曲线
    def draw_parity_lines(self, n: int, dot_count: int = 200):
        m = self.m
        step = self.step
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()
        common.save_file("./images/"+str(m)+"_"+str(step)+"/parity/last_stage", date_stage[-1])

        before_lines_data = {"1": np.zeros((len(date_stage),), dtype=float).tolist(), "2": np.zeros((len(date_stage),), dtype=float).tolist()}
        after_lines_data = {"1": np.zeros((len(date_stage),), dtype=float).tolist(), "2": np.zeros((len(date_stage),), dtype=float).tolist()}

        for i in range(len(self.before)):
            parity = common.parity(self.before[i])
            for i0 in range(len(date_stage)):
                before_lines_data[str(parity)][i0] += single_data_before[i][i0]

        for i in range(len(self.after)):
            parity = common.parity(self.after[i])
            for i0 in range(len(date_stage)):
                after_lines_data[str(parity)][i0] += single_data_after[i][i0]

        before_lines_average = {"1": np.average(np.array(before_lines_data["1"])), "2": np.average(np.array(before_lines_data["2"]))}
        after_lines_average = {"1": np.average(np.array(after_lines_data["1"])), "2": np.average(np.array(after_lines_data["2"]))}

        before_speed_data = []
        after_speed_data = []
        for i in ["1", "2"]:
            average_line = []
            for i0 in range(len(date_stage)):
                average_line.append(before_lines_average[i])

            item_ema = common.ema(before_lines_data[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad").stack()
            before_speed_data.append(data_speed)

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
        common.save_file("./data/"+str(m)+"_"+str(step)+"/parity/before_lines_data.json", json.dumps(np.array(before_lines_data).tolist()))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/parity/after_lines_data.json", json.dumps(np.array(after_lines_data).tolist()))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/parity/before_speed_data.json", json.dumps(np.array(before_speed_data).tolist()))
        common.save_file("./data/"+str(m)+"_"+str(step)+"/parity/after_speed_data.json", json.dumps(np.array(after_speed_data).tolist()))

    # 绘制分块曲线
    def draw_piece_lines(self, b_n: int, a_n: int, n: int, dot_count: int = 200):
        m = self.m
        step = self.step
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()

        before_pieces_data = []
        after_pieces_data = []
        before_pieces = []
        after_pieces = []
        for i in range(len(self.before)):
            piece_index = int(i/b_n)
            if len(before_pieces_data) <= piece_index:
                before_pieces_data.append(single_data_before[i])
                before_pieces.append([i])
            else:
                before_pieces_data[piece_index] = (np.array(before_pieces_data[piece_index]) + np.array(single_data_before[i])).tolist()
                before_pieces[piece_index].append(i)

        for i in range(len(self.after)):
            piece_index = int(i/a_n)
            if len(after_pieces_data) <= piece_index:
                after_pieces_data.append(single_data_before[i])
                after_pieces.append([i])
            else:
                after_pieces_data[piece_index] = (np.array(after_pieces_data[piece_index]) + np.array(single_data_after[i])).tolist()
                after_pieces[piece_index].append(i)

        common.save_file("./images/"+str(m)+"_"+str(step)+"/piece/before_pieces", json.dumps(before_pieces))
        common.save_file("./images/"+str(m)+"_"+str(step)+"/piece/after_pieces", json.dumps(after_pieces))

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
