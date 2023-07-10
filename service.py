import json
import pandas as pd
import common
import matplotlib.pyplot as plt
import numpy as np


class Service:
    data = []
    before = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
              "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
              "35"]
    after = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # 初始化获取数据
    def __init__(self):
        self.data = common.read_json("./data/data.json")

    # 生成单号码概率曲线
    def get_single_data(self, m: int, step: int, ignore: int = 0):

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
    @staticmethod
    def _single_data():
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

    # 绘曲线
    def draw_single_list(self, m: int, step: int, n: int, dot_count: int = 200):
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()
        common.save_file("./images/"+str(m)+"_"+str(step)+"/single/last_stage", date_stage[-1])

        print("single_before")
        for i in range(len(self.before)):
            item_ema = common.ema(single_data_before[i], n)
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad")

            all_before_average_item = all_before_average[i]
            all_before_average_item_line = []
            for i0 in range(len(date_stage)):
                all_before_average_item_line.append(all_before_average_item)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count/80) * 16, 16))
            ax1.scatter(range(len(single_data_before[i][-dot_count:])), single_data_before[i][-dot_count:], c="#cccccc", linewidths=1)
            ax1.plot(all_before_average_item_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
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
            data_speed = pd.DataFrame(item_ema).pct_change(periods=1, fill_method="pad")

            all_after_average_item = all_after_average[i]
            all_after_average_item_line = []
            for i0 in range(len(date_stage)):
                all_after_average_item_line.append(all_after_average_item)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=((dot_count/80) * 16, 16))
            ax1.scatter(range(len(single_data_after[i][-dot_count:])), single_data_after[i][-dot_count:], c="#cccccc", linewidths=1)
            ax1.plot(all_after_average_item_line[-dot_count:], linestyle="-", color="#F52D2D", linewidth=2)
            ax1.plot(item_ema[-dot_count:], linestyle="-", color="#ababab", linewidth=2)

            ax2.plot(np.array(data_speed[-dot_count:]), linestyle="-", color="#F52D2D", linewidth=2)
            ax2.plot((np.zeros((len(data_speed[-dot_count:]),), dtype=int)), linestyle="-", color="#000000", linewidth=2)

            plt.xlabel("single_after:" + self.before[i])
            plt.savefig("./images/"+str(m)+"_"+str(step)+"/single/after_" + self.after[i] + ".jpg", format="jpg", bbox_inches="tight", pad_inches=0,
                        transparent=True, dpi=64)
            plt.axis("off")
            plt.clf()
            plt.close("all")

    def draw_parity_list(self, m: int, step: int, n: int, dot_count: int = 200):
        all_before_average, all_after_average, single_data_before, single_data_after, date_stage, date_index = self._single_data()
        common.save_file("./images/"+str(m)+"_"+str(step)+"/single/last_stage", date_stage[-1])

        for i in range(len(self.before)):
            common.parity(self.before)
