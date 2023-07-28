
import requests
import json
import common
from model.history_model import HistoryModel
import pandas as pd

# 获取历史数据
if __name__ == '__main__':

    response = requests.get("https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=5000&isVerify=1&pageNo=1")
    response_data = json.loads(response.text)
    data = response_data['value']['list']

    data = data
    csv_content = []
    history_model = HistoryModel()
    max_num = history_model.max("num")
    if max_num is None or max_num == "":
        max_num = 0
    data_tmp = []
    for item in data:
        # csv_content += item['lotteryDrawNum'] + "," + str.replace(item['lotteryDrawResult'],  " ",  ",") + "\n"
        codes = str(item['lotteryDrawResult']).split(" ")
        csv_content.append({
                "num": item['lotteryDrawNum'],
                "b1": codes[0],
                "b2": codes[1],
                "b3": codes[2],
                "b4": codes[3],
                "b5": codes[4],
                "a1": codes[5],
                "a2": codes[6],
            })
        if int(item['lotteryDrawNum']) > int(max_num):
            data_tmp.append({
                "num": item['lotteryDrawNum'],
                "b1": codes[0],
                "b2": codes[1],
                "b3": codes[2],
                "b4": codes[3],
                "b5": codes[4],
                "a1": codes[5],
                "a2": codes[6],
            })

    data_tmp = data_tmp[::-1]
    history_model.save_all(data_tmp)
    data_content = json.dumps(data[::-1])
    common.save_file("./data/data.json", data_content)

    csv_content = pd.DataFrame(csv_content)
    list = csv_content.T.iloc[1:8].T.to_numpy().tolist()
    csv_data = pd.DataFrame(list, csv_content["num"], ["b1", "b2", "b3", "b4", "b5", "a1", "a2"])
    csv_data.T.to_csv("./data/data.csv")
    # common.save_file("./data/data.csv", csv_content)


