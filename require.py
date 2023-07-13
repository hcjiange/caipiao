
import requests
import json
import common

# 获取历史数据
if __name__ == '__main__':

    response = requests.get("https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=5000&isVerify=1&pageNo=1")
    response_data = json.loads(response.text)
    data = response_data['value']['list']

    data = data[30:]
    csv_content = ""
    for item in data:
        csv_content += item['lotteryDrawNum'] + "," + str.replace(item['lotteryDrawResult'],  " ",  ",") + "\n"

    data_content = json.dumps(data[::-1])
    common.save_file("./data/data.json", data_content)
    common.save_file("./data/data.csv", csv_content)
