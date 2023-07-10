import os
import numpy as np
import json


# 保存文件
def save_file(real_path, data_content):

    if not os.path.exists(os.path.dirname(real_path)):
        os.makedirs(os.path.dirname(real_path), mode=7777)
    try:
        with open(real_path, "w+", encoding="utf8") as fp:
            fp.write(data_content)
    except Exception as e:
        print(e)
    return


# 读取json文件
def read_json(real_path):
    try:
        with open(real_path, "r", encoding="utf8") as fp:
            data = json.load(fp)
            fp.close()
    except Exception as e:
        print(e)
    return data


# 计算动态均值，m：窗口大小
def ema(values, m):
    values = np.array(values)
    res = []
    for index in range(1, len(values)):
        start = index - m if index - m >= 0 else 0
        res.append(np.average(values[start:index]))
    return res


# 获取奇偶 1奇，2偶
def parity(value):
    if int(value) % 2 == 0:
        return 2
    else:
        return 1
