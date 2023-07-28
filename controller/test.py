import service
import numpy as np
import pandas as pd
import os

# 训练
if __name__ == '__main__':

    m, step, n, draw, dot_count = 8, 1, 15, False, 200
    s = service.Service()
    s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    b_single_count, a_single_count = s.get_single_count()

    org_data = pd.read_csv(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/data.csv").iloc[:, 1:].T

    num = "23081"
    before_num = str(int(num) - 1)
    print("当期:",  " ".join(org_data.loc[num].astype("string")[:-2]))

    data = b_single_count[before_num]
    sort_data = data.sort_values()
    items1 = (sort_data.iloc[0:10].index + 1).sort_values()
    items2 = (sort_data.iloc[10:-10].index + 1).sort_values()
    items3 = (sort_data.iloc[-10:].index + 1).sort_values()
    print("热:", " ".join(items1.astype("string")))
    print("温:", " ".join(items2.astype("string")))
    print("冷:", " ".join(items3.astype("string")))

    print("当期:",  " ".join(org_data.loc[num].astype("string")[-2:]))
    data = a_single_count[before_num]
    sort_data = data.sort_values()
    items1 = (sort_data.iloc[0:4].index + 1).sort_values()
    items2 = (sort_data.iloc[4:8].index + 1).sort_values()
    items3 = (sort_data.iloc[8:].index + 1).sort_values()
    print("热:", " ".join(items1.astype("string")))
    print("温:", " ".join(items2.astype("string")))
    print("冷:", " ".join(items3.astype("string")))


    exit()

    m, step, n, draw, dot_count = 56, 1, 15, False, 200
    s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    b_single_count, a_single_count = s.get_single_count()

    # print(org_data)
    # exit()
    # org_data = pd.DataFrame(org_data, [])

    data = b_single_count[before_num]
    sort_data = data.sort_values()
    print("热:", " ".join((sort_data.iloc[0:10].index + 1).sort_values().astype("string")))
    print("温:", " ".join((sort_data.iloc[10:-10].index + 1).sort_values().astype("string")))
    print("冷:", " ".join((sort_data.iloc[-10:].index + 1).sort_values().astype("string")))


    m, step, n, draw, dot_count = 32, 1, 15, False, 200
    s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    b_single_count, a_single_count = s.get_single_count()

    # print(org_data)
    # exit()
    # org_data = pd.DataFrame(org_data, [])

    data = b_single_count[before_num]
    sort_data = data.sort_values()
    print("热:", " ".join((sort_data.iloc[0:10].index + 1).sort_values().astype("string")))
    print("温:", " ".join((sort_data.iloc[10:-10].index + 1).sort_values().astype("string")))
    print("冷:", " ".join((sort_data.iloc[-10:].index + 1).sort_values().astype("string")))
