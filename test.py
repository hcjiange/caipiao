import service
import pandas as pd
import os

# 训练
if __name__ == '__main__':

    m_list = [3, 8, 16, 32, 56, 80, 100]
    m_list = [32]
    num_list = ["23078", "23078", "23078", "23078", "23078", "23078", "23078", "23078"]
    num_list = ["23088"]
    s = service.Service()
    for m_item in m_list:
        # m：窗口大小，step：步长, n：均线窗口大小
        m, step, n, draw, dot_count = m_item, 1, 15, False, 500
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        b_single_count, a_single_count = s.get_single_count()

        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/data/data.csv").iloc[:, 1:]

        num = "23078"
        num1 = str(int(num)+1)

        b_sort = b_single_count[num].sort_values(ascending=False).index + 1
        a_sort = a_single_count[num].sort_values(ascending=False).index + 1

        print(m, num1, ",".join(data[num1][:-2].astype("string")), ",".join(data[num1][-2:].astype("string")))
        print("host", ",".join(b_sort[5:10].astype("string")), "||", ",".join(a_sort[:6].astype("string")))
        print("wean", ",".join(b_sort[15:-10].astype("string")))
        print("cold", ",".join(b_sort[-5:].astype("string")), "||", ",".join(a_sort[-6:].astype("string")))

        print()


