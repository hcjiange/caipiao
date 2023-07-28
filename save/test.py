import service
from model.single_count_model import SingleCountModel

# 训练
if __name__ == '__main__':

    m, step, n, draw, dot_count = 56, 1, 15, False, 200
    s = service.Service()
    s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    b_single_count, a_single_count = s.get_single_count()

    model = SingleCountModel()
    max_num = model.max("num")
    if max_num is None:
        max_num = 0

    data_tmp = []
    for key in b_single_count:
        if int(key) > max_num:
            data_item_tmp = {"num": key}
            for code in b_single_count[key].index:
                data_item_tmp["b"+str(code+1)] = b_single_count[key][code]
            for code in a_single_count[key].index:
                data_item_tmp["a"+str(code+1)] = a_single_count[key][code]
            data_tmp.append(data_item_tmp)
    # print(data_tmp)
    # exit()
    model.save_all(data_tmp)
