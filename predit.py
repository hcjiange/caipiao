import service
import analysis
import argparse


# 中控
if __name__ == '__main__':
    #
    # # m：窗口大小，step：步长, n：均线窗口大小
    # m, step, n, draw, dot_count = 20, 1, 20, False, 200
    #
    # s = service.Service()
    # s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    #
    # # 获取单号码基础数据
    # s.get_base_data()
    # # 绘制单号码曲线
    # s.get_single_data()
    # # 绘制奇偶曲线
    # s.get_parity_data()
    # # 绘制分块曲线
    # s.get_piece_data(b_n=7, a_n=4)


    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-before", type=int, default=1)
    args = parser.parse_args()

    s = analysis.Analysis()
    s.to_predit(args.before)
