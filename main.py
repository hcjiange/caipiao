import service

# 中控
if __name__ == '__main__':

    s = service.Service()

    # m：窗口大小，step：步长, n：均线窗口大小
    m, step, n = 50, 1, 15

    # 获取单号码基础数据
    s.get_single_data(m=m, step=step)
    # 绘制单号码曲线
    s.draw_single_list(m=m, step=step, n=n, dot_count=200)
