import service

# 中控
if __name__ == '__main__':

    # m：窗口大小，step：步长, n：均线窗口大小
    m, step, n = 50, 1, 50

    s = service.Service()
    s.init(m=m, step=step)

    # 获取单号码基础数据
    s.get_single_data()
    # 绘制单号码曲线
    s.draw_single_lines(n=n, dot_count=500)
    # 绘制奇偶曲线
    s.draw_parity_lines(n=n, dot_count=500)
    # 绘制分块曲线
    s.draw_piece_lines(b_n=7, a_n=4, n=n, dot_count=500)
