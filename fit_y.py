import service
import analysis


# 训练
if __name__ == '__main__':

    m, step, n, draw, dot_count = 16, 1, 15, False, 200
    s = service.Service()
    s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    s.get_y_data(False)
