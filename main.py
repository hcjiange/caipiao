import service
import analysis


# 中控
if __name__ == '__main__':

    s = service.Service()

    m_list = [3, 8, 16, 32, 56, 80, 100]
    for m_item in m_list:
        # m：窗口大小，step：步长, n：均线窗口大小
        m, step, n, draw, dot_count = m_item, 1, 15, False, 500
        s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
        s.get_single_count(False)
        # s.get_single_prob(False)
        # s.get_single_prob_ema(False)
        # s.get_single_prob_ema_speed(False)
        # s.draw_single()

        s.get_piece_count(False)
        # s.get_piece_prob(False)
        # s.get_piece_prob_ema(False)
        # s.get_piece_prob_ema_speed(False)
        # s.draw_piece()

        s.get_parity_count(False)
        # s.get_parity_prob(False)
        # s.get_parity_prob_ema(False)
        # s.get_parity_prob_ema_speed(False)
    # s.draw_parity()
    #
    #
    # # m：窗口大小，step：步长, n：均线窗口大小
    # m, step, n, draw, dot_count = 16, 1, 15, False, 500
    # s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    # s.get_single_count(False)
    # s.get_single_prob(False)
    # s.get_single_prob_ema(False)
    # s.get_single_prob_ema_speed(False)
    # # s.draw_single()
    #
    # s.get_piece_count(False)
    # s.get_piece_prob(False)
    # s.get_piece_prob_ema(False)
    # s.get_piece_prob_ema_speed(False)
    # # s.draw_piece()
    #
    # s.get_parity_count(False)
    # s.get_parity_prob(False)
    # s.get_parity_prob_ema(False)
    # s.get_parity_prob_ema_speed(False)
    # # s.draw_parity()
    #
    #
    # m, step, n, draw, dot_count = 32, 1, 15, False, 500
    # s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    # s.get_single_count(False)
    # s.get_single_prob(False)
    # s.get_single_prob_ema(False)
    # s.get_single_prob_ema_speed(False)
    # # s.draw_single()
    #
    # s.get_piece_count(False)
    # s.get_piece_prob(False)
    # s.get_piece_prob_ema(False)
    # s.get_piece_prob_ema_speed(False)
    # # s.draw_piece()
    #
    # s.get_parity_count(False)
    # s.get_parity_prob(False)
    # s.get_parity_prob_ema(False)
    # s.get_parity_prob_ema_speed(False)
    # # s.draw_parity()
    #
    #
    # m, step, n, draw, dot_count = 56, 1, 15, False, 500
    # s.init(m=m, step=step, n=n, draw=draw, dot_count=dot_count)
    # s.get_single_count(False)
    # s.get_single_prob(False)
    # s.get_single_prob_ema(False)
    # s.get_single_prob_ema_speed(False)
    # # s.draw_single()
    #
    # s.get_piece_count(False)
    # s.get_piece_prob(False)
    # s.get_piece_prob_ema(False)
    # s.get_piece_prob_ema_speed(False)
    # # s.draw_piece()
    #
    # s.get_parity_count(False)
    # s.get_parity_prob(False)
    # s.get_parity_prob_ema(False)
    # s.get_parity_prob_ema_speed(False)
    # # s.draw_parity()

