import service
import analysis
import argparse


# 中控
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-before", type=int, default=1)
    args = parser.parse_args()

    s = analysis.Analysis()
    s.to_predit(args.before)
