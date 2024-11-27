#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse
import os
from recorder import Recorder


def fed_args():
    """
    Arguments for running postprocessing on FedD3
    :return: Arguments for postprocessing on FedD3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-rr', '--sys-res_root', type=str, required=True, help='Root directory of the results')

    args = parser.parse_args()
    return args


def res_eval():
    """
    Main function for result evaluation
    """
    #args = fed_args()
    path_r="results"
    recorder = Recorder()
    res_files = [f for f in os.listdir(path_r)]

    for f in res_files:
        recorder.load(os.path.join(path_r, f), label=f)
    recorder.plot()

    #plt.show()#服务器无法显示图片
    plt.savefig('savefig_example.png')


if __name__ == "__main__":
    res_eval()