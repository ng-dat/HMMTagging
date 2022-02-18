import numpy as np
import sys

from model import  HMM

def main(test_file_path):
    model = HMM()
    model.load_model()

    outputs = []
    with open(test_file_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines[:3]:
            words = [w.strip() for w in line.split(' ')]
            pred_tags = model.viterbi(words)
            print(words)
            print(pred_tags)


if __name__ == '__main__':
    test_file_path = sys.argv[1]
    main(test_file_path)

