import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from lda import LDA

if __name__ == "__main__":
    lda = LDA(sys.argv[1], lda_weight=float(sys.argv[2]))

    with open(os.path.join(sys.argv[1], "file-list1"), 'r') as f:
        target_files = [i.strip() for i in f.readlines()]
    result = []
    for target_file in target_files:
        result.append(lda.retrieve_single(target_file, [5]))
    print(np.mean(np.where(np.array(result) == 1, 1, 0)))
    print(np.mean(np.where(np.array(result) <= 5, 1, 0)))
    plt.hist(result)
    plt.savefig("result.jpg")
    
    lda.retrieve_multiple(["D00076", "D01032", "D01350", "D02582", "D05005"], [10, 5, 1])

