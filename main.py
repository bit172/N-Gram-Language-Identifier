from time import time
from bigram import Bigram
from byom import BYOM
from trigram import Trigram
from unigram import Unigram
from utils import *
import matplotlib.pyplot as plt

inputs = read('input.txt')[0].strip().split(" ")
V, N, S_FACTOR, TRAINING_FILE, TEST_FILE = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])
OUTPUT_FILE_NAME = f"./results/trace_{V}_{N}_{S_FACTOR}.txt"

delete_results()

increment_value = 0.1
t1 = time()
d = increment_value
x = []
y = []
while d < 1:
    x.append(d)
    if V == 3:
        print(f"BYOM: V = {V} n = 3 d = {d}")
        OUR_MODEL = BYOM(V, d, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
        weighted_average_F1 = OUR_MODEL.execute()
    elif N == 1:
        print(f"unigram: V = {V} d = {d}")
        UNIGRAM = Unigram(V, d, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
        weighted_average_F1 = UNIGRAM.execute()
    elif N == 2:
        print(f"bigram: V = {V} d = {d}")
        BIGRAM = Bigram(V, d, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
        weighted_average_F1 = BIGRAM.execute()
    elif N == 3:
        print(f"trigram: V = {V} d = {d}")
        TRIGRAM = Trigram(V, d, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
        weighted_average_F1 = TRIGRAM.execute()
    y.append(weighted_average_F1)
    # print(y)
    d += increment_value

plt.plot(x, y)
# naming the x axis
plt.xlabel('Smoothing Factor')
# naming the y axis
plt.ylabel('Weighted Average F1')
plt.show()
t2 = time()

delete_results()
print(f"execution time: {t2 - t1}s")
