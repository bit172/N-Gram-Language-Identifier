from time import time
from utils import read
from byom import BYOM
from unigram import Unigram
from bigram import Bigram
from trigram import Trigram

inputs = read('input.txt')[0].strip().split(" ")
V, N, S_FACTOR, TRAINING_FILE, TEST_FILE = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])
OUTPUT_FILE_NAME = f"./results/trace_{V}_{N}_{S_FACTOR}.txt"

t1 = time()
if V == 3:
    print(f"BYOM: V = {V} n = 3 d = {S_FACTOR}")
    BYOM = BYOM(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    BYOM.execute()
elif N == 1:
    print(f"unigram: V = {V} d = {S_FACTOR}")
    UNIGRAM = Unigram(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    UNIGRAM.execute()
elif N == 2:
    print(f"bigram: V = {V} d = {S_FACTOR}")
    BIGRAM = Bigram(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    BIGRAM.execute()
elif N == 3:
    print(f"trigram: V = {V} d = {S_FACTOR}")
    TRIGRAM = Trigram(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    TRIGRAM.execute()
t2 = time()

print(f"execution time: {t2 - t1}s")
