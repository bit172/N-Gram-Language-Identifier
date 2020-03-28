from utils import *
import unigram
import bigram
import bigram_v2
import trigram
import trigram_v2
from time import time

inputs = read('input.txt')[0].split(" ")
(v, n, s_factor, training_file, test_file) = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])
t1 = time()
if n == 1:
    unigram.execute(v, n, s_factor, training_file, test_file)
if n == 2:
    bigram_v2.execute(v, n, s_factor, training_file, test_file)
if n == 3:
    trigram_v2.execute(v, n, s_factor, training_file, test_file)
t2 = time()

print(f"execution time: {t2-t1}s")
