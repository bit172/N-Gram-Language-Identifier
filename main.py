from utils import *
import unigram
import bigram
import trigram
import byom
from time import time

inputs = read('input.txt')[0].split(" ")
v, n, s_factor, training_file, test_file = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])
t1 = time()
if v == 3:
    byom.execute(v, n, s_factor, training_file, test_file)
elif n == 1:
    unigram.execute(v, n, s_factor, training_file, test_file)
elif n == 2:
    bigram.execute(v, n, s_factor, training_file, test_file)
elif n == 3:
    trigram.execute(v, n, s_factor, training_file, test_file)
t2 = time()

print(f"execution time: {t2-t1}s")
