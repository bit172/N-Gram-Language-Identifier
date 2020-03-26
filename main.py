from utils import *
import unigram
import bigram

inputs = read('input.txt')[0].split(" ")
(v, n, s_factor, training_file, test_file) = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])

if n == 1:
    unigram.execute(v, n, s_factor, training_file, test_file)
if n == 2:
    bigram.execute(v, n, s_factor, training_file, test_file)

