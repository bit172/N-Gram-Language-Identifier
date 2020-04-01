from utils import read
from byom import BYOM
from unigram import Unigram
from bigram import Bigram
from trigram import Trigram

inputs = read('input.txt')[0].split(" ")
V, N, S_FACTOR, TRAINING_FILE, TEST_FILE = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])
OUTPUT_FILE_NAME = f"./results/trace_{V}_{N}_{S_FACTOR}.txt"

if V == 3:
    BYOM = BYOM(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    BYOM.execute()
elif N == 1:
    UNIGRAM = Unigram(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    UNIGRAM.execute()
elif N == 2:
    BIGRAM = Bigram(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    BIGRAM.execute()
elif N == 3:
    TRIGRAM = Trigram(V, S_FACTOR, TRAINING_FILE, TEST_FILE, OUTPUT_FILE_NAME)
    TRIGRAM.execute()
