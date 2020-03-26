import numpy as np
import pandas as pd
from utils import *

inputs = read('input.txt')[0].split(" ")
(v, n, s_factor, training_file, test_file) = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])

raw_training_tweets = read(training_file)
print(process_tweets(raw_training_tweets, v))