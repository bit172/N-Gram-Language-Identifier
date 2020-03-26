import numpy as np
import pandas as pd
import pprint
from utils import *

v = n = s_factor = training_file = test_file = None


def execute(input_v, input_n, input_s, input_train, input_test):
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = {}
    # concatenate all strings in a given language and find the unique characters by using join
    for language, tweets in training_tweets.items():
        unique_characters[language] = (list(set(''.join(tweets))))
        unique_characters[language].sort()
