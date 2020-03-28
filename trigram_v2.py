import numpy as np
import pandas as pd
from utils import *
import math

v = n = s_factor = training_file = test_file = None


def create_3d_arrays(unique_characters, initial_val):
    three_d_arrs = {}
    for lang, chars in unique_characters.items():
        size = len(chars)
        three_d_arrs[lang] = np.full((size, size, size), initial_val, dtype=np.float64)
    return three_d_arrs


def split_tweet_into_trigrams(tweet):
    for i in range(len(tweet) - 2):
        c1 = tweet[i]
        c2 = tweet[i + 1]
        c3 = tweet[i + 2]
        yield c1, c2, c3


def execute(input_v, input_n, input_s, input_train, input_test):
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = unique_c_v2(training_tweets, v)

    frequency_counts = create_3d_arrays(unique_characters, s_factor)

    for lang, tweets in training_tweets.items():
        for tweet in tweets:
            for c in split_tweet_into_trigrams(tweet):
                c1, c2, c3 = c
                if c1 not in unique_characters[lang]:
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang]:
                    c2 = '<NOT-APPEAR>'
                if c3 not in unique_characters[lang]:
                    c3 = '<NOT-APPEAR>'
                c1_idx = unique_characters[lang][c1]
                c2_idx = unique_characters[lang][c2]
                c3_idx = unique_characters[lang][c3]
                frequency_counts[lang][c1_idx][c2_idx][c3_idx] += 1

    cond_prob_3d_arrs = create_3d_arrays(unique_characters, 0)
    for lang, unique_character in unique_characters.items():
        for c1_idx in unique_character.values():
            for c2_idx in unique_character.values():
                sums = np.sum(frequency_counts[lang][c1_idx, c2_idx])
                for c3_idx in unique_character.values():
                    cond_prob_3d_arrs[lang][c1_idx][c2_idx, c3_idx] = \
                        math.log10(frequency_counts[lang][c1_idx, c2_idx, c3_idx] / sums)
    print(cond_prob_3d_arrs["en"])
    # data_frames['en'].loc[idx['x', 'x'], 'x'] = 3
    # print(data_frames['en']['x']['x']['x'])
