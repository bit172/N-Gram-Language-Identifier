import numpy as np
import pandas as pd
from utils import *
import math

v = n = s_factor = training_file = test_file = None


def create_data_frames(unique_characters, initial_value):
    data_frames = {}
    for lang, characters in unique_characters.items():
        data_frames[lang] = pd.DataFrame(initial_value,
                                         index=pd.MultiIndex.from_product([characters, characters]),
                                         columns=characters,
                                         dtype=np.float64)
    return data_frames


def split_tweet_into_trigrams(tweet):
    for i in range(len(tweet) - 2):
        c1 = tweet[i]
        c2 = tweet[i + 1]
        c3 = tweet[i + 2]
        yield c1, c2, c3


def execute(input_v, input_n, input_s, input_train, input_test):
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)
    idx = pd.IndexSlice

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = unique_c(training_tweets, v)

    data_frames = create_data_frames(unique_characters, s_factor)

    for lang, tweets in training_tweets.items():
        pass
        for tweet in tweets:
            for c in split_tweet_into_trigrams(tweet):
                c1, c2, c3 = c
                if c1 not in unique_characters[lang]:
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang]:
                    c2 = '<NOT-APPEAR>'
                if c3 not in unique_characters[lang]:
                    c3 = '<NOT-APPEAR>'
                data_frames[lang].loc[idx[c1, c2], c3] += 1


    # data_frames['en'].loc[idx['x', 'x'], 'x'] = 3
    # print(data_frames['en']['x']['x']['x'])
    data_frames['en'].to_csv(f'./cond_prob_tables/trigram_en.csv', index=True, encoding='utf-8')
