import numpy as np
import pandas as pd
import pprint
from utils import *
import math

v = n = s_factor = training_file = test_file = None


def output_most_prob_lang_and_required_els(test_tweets, cond_prob_frams):
    return


def execute(input_v, input_n, input_s, input_train, input_test):
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = {}
    # concatenate all strings in a given language and find the unique characters by using join
    for language, tweets in training_tweets.items():
        unique_characters[language] = list(set(''.join(tweets)))
        unique_characters[language].sort()

    data_frames = {}
    for lang, characters in unique_characters.items():
        if len(characters) < total_c_in_v(v):
            characters.append('<NOT-APPEAR>')
        data_frames[lang] = pd.DataFrame(s_factor, columns=characters, index=characters, dtype=np.float64)

    for lang, tweets in training_tweets.items():
        for tweet in tweets:
            for i in range(len(tweet) - 1):
                c1 = tweet[i]
                c2 = tweet[i + 1]
                if c1 not in unique_characters[lang]:
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang]:
                    c2 = '<NOT-APPEAR>'
                data_frames[lang][c1][c2] += 1

    cond_prob_frames = {}
    for lang, characters in unique_characters.items():
        cond_prob_frames[lang] = pd.DataFrame(s_factor, columns=characters, index=characters, dtype=np.float64)

    for lang, unique_character in unique_characters.items():
        for c1 in unique_character:
            for c2 in unique_character:
                cond_prob_frames[lang][c1][c2] = math.log10(data_frames[lang][c1][c2] / data_frames[lang][c1].sum())

    for lang, frame in cond_prob_frames.items():
        frame.to_csv(f'./cond_prob_tables/bigram_{lang}.csv', index=False, encoding='utf-8')

    raw_test_tweets = read(test_file)
    test_tweets = process_tweets(raw_test_tweets, v)

    # output_most_prob_lang_and_required_els(test_tweets, cond_prob_frames)
    # print(compute_accuracy(v, n, s_factor))
