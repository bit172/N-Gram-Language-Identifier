import numpy as np
import pandas as pd
import pprint
import io
from utils import *
import math

v = n = s_factor = training_file = test_file = None


def output_most_prob_lang_and_required_els(test_tweets, unique_characters, cond_prob_frames):
    f = io.open(output_file_name(v, n, s_factor), "w", "utf-8")
    for test_tweet in test_tweets:
        probabilities = {}  # stores the probability of all languages for each tweet
        tweet = test_tweet[2]
        for lang, c_prob in cond_prob_frames.items():
            probabilities[lang] = 0
            for i in range(len(tweet) - 1):
                c1 = tweet[i]
                c2 = tweet[i + 1]
                if c1 not in unique_characters[lang]:
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang]:
                    c2 = '<NOT-APPEAR>'
                probabilities[lang] += c_prob[c1][c2]
        f.write(generate_output_str(probabilities, test_tweet))
    f.close()


def create_data_frames(unique_characters, initial_value):
    data_frames = {}
    for lang, characters in unique_characters.items():
        data_frames[lang] = pd.DataFrame(initial_value, columns=characters, index=characters, dtype=np.float64)
    return data_frames


def execute(input_v, input_n, input_s, input_train, input_test):
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = unique_c_arr(training_tweets, v)

    data_frames = create_data_frames(unique_characters, s_factor)

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

    for lang, frame in data_frames.items():
        frame.to_csv(f'./cond_prob_tables/freq_bigram_{lang}.csv', index=True, encoding='utf-8')

    cond_prob_frames = create_data_frames(unique_characters, 0)

    for lang, unique_character in unique_characters.items():
        for c1 in unique_character:
            for c2 in unique_character:
                cond_prob_frames[lang][c1][c2] = math.log10(data_frames[lang][c1][c2] / data_frames[lang][c1].sum())

    for lang, frame in cond_prob_frames.items():
        frame.to_csv(f'./cond_prob_tables/bigram_{lang}.csv', index=True, encoding='utf-8')

    raw_test_tweets = read(test_file)
    test_tweets = process_tweets(raw_test_tweets, v)
    output_most_prob_lang_and_required_els(test_tweets, unique_characters, cond_prob_frames)
    print(compute_accuracy(v, n, s_factor))
