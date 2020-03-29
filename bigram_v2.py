import numpy as np
import pandas as pd
from utils import *
import math
import io

v = n = s_factor = training_file = test_file = None


def evaluate_test_set(test_tweets, unique_characters, cond_prob_2d, total_tweet_num, training_tweets):
    """
    Evaluates the test set based on with a bigram model
    :param test_tweets: cleaned test tweets
    :param unique_characters: dictionary of dictionary: language:character:index
    :param cond_prob_2d: bigram probability matrix
    :param total_tweet_num: number of tweets in the training set
    :param training_tweets: dictionary where the key is the language and the value is a set of unique characters
    :return: None
    """
    f = io.open(output_file_name(v, n, s_factor), "w")
    for test_tweet in test_tweets:
        probabilities = {}  # stores the probability of all languages for each tweet
        tweet = test_tweet[2]
        for lang, c_prob in cond_prob_2d.items():
            probabilities[lang] = math.log10(len(training_tweets[lang]) / total_tweet_num)
            for i in range(len(tweet) - 1):
                c1 = tweet[i]
                c2 = tweet[i + 1]
                if c1 not in unique_characters[lang].keys():
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang].keys():
                    c2 = '<NOT-APPEAR>'
                c1_idx = unique_characters[lang][c1]
                c2_idx = unique_characters[lang][c2]
                probabilities[lang] += c_prob[c1_idx, c2_idx]
        f.write(generate_output_str(probabilities, test_tweet))
    f.close()


def create_2d_arrays(unique_characters, initial_val):
    """
    Creates an n x n matrix where n is the vocabulary size
    :param unique_characters: set of characters in a vocabulary
    :param initial_val: initial value of all cells
    :return: n x n matrix
    """
    two_d_arrs = {}
    for lang, chars in unique_characters.items():
        size = len(chars)
        two_d_arrs[lang] = np.full((size, size), initial_val, dtype=np.float64)
    return two_d_arrs


def create_data_frames(unique_characters, initial_value):
    """
    Creates a n x n pandas DataFrame where n is the vocabulary size
    :param unique_characters: set of unique characters in a vocabulary
    :param initial_value: initial value of all cells
    :return: n x n pandas DataFrame
    """
    data_frames = {}
    for lang, characters in unique_characters.items():
        data_frames[lang] = pd.DataFrame(data=initial_value[lang], columns=characters.keys(), index=characters.keys(),
                                         dtype=np.float64)
    return data_frames


def execute(input_v, input_n, input_s, input_train, input_test):
    """
    Creates the model with a training set and evaluates it with a test set
    :param input_v: vocabulary
    :param input_n: n-gram to use
    :param input_s: smoothing factor
    :param input_train: training set file name
    :param input_test: test set file name
    :return: None
    """
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = unique_c_v2(training_tweets, v)

    frequency_counts = create_2d_arrays(unique_characters, s_factor)

    for lang, tweets in training_tweets.items():
        for tweet in tweets:
            for i in range(len(tweet) - 1):
                c1 = tweet[i]
                c2 = tweet[i + 1]
                if c1 not in unique_characters[lang].keys():
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang].keys():
                    c2 = '<NOT-APPEAR>'
                c1_idx = unique_characters[lang][c1]
                c2_idx = unique_characters[lang][c2]
                frequency_counts[lang][c1_idx, c2_idx] += 1

    # frequency_frames = create_data_frames(unique_characters, frequency_counts)
    #
    # for lang, frame in data_frames.items():
    #    frame.to_csv(f'./cond_prob_tables/freq_bigram_{lang}.csv', index=True, encoding='utf-8')

    cond_prob_2d_arrs = create_2d_arrays(unique_characters, 0)

    for lang, unique_character in unique_characters.items():
        for c1_idx in unique_character.values():
            for c2_idx in unique_character.values():
                cond_prob_2d_arrs[lang][c1_idx, c2_idx] = math.log10(
                    frequency_counts[lang][c1_idx, c2_idx] / np.sum(frequency_counts[lang][c1_idx]))

    raw_test_tweets = read(test_file)
    test_tweets = process_tweets(raw_test_tweets, v)
    evaluate_test_set(test_tweets, unique_characters, cond_prob_2d_arrs, len(raw_training_tweets), training_tweets)
    print(compute_accuracy(v, n, s_factor))
