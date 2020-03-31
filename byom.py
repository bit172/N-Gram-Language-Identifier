import numpy as np
from utils import *
import math
import io

v = n = s_factor = training_file = test_file = None


def evaluate_test_set(test_tweets, unique_characters, cond_prob_3d, total_tweet_num, training_tweets):
    """
    Evaluates the test set based on with a trigram model
    :param test_tweets: cleaned test tweets
    :param unique_characters: dictionary of dictionary: language:character:index
    :param cond_prob_3d: trigram probability matrix
    :param total_tweet_num: number of tweets in the training set
    :param training_tweets: dictionary where the key is the language and the value is a set of unique characters
    :return: None
    """
    f = io.open(output_file_name(v, n, s_factor), "w")
    for test_tweet in test_tweets:
        probabilities = {}  # stores the probability of all languages for each tweet
        tweet = test_tweet[2]
        for lang, c_prob in cond_prob_3d.items():
            probabilities[lang] = math.log10(len(training_tweets[lang]) / total_tweet_num)
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
                probabilities[lang] += c_prob[c1_idx][c2_idx][c3_idx]
        f.write(generate_output_str(probabilities, test_tweet))
    f.close()


def create_3d_arrays(unique_characters, initial_val):
    """
    Creates an n x n x n matrix where n is the vocabulary size
    :param unique_characters: set of characters in a vocabulary
    :param initial_val: initial value of all cells
    :return: n x n x n matrix
    """
    three_d_arrs = {}
    for lang, chars in unique_characters.items():
        size = len(chars)
        three_d_arrs[lang] = np.full((size, size, size), initial_val, dtype=np.float64)
    return three_d_arrs


def unique_c_byom(training_tweets):
    """
    Finds all indexes of unique characters in a training set based on vocabulary.
    :param training_tweets: dictionary of training tweets
    :param v: vocabulary
    :return: dictionary of dictionary: language:character:index
    """
    unique_characters = {}
    # concatenate all strings in a given language and find the unique characters by using join
    for language, tweets in training_tweets.items():
        unique_characters[language] = list(set(''.join(tweets)))
        unique_characters[language].sort()
    for lang, characters in unique_characters.items():
        if len(characters) < total_c_in_v(v):
            characters.append('<NOT-APPEAR>')
    unique = {}
    for lan, cha in unique_characters.items():
        counter = 0
        unique[lan] = {}
        for c in cha:
            unique[lan][c] = counter
            counter += 1

    return unique


def split_tweet_into_trigrams(tweet):
    """
    Generator that splits a tweet into trigrams
    :param tweet: tweet
    :return: tuple of trigrams
    """
    for i in range(len(tweet) - 2):
        c1 = tweet[i]
        c2 = tweet[i + 1]
        c3 = tweet[i + 2]
        yield c1, c2, c3


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
    training_tweets = process_tweets(raw_training_tweets, v)
    training_tweets = categorize(training_tweets, v)
    unique_characters = unique_c_byom(training_tweets)

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
                frequency_counts[lang][c1_idx, c2_idx, c3_idx] += 1

    cond_prob_3d_arrs = create_3d_arrays(unique_characters, 0)
    for lang, unique_character in unique_characters.items():
        for c1_idx in unique_character.values():
            for c2_idx in unique_character.values():
                sums = np.sum(frequency_counts[lang][c1_idx, c2_idx])
                for c3_idx in unique_character.values():
                    cond_prob_3d_arrs[lang][c1_idx, c2_idx, c3_idx] = \
                        math.log10(frequency_counts[lang][c1_idx, c2_idx, c3_idx] / sums)

    raw_test_tweets = read(test_file)
    test_tweets = process_tweets(raw_test_tweets, v)
    evaluate_test_set(test_tweets, unique_characters, cond_prob_3d_arrs, len(raw_training_tweets), training_tweets)
    print(compute_accuracy(v, n, s_factor))
