from utils import *
import numpy as np
# print(clean_tweet("Let's #Celebrate #Adidas Originals der #Kaiser 1974 #Track top http://t.co/gMcaY3VcbX http://t.co/Vhut3i6Vr4", 3))




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


def create_2d_arrays(unique_characters, initial_val):
    two_d_arrs = {}
    for lang, chars in unique_characters.items():
        size = len(chars)
        two_d_arrs[lang] = np.full((size, size), initial_val, dtype=np.float64)
    return two_d_arrs



def execute(input_v, input_n, input_s, input_train, input_test):
    global v, n, s_factor, training_file, test_file
    (v, n, s_factor, training_file, test_file) = (input_v, input_n, input_s, input_train, input_test)

    raw_training_tweets = read(training_file)
    training_tweets = categorize(raw_training_tweets, v)

    unique_characters = unique_c_v2(training_tweets, v)

    frequency_counts_3d = create_3d_arrays(unique_characters, 0)
    frequency_counts_2d = create_2d_arrays(unique_characters, 0)
    idx = unique_characters['en']['a']

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
                frequency_counts_2d[lang][c1_idx][c2_idx] += 1

            for i in range(len(tweet) - 2):
                c1 = tweet[i]
                c2 = tweet[i + 1]
                c3 = tweet[i + 2]
                if c1 not in unique_characters[lang].keys():
                    c1 = '<NOT-APPEAR>'
                if c2 not in unique_characters[lang].keys():
                    c2 = '<NOT-APPEAR>'
                if c3 not in unique_characters[lang].keys():
                    c3 = '<NOT-APPEAR>'
                c1_idx = unique_characters[lang][c1]
                c2_idx = unique_characters[lang][c2]
                c3_idx = unique_characters[lang][c3]
                frequency_counts_3d[lang][c1_idx][c2_idx][c3_idx] += 1

        sums = np.sum(frequency_counts_3d[lang][unique_characters[lang]['a'], unique_characters[lang]['a']])
        sums_2d = frequency_counts_2d[lang][unique_characters[lang]['a']][unique_characters[lang]['a']]
        if sums > sums_2d:
            print(tweet)

    print(np.sum(frequency_counts_3d['en'][idx, idx]))
    print(frequency_counts_2d['en'][idx, idx])