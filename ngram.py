import abc
import math
import re
import numpy as np
from evaluate_model import evaluate_model
from utils import *
from decimal import Decimal


def categorize(tweets):
    """
    Categorizes tweets based on a language
    :param tweets: raw training tweets
    :return: dictionary of tweets where the key is the language and value is a list of tweets
    """
    ts_per_lang = {"eu": [], "ca": [], "gl": [], "es": [], "en": [], "pt": []}
    for t in tweets:
        ts_per_lang[t[1]].append(t[2])
    return ts_per_lang


def total_c_in_isalpha():
    """
    Gives the total number of unicode characters accepted by isalpha()
    Taken from FAQ
    :return: total number of accepted characters
    """
    count = 0
    # unicode = 17 planes of 2**16 symbols
    for codepoint in range(17 * 2 ** 16):
        ch = chr(codepoint)
        if ch.isalpha():
            count = count + 1
    return count


def unique_characters(training_tweets):
    unique_chars = {}
    # concatenate all strings in a given language and find the unique characters by using join
    for language, tweets in training_tweets.items():
        unique_chars[language] = list(set(''.join(tweets)))
        unique_chars[language].remove(" ")
        unique_chars[language].sort()
    return unique_chars


def lang_probs(training_tweets, total_tweet_num):
    """
    :param training_tweets: dictionary of training tweets
    :param total_tweet_num: the total number of tweets in the training set
    :return: a dictionary of the probability of each language
    """
    probabilities = {}  # stores the probability of all languages for each tweet
    for lang, training_tweet in training_tweets.items():
        probabilities[lang] = math.log10(len(training_tweet) / total_tweet_num)
    return probabilities


class NGram(abc.ABC):

    def __init__(self, v, s_factor, training_factor, test_file, output_file_name):
        self.V = v
        self.S_FACTOR = s_factor
        self.TRAINING_FILE = training_factor
        self.TEST_FILE = test_file
        self.OUTPUT_FILE_NAME = output_file_name

    def execute(self):
        raw_training_tweets = read(self.TRAINING_FILE)
        training_tweets = self.process_tweets(raw_training_tweets)
        training_tweets = categorize(training_tweets)
        unique_chars = self.categorized_unique_characters(training_tweets)

        raw_test_tweets = read(self.TEST_FILE)
        test_tweets = self.process_tweets(raw_test_tweets)

        self.evaluate_test_set(test_tweets, unique_chars, self.cond_prob_matrix(training_tweets, unique_chars),
                               lang_probs(training_tweets, len(raw_training_tweets)))
        evaluate_model(self.OUTPUT_FILE_NAME)

    def generate_output_str(self, probabilities, test_tweet):
        """
          Creates the string for trace files
          :param probabilities: dictionary of conditional probability matrix for all languages
          :param test_tweet: tweet from testing set
          :return: trace file line
          """
        most_prob_lang = max(iter(probabilities.keys()), key=(lambda key: probabilities[key]))
        correctness = "correct" if most_prob_lang == test_tweet[1] else "wrong"
        return f"{test_tweet[0]}  {most_prob_lang}  {'%.2E' % Decimal(probabilities[most_prob_lang])}  {test_tweet[1]}  {correctness}\r"

    def process_tweets(self, raw_tweets):
        """
        Removes tabs from raw_tweets and cleans a tweet base on vocabulary
        :param raw_tweets: raw training tweets
        :return: list of of tuples: (id, language, cleaned tweet)
        """
        tweets = []
        for i in raw_tweets:
            tweet = i.split("\t")  # separates the string by tab and put into a array
            tweets.append([tweet[0], tweet[2], self.clean_tweet(tweet[3].strip())])  # (id, language, tweet)
        return tweets

    def clean_tweet(self, tweet):
        """
        Cleans a tweet based on the vocabulary requirements
        :param tweet: tweet
        :param v: vocabulary
        :return: cleaned tweet
        """
        if self.V == 0:
            return re.sub(r"[^A-Za-z]", ' ', tweet).lower()
        if self.V == 1:
            return re.sub(r"[^A-Za-z]", ' ', tweet)
        if self.V == 2:
            return "".join([x if x.isalpha() else ' ' for x in tweet])

    def categorized_unique_characters(self, training_tweets):
        """
        Finds all indexes of unique characters in a training set based on vocabulary.
        :param training_tweets: dictionary of training tweets
        :param v: vocabulary
        :return: dictionary of dictionary: language:character:index
        """
        unique_chars = unique_characters(training_tweets)
        for lang, characters in unique_chars.items():
            if len(characters) < self.total_c_in_v():
                characters.append('<NOT-APPEAR>')
        unique = {}
        for lan, cha in unique_chars.items():
            counter = 0
            unique[lan] = {}
            for c in cha:
                unique[lan][c] = counter
                counter += 1
        return unique

    def total_c_in_v(self):
        """
        Returns the total number of characters for a given vocabulary
        :param v: vocabulary
        :return: number of characters
        """
        if self.V == 0:
            return 26
        if self.V == 1:
            return 52
        if self.V == 2:
            return total_c_in_isalpha()

    def find_c_indices(self, unique_chars_in_lang, n_grams):
        indices = list(n_grams)
        for idx, c in enumerate(indices):
            if c not in unique_chars_in_lang.keys():
                indices[idx] = '<NOT-APPEAR>'
            indices[idx] = unique_chars_in_lang[indices[idx]]
        return indices

    def create_matrices(self, unique_chars, initial_val, dimension):
        """
        Creates an n x n matrix where n is the vocabulary size
        :param dimension: the dimension of the matrix
        :param unique_chars: set of characters in a vocabulary
        :param initial_val: initial value of all cells
        :return: n x n matrix
        """
        matrices = {}
        for lang, chars in unique_chars.items():
            size = len(chars)
            shape = tuple(size for _ in range(dimension))
            matrices[lang] = np.full(shape, initial_val, dtype=np.float64)
        return matrices

    def split_tweet_into_ngrams(self, tweet, n):
        for i in range(len(tweet) - (n - 1)):
            c = [tweet[j] for j in range(i, i + n)]
            if " " in c:
                continue
            yield c

    @abc.abstractmethod
    def evaluate_test_set(self, test_tweets, unique_chars, matrix, language_probability):
        return

    @abc.abstractmethod
    def cond_prob_matrix(self, training_tweets, unique_chars):
        return
