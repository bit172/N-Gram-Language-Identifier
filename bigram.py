import io
from ngram import NGram
import numpy as np
import math


class Bigram(NGram):

    def conditional_prob_matrix(self, training_tweets, unique_characters):
        """
        Generates the conditional probability matrix for the bigram model
        :param training_tweets: list of cleaned training tweets
        :param unique_characters: dictionary where the key is a language
        and the value the number of unique characters
        :return: 2-dimensional conditional probability matrix
        """
        frequency_counts = self.create_matrices(unique_characters, self.S_FACTOR, 2)

        for lang, tweets in training_tweets.items():
            for tweet in tweets:
                for c in self.split_tweet_into_ngrams(tweet, 2):
                    c1_idx, c2_idx = self.char_index_dictionary(unique_characters[lang], c)
                    frequency_counts[lang][c1_idx, c2_idx] += 1

        cond_prob_2d_arrs = self.create_matrices(unique_characters, self.S_FACTOR, 2)

        for lang in unique_characters.keys():
            for index, val in np.ndenumerate(cond_prob_2d_arrs[lang]):
                cond_prob_2d_arrs[lang][index] = math.log10(
                    frequency_counts[lang][index] / np.sum(frequency_counts[lang][index[0]]))
        return cond_prob_2d_arrs

    def evaluate_test_set(self, test_tweets, unique_characters, cond_prob_2d, language_probabilities):
        """
        Evaluates the test set based on with a bigram model and writes result to a file
        :param test_tweets: cleaned test tweets
        :param unique_characters: dictionary of dictionary: language:character:index
        :param cond_prob_2d: bigram probability matrix
        :param language_probabilities: a dictionary of the probability of a language in the training tweets
        """
        f = io.open(self.OUTPUT_FILE_NAME, "w")
        for test_tweet in test_tweets:
            probabilities = {}  # stores the probability of all languages for each tweet
            tweet = test_tweet[2]
            for lang, c_prob in cond_prob_2d.items():
                probabilities[lang] = language_probabilities[lang]
                for c in self.split_tweet_into_ngrams(tweet, 2):
                    c1_idx, c2_idx = self.char_index_dictionary(unique_characters[lang], c)
                    probabilities[lang] += c_prob[c1_idx, c2_idx]
            f.write(self.generate_output_str(probabilities, test_tweet))
        f.close()
