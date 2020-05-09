import numpy as np
import math
import io
from ngram import NGram


class Trigram(NGram):
    def conditional_prob_matrix(self, training_tweets, unique_characters):
        """
        :param training_tweets: dictionary of training tweets
        :param unique_characters: set of characters in a vocabulary
        :return:conditional probability matrix of each language
        """
        frequency_counts = self.create_matrices(unique_characters, self.S_FACTOR, 3)

        for lang, tweets in training_tweets.items():
            for tweet in tweets:
                for c in self.split_tweet_into_ngrams(tweet, 3):
                    c1_idx, c2_idx, c3_idx = self.char_index_dictionary(unique_characters[lang], c)
                    frequency_counts[lang][c1_idx, c2_idx, c3_idx] += 1

        cond_prob_3d_arrs = self.create_matrices(unique_characters, self.S_FACTOR, 3)

        for lang, unique_character in unique_characters.items():
            for index, val in np.ndenumerate(cond_prob_3d_arrs[lang]):
                cond_prob_3d_arrs[lang][index] = \
                    math.log10(frequency_counts[lang][index] / np.sum(frequency_counts[lang][index[:2]]))
        return cond_prob_3d_arrs

    def evaluate_test_set(self, test_tweets, unique_chars, cond_prob_3d, language_probability):
        """
        :param test_tweets: list of testing tweets
        :param unique_chars: set of characters in a vocabulary
        :param cond_prob_3d: dictionary of conditional probabilities mapped to a language key
        :param language_probability: a dictionary of the probability of a language in the training tweets
        """
        f = io.open(self.OUTPUT_FILE_NAME, "w")
        for test_tweet in test_tweets:
            probabilities = {}
            tweet = test_tweet[2]
            for lang, c_prob in cond_prob_3d.items():
                probabilities[lang] = language_probability[lang]
                for c in self.split_tweet_into_ngrams(tweet, 3):
                    c1_idx, c2_idx, c3_idx = self.char_index_dictionary(unique_chars[lang], c)
                    probabilities[lang] += c_prob[c1_idx, c2_idx, c3_idx]
            f.write(self.generate_output_str(probabilities, test_tweet))
        f.close()
