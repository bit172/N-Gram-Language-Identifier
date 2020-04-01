from ngram import NGram
import re
import math


class Unigram(NGram):

    def clean_tweet(self, tweet):
        """
           Cleans a tweet based on the vocabulary requirements for unigram model
           :param tweet: tweet
           :return: cleaned tweet
           """
        if self.V == 0:
            return re.sub(r"[^A-Za-z]", '', tweet).lower()
        if self.V == 1:
            return re.sub(r"[^A-Za-z]", '', tweet)
        if self.V == 2:
            return "".join([x for x in tweet if x.isalpha()])

    def total_c(self, categorized_tweets):
        """
        Count the total number of characters found in the tweets per language
        :param categorized_tweets: dictionary of tweets by language
        :return: dictionary of total number of characters based on language
        """
        c_totals = {}
        for language, tweets in categorized_tweets.items():
            count = 0
            for tweet in tweets:
                for character in tweet:
                    count += 1
            count += self.total_c_in_v() * self.S_FACTOR
            c_totals[language] = count
        return c_totals

    def count_c_frequencies(self, tweets):
        """
        Counts the frequency of each characters in an array of tweet strings
        :param tweets: list of tweets
        :return: dictionary of { character: # times character appears in all tweets }
        """
        bag = {}
        for tweet in tweets:
            for c in tweet:
                if c in bag.keys():
                    bag[c] += 1
                else:
                    bag[c] = 1 + self.S_FACTOR
        return bag

    def c_frequencies_in_langs(self, categorized_tweets):
        """
        Count the frequency for each character per language
        :param categorized_tweets: dictionary of tweets by language
        :return: dictionary of character frequencies with language key
        """
        frequencies = {}
        for language, tweets in categorized_tweets.items():
            if language in frequencies.keys():
                frequencies[language].append(self.count_c_frequencies(tweets))
            else:
                frequencies[language] = self.count_c_frequencies(tweets)
        return frequencies

    def categorized_unique_characters(self, training_tweets):
        return None

    def compute_cond_probs(self, frequencies, total_c_counts):
        """
        Find conditional probabilities for each c per lang
        :param frequencies: dictionary of character frequencies with language key
        :param total_c_counts: dictionary of total number of characters based on language
        :return: dictionary of conditional probabilities with language key
        """
        cond_probs = {}
        for lang, frequency in frequencies.items():
            bag = {}
            total = total_c_counts[lang]
            for c, count in frequency.items():
                bag[c] = math.log10(count / total)
            if len(bag) < self.total_c_in_v():  # if we don't have all characters in the bag
                bag['<NOT-APPEAR>'] = math.log10(self.S_FACTOR / total)
            cond_probs[lang] = bag

        return cond_probs

    def cond_prob_matrix(self, training_tweets, unique_chars):
        """
        :param training_tweets: dictionary of training tweets
        :param unique_chars: None
        :return: array containing the conditional probabilities of each character
        """
        total_c_counts = self.total_c(training_tweets)
        frequencies = self.c_frequencies_in_langs(training_tweets)
        return self.compute_cond_probs(frequencies, total_c_counts)

    def evaluate_test_set(self, test_tweets, unique_chars, cond_prob_matrix, language_probability):
        """
        Find most probable language for each tweet and
        store most prob lang and required elements in an output file
        :param language_probability: a dictionary of the probability of a language in the training tweets
        :param cond_prob_matrix: dictionary of conditional probabilities mapped to a language key
        :param unique_chars: None
        :param test_tweets: list of testing tweets
        """
        f = open(self.OUTPUT_FILE_NAME, "w")
        for test_tweet in test_tweets:
            probabilities = {}  # stores the probability of all languages for each tweet
            tweet = test_tweet[2]
            for language, c_probs in cond_prob_matrix.items():
                probabilities[language] = language_probability[language]
                # compute the probability of each language by adding the probabilities of
                # each characters that appear in the tweet
                for c in tweet:
                    if c in c_probs.keys():
                        probabilities[language] += c_probs[c]
                    else:
                        probabilities[language] += c_probs['<NOT-APPEAR>']
            f.write(self.generate_output_str(probabilities, test_tweet))
        f.close()
