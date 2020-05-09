from trigram import Trigram
import re

"""
This set contains all possible unique characters in all the 6 languages
as describe in the following wikipedia article:
https://en.wikipedia.org/wiki/Wikipedia:Language_recognition_chart
"""
CHARACTER_SET = {'ü', 'q', 'z', 'h', 'j', 'u', 'c', 'a', 'e', 'ç', 'f', 'n', 'è', 'ó', 's', 'i', 'd', 'é', 'ï',
                 'ã', 't', 'à', 'ú', 'p', 'ê', 'l', 'b', 'ò', 'w', 'á', 'k', 'â', 'v', 'ñ', 'r', 'y', 'g', 'õ',
                 'm', 'í', 'o', 'x', 'ô'}


class BYOM(Trigram):

    def vocabulary_size(self):
        """
        Total unique characters in new character set
        :return: number of unique characters including whitespace and * delimiter
        """
        return len(CHARACTER_SET) + 2

    def clean_tweet(self, tweet):
        """
        Cleans a tweet by doing the following:
            1. Converting all characters to lowercase
            2. Removing http/https links from the tweet
            3. Removing @usernames and #hashtags
            4. Removing characters not part of our byom characters
            5. Converting all whitespaces to a single whitespace
            6. Replacing characters that repeat more than 3 times to a 2 characters
            7. Adding delimiters to the start/end of the tweet
        :param tweet: tweet
        :return: cleaned byom tweet
        """
        tweet = tweet.lower()
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = " ".join(filter(lambda x: x[0] != '@' and x[0] != '#', tweet.split()))
        tweet = "".join([x for x in tweet if x in CHARACTER_SET or x == ' '])
        tweet = re.sub(r"\s\s+", ' ', tweet)
        tweet = re.sub(r'(.)\1{3,}', r'\1\1', tweet)
        if tweet[-1] is " ":
            tweet = tweet[:-1]  # Remove the space at the end if there is one
        tweet = '*' + tweet + '*'
        return tweet

    def split_tweet_into_ngrams(self, tweet, n):
        """
        Generator that splits tweets into ngrams of size n without omitting
        whitespace characters
        :param tweet: tweet
        :param n: size of ngram
        :return: list of ngrams
        """
        for i in range(len(tweet) - (n - 1)):
            c = [tweet[j] for j in range(i, i + n)]
            yield c
