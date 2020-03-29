import re
from decimal import Decimal
import io

"""
This set contains all possible unique characters in all the 6 languages
as describe in the following wikipedia article:
https://en.wikipedia.org/wiki/Wikipedia:Language_recognition_chart
"""
character_set_byom = {'ü', 'q', 'z', 'h', 'j', 'u', 'c', 'a', 'e', 'ç', 'f', 'n', 'è', 'ó', 's', 'i', 'd', 'é', 'ï',
                      'ã', 't', 'à', 'ú', 'p', 'ê', 'l', 'b', 'ò', 'w', 'á', 'k', 'â', 'v', 'ñ', 'r', 'y', 'g', 'õ',
                      'm', 'í', 'o', 'x', 'ô'}


def clean_tweet(t, v):
    """
    Cleans a tweet based on the vocabulary requirements

    :param t: tweet
    :param v: vocabulary
    :return: cleaned tweet
    """
    if v == 0:
        return re.sub(r"[^A-Za-z]", '', t).lower()
    if v == 1:
        return re.sub(r"[^A-Za-z]", '', t)
    if v == 2:
        return "".join([x for x in t if x.isalpha()])
    if v == 3:
        return clean_tweet_byom(t)


def clean_tweet_byom(t):
    """
    Cleans a tweet by doing the following:
        1. Converting all characters to lowercase
        2. Removing http/https links from the tweet
        3. Removing @usernames and #hashtags
        4. Removing characters not part of our byom characters
        5. Converting all whitespaces to a single whitespace
        6. Replacing characters that repeat more than 2 times to a single character
        7. Adding delimiters to the start/end of the tweet
    :param t: tweet
    :return: cleaned byom tweet
    """
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = " ".join(filter(lambda x: x[0] != '@' and x[0] != '#', t.split()))
    t = "".join([x for x in t if x in character_set_byom or x == ' '])
    t = re.sub(r"\s\s+", ' ', t)
    t = re.sub(r'(.)\1{2,}', r'\1', t)
    if t[-1] is " ":
        t = t[:-1]  # Remove the space at the end if there is one
    t = '*' + t + '*'
    return t


def total_c_in_isalpha():
    """
    Gives the total number of unicode characters accepted by isalpha()
    Taken from FAQ
    :return:
    """
    count = 0
    # unicode = 17 planes of 2**16 symbols
    for codepoint in range(17 * 2 ** 16):
        ch = chr(codepoint)
        if ch.isalpha():
            count = count + 1
    return count


def total_c_in_v(v):
    """
    Returns the total number of characters for a given vocabulary
    :param v: vocabulary
    :return: number of characters
    """
    if v == 0:
        return 26
    if v == 1:
        return 52
    if v == 2:
        return total_c_in_isalpha()
    if v == 3:
        return len(character_set_byom) + 2  # accounts for whitespace and * delimiter


def process_tweets(raw_tweets, v):
    """
    Removes tabs from raw_tweets and cleans a tweet base on vocabulary
    :param raw_tweets: raw training tweets
    :param v: vocabulary
    :return: list of of tuples: (id, language, cleaned tweet)
    """
    tweets = []
    for i in raw_tweets:
        tweet = i.split("\t")  # separates the string by tab and put into a array
        tweets.append([tweet[0], tweet[2], clean_tweet(tweet[3].strip(), v)])  # (id, language, tweet)
    return tweets


def categorize(raw_tweets, v):
    """
    Categorizes tweets based on a language
    :param raw_tweets: raw training tweets
    :param v: vocabulary
    :return: dictionary of tweets where the key is the language and value is a list of tweets
    """
    tweets = process_tweets(raw_tweets, v)
    ts_per_lang = {"eu": [], "ca": [], "gl": [], "es": [], "en": [], "pt": []}
    for t in tweets:
        ts_per_lang[t[1]].append(t[2])
    return ts_per_lang


def read(file):
    """
    Reads all the lines of a file
    :param file: file name
    :return: list of all lines
    """
    f = io.open(file, "r", encoding="utf8")
    contents = f.readlines()
    f.close()
    return contents


def generate_output_str(probabilities, test_tweet):
    """
    Creates the string for trace files
    :param probabilities: dictionary of conditional probability matrix for all languages
    :param test_tweet: tweet from testing set
    :return: trace file line
    """
    most_prob_lang = max(iter(probabilities.keys()), key=(lambda key: probabilities[key]))
    correctness = "correct" if most_prob_lang == test_tweet[1] else "wrong"
    return f"{test_tweet[0]}  {most_prob_lang}  {'%.2E' % Decimal(probabilities[most_prob_lang])}  {test_tweet[1]}  {correctness}\n"


def output_file_name(v, n, s_factor):
    """
    Creates the file name for tracing files
    :param v: vocabulary
    :param n: n-gram model being used
    :param s_factor: smoothing factor given
    :return: trace file name
    """
    return f"./results/trace_{v}_{n}_{s_factor}.txt"


def compute_accuracy(v, n, s_factor):
    """
    Computes the accuracy of a model based on a trace file
    :param v: vocabulary used
    :param n: n-gram model used
    :param s_factor: smoothing used
    :return:
    """
    output_file = output_file_name(v, n, s_factor)
    outputs = read(output_file)
    accuracy = ([i.split()[4] for i in outputs].count("correct") / len(outputs)) * 100
    return f"| v:{v} | n:{n} | s_factor:{s_factor} | accuracy: {accuracy}%"


def unique_c(training_tweets, v):
    """
    Finds all unique characters in a training set based on vocabulary.
    :param training_tweets: dictionary of training tweets
    :param v:
    :return:
    """
    unique_characters = {}
    # concatenate all strings in a given language and find the unique characters by using join
    for language, tweets in training_tweets.items():
        unique_characters[language] = set(''.join(tweets))
        # unique_characters[language].sort()
    for lang, characters in unique_characters.items():
        if len(characters) < total_c_in_v(v):
            characters.add('<NOT-APPEAR>')
    return unique_characters


def unique_c_v2(training_tweets, v):
    """

    :param training_tweets:
    :param v:
    :return:
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
