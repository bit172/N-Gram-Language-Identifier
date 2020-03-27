import re
from decimal import Decimal


def clean_tweet(t, v):
    """
    clean a single tweet
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


def total_c_in_v(v):
    if v == 0:
        return 26
    if v == 1:
        return 52
    if v == 2:
        return 116766


def categorize(raw_tweets, v):
    tweets = process_tweets(raw_tweets, v)
    ts_per_lang = {"eu": [], "ca": [], "gl": [], "es": [], "en": [], "pt": []}
    for t in tweets:
        ts_per_lang[t[1]].append(t[2])
    return ts_per_lang


def read(file):
    f = open(file, "r", encoding="utf8")
    contents = f.readlines()
    f.close()
    return contents


def process_tweets(raw_tweets, v):
    tweets = []
    for i in raw_tweets:
        tweet = i.split("\t")  # separates the string by tab and put into a array
        tweets.append([tweet[0], tweet[2], clean_tweet(tweet[3].strip(), v)])  # (id, language, tweet)
    return tweets


def generate_output_str(probabilities, test_tweet):
    most_prob_lang = max(iter(probabilities.keys()), key=(lambda key: probabilities[key]))
    correctness = "correct" if most_prob_lang == test_tweet[1] else "wrong"
    return f"{test_tweet[0]}  {most_prob_lang}  {'%.2E' % Decimal(probabilities[most_prob_lang])}  {test_tweet[1]}  {correctness}\n"


def output_file_name(v, n, s_factor):
    return f"./results/trace_{v}_{n}_{s_factor}.txt"


def compute_accuracy(v, n, s_factor):
    output_file = output_file_name(v, n, s_factor)
    outputs = read(output_file)
    accuracy = ([i.split()[4] for i in outputs].count("correct") / len(outputs)) * 100
    return f"| v:{v} | n:{n} | s_factor:{s_factor} | accuracy: {accuracy}%"


def unique_c_arr(training_tweets, v):
    unique_characters = {}
    # concatenate all strings in a given language and find the unique characters by using join
    for language, tweets in training_tweets.items():
        unique_characters[language] = set(''.join(tweets))
        # unique_characters[language].sort()
    for lang, characters in unique_characters.items():
        if len(characters) < total_c_in_v(v):
            characters.add('<NOT-APPEAR>')
    return unique_characters
