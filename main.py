import numpy as np
import re
import math
import pprint
s_factor = 0.3


def categorize(tweets):
    ts_per_lang = {"eu": [], "ca": [], "gl": [], "es": [], "en": [], "pt": []}
    for t in tweets:
        ts_per_lang[t[0]].append(clean_tweet(t[1]))
    return ts_per_lang


def clean_tweet(t):
    return re.sub(r"[^A-Za-z]", '', t).lower()


def count_c_frequency(ts):
    bag = {}
    for t in ts:
        for c in t:
            if c in bag.keys():
                bag[c] += 1
            else:
                bag[c] = 1 + s_factor
    return bag


def total_c(ts):
    count = 0
    for t in ts:
        for c in t:
            count += 1
    count += 26 * s_factor
    return count


if __name__ == '__main__':
    f = open('training-tweets.txt', "r", encoding="utf8")
    input_tweets = f.readlines()

    training_tweets = []

    for i in input_tweets:
        training_tweet = i.split("\t")
        training_tweets.append([training_tweet[2], training_tweet[3].strip()])

    frequencies = {}
    c_totals = {}
    for k, v in categorize(training_tweets).items():
        if k in frequencies.keys():
            frequencies[k].append(count_c_frequency(v))
        else:
            frequencies[k] = count_c_frequency(v)
        c_totals[k] = total_c(v)

    cond_prob = {}
    for lang, frequency in frequencies.items():
        bag = {}
        total = c_totals[lang]
        for c, count in frequency.items():
            bag[c] = math.log10(count / total)
        if len(bag) < 26:
            bag['<NOT-APPEAR>'] = math.log10(s_factor / total)
        cond_prob[lang] = bag
    pprint.pprint(cond_prob,  width=1)