import numpy as np
import re
from languages import Languages

LANGUAGES = {
    "eu": 0,
    "ca": 1,
    "gl": 2,
    "es": 3,
    "en": 4,
    "pt": 5
}


ts_per_lang = [[], [], [], [], [], []]
frequencies = []
c_totals = []

def categorize(tweets):
    for t in tweets:
        ts_per_lang[LANGUAGES[t[0]]].append(clean_tweet(t[1]))


def clean_tweet(t):
    return re.sub(r"[^A-Za-z]", '', t).lower()

def count_c_frequency(ts):
    bag = {}
    count = 0
    for t in ts:
        if t in bag.keys():
            bag[t] += 1
        else:
            bag[t]




f = open('training-tweets.txt', "r", encoding="utf8")
input_tweets = f.readlines()

training_tweets = []

for i in input_tweets:
    training_tweet = i.split("\t")
    training_tweets.append([training_tweet[2], training_tweet[3].strip()])

categorize(training_tweets)
