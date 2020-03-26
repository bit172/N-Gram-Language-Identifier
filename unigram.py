from utils import *
import math
import pprint
from decimal import Decimal

inputs = read('input.txt')[0].split(" ")
(v, n, s_factor, training_file, test_file) = (int(inputs[0]), int(inputs[1]), float(inputs[2]), inputs[3], inputs[4])


# counts the frequency of each characters in an array of tweet strings
# returns a dictionary of { character: # times character appears in all tweets }
def count_c_frequencies(ts):
    bag = {}
    for t in ts:
        for c in t:
            if c in bag.keys():
                bag[c] += 1
            else:
                bag[c] = 1 + s_factor
    return bag


# count the total number of characters found in the tweets per language
def total_c(categorized_tweets):
    c_totals = {}
    for language, tweets in categorized_tweets.items():
        count = 0
        for tweet in tweets:
            for character in tweet:
                count += 1
        count += 26 * s_factor
        c_totals[language] = count
    return c_totals


# count the frequency for each character per language
def c_frequencies_in_langs(categorized_tweets):
    frequencies = {}
    for language, tweets in categorized_tweets.items():
        if language in frequencies.keys():
            frequencies[language].append(count_c_frequencies(tweets))
        else:
            frequencies[language] = count_c_frequencies(tweets)
    return frequencies


# find conditional probabilities for each c per lang
def compute_cond_probs(frequencies, total_c_counts):
    cond_probs = {}
    for lang, frequency in frequencies.items():
        bag = {}
        total = total_c_counts[lang]
        for c, count in frequency.items():
            bag[c] = math.log10(count / total)
        if len(bag) < 26:  # if we don't have all 26 letters
            bag['<NOT-APPEAR>'] = math.log10(s_factor / total)
        cond_probs[lang] = bag
    return cond_probs


# find most probable language for each tweet
# store most prob lang and required elements in an output file
def output_most_prob_lang_and_required_els(test_tweets, cond_probs):
    f = open(output_file_name(v, n, s_factor), "w")
    for test_tweet in test_tweets:
        probabilities = {}  # stores the probability of all languages for each tweet
        tweet = test_tweet[2]
        for language, c_probs in cond_probs.items():
            probabilities[language] = 0
            # compute the probability of each language by adding the probabilities of
            # each characters that appear in the tweet
            for c in tweet:
                probabilities[language] += c_probs[c]
        f.write(generate_output_str(probabilities, test_tweet))
    f.close()


def execute():
    raw_training_tweets = read(training_file)
    categorize_tweets = categorize(raw_training_tweets, v)

    total_c_counts = total_c(categorize_tweets)
    frequencies = c_frequencies_in_langs(categorize_tweets)

    cond_probs = compute_cond_probs(frequencies, total_c_counts)
    # pprint.pprint(cond_probs, width=1)

    raw_test_tweets = read(test_file)
    test_tweets = process_tweets(raw_test_tweets, v)

    output_most_prob_lang_and_required_els(test_tweets, cond_probs)
    print(compute_accuracy(v, n, s_factor))


execute()
