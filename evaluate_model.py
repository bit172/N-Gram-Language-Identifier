import numpy as np
import io
from decimal import Decimal
from utils import *


def create_confusion_matrix(nb_of_languages):
    return np.zeros((nb_of_languages, nb_of_languages), dtype=np.uint16)


def evaluate_model(trace_file):
    lang_idx = {"eu": 0, "ca": 1, "gl": 2, "es": 3, "en": 4, "pt": 5}
    precision_per_lang = np.zeros(6, dtype=np.float64)
    recall_per_lang = np.zeros(6, dtype=np.float64)
    f1_per_lang = np.zeros(6, dtype=np.float64)

    confusion_matrix = create_confusion_matrix(6)
    trace = read(trace_file)
    size = len(trace)

    for line in trace:
        values = line.split()
        predicted, actual = values[1], values[3]
        confusion_matrix[lang_idx[predicted], lang_idx[actual]] += 1

    accuracy = compute_accuracy(confusion_matrix, size)
    w_a_f1 = 0

    for idx in lang_idx.values():
        precision_per_lang[idx] = compute_precision(confusion_matrix, idx)
        recall_per_lang[idx] = compute_recall(confusion_matrix, idx)
        f1_per_lang[idx] = compute_f1_measure(precision_per_lang[idx], recall_per_lang[idx])
        w_a_f1 += (confusion_matrix.T[idx].sum() * f1_per_lang[idx]) / size

    macro_f1 = np.average(f1_per_lang)

    f = io.open("./results/eval" + trace_file[15:], "w")
    f.write(f"{'%.4f' % Decimal(accuracy)}\r")
    f.write(np.array2string(precision_per_lang, precision=4, separator="  ", floatmode="fixed")[1:-1] + "\r")
    f.write(np.array2string(recall_per_lang, precision=4, separator="  ", floatmode="fixed")[1:-1] + "\r")
    f.write(np.array2string(f1_per_lang, precision=4, separator="  ", floatmode="fixed")[1:-1] + "\r")
    f.write(f"{'%.4f' % Decimal(macro_f1)}  {'%.4f' % Decimal(w_a_f1)}")
    f.close()


def compute_accuracy(confusion_matrix, total):
    return np.sum(confusion_matrix.diagonal()) / total


def compute_recall(confusion_matrix, idx):
    try:
        col_sum = confusion_matrix.T[idx].sum()
        if col_sum == 0:
            raise ZeroDivisionError
        return confusion_matrix[idx, idx] / col_sum
    except ZeroDivisionError:
        return 0.0


def compute_precision(confusion_matrix, idx):
    try:
        row_sum = confusion_matrix[idx].sum()
        if row_sum == 0:
            raise ZeroDivisionError
        return confusion_matrix[idx, idx] / row_sum
    except ZeroDivisionError:
        return 0.0


def compute_f1_measure(precision, recall):
    try:
        if recall + precision == 0:
            raise ZeroDivisionError
        return (2 * precision * recall) / (recall + precision)
    except ZeroDivisionError:
        return 0.0
