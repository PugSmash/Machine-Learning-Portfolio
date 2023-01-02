from algo.tfidf_from_scratch import word_index_mapping, count_vectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import download
from scipy.sparse import lil_matrix
import re
import numpy as np
from math import log10

EPSILON = 1000000

def clean(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    for idx, line in enumerate(lines):
        lines[idx] = re.sub("\n", "", line)

    for line in lines:
        if line:
            pass
        else:
            lines.remove(line)
    return lines


def generate_pi(corpus, wim, epsilon):
    pi_matrix = np.zeros((len(wim), 1))
    n = len(corpus)
    m = len(wim)
    for line in corpus:
        no_punc = re.sub("[./\,<>()#¢$%!?:;&*^]", " ", line)
        tokenized_line = word_tokenize(no_punc)
        for idx, word in enumerate(tokenized_line):
            tokenized_line[idx] = word.lower()
        first_word_index = wim[tokenized_line[0]]
        pi_matrix[first_word_index, 0] += 1
    #pi_matrix += epsilon
    pi_matrix = pi_matrix / (n)
    return pi_matrix

def generate_A(corpus, wim, epislon, count_vec):
    n = count_vec.sum(axis=1)
    m = len(wim)
    A_matrix = np.zeros((m, m))
    for line in corpus:
        no_punc = re.sub("[./\,<>()#¢$%!?:;&*^]", " ", line)
        tokenized_line = word_tokenize(no_punc)
        for idx, word in enumerate(tokenized_line):
            tokenized_line[idx] = word.lower()
        for i in range(len(tokenized_line) - 1):
            wd1 = tokenized_line[i]
            wd2 = tokenized_line[i+1]
            A_matrix[wim[wd1], wim[wd2]] += 1
    A_matrix += epislon
    for i in range(m):
        for j in range(m):
            A_matrix[i, j] = A_matrix[i, j] / (n[i, 0] + epislon*m)
    return A_matrix


def gen_markov_model(xtrain, EPSILON):
    global wim
    wim = word_index_mapping(xtrain)
    count_vector = count_vectorizer(xtrain, wim)
    pi_mat = generate_pi(xtrain, wim, EPSILON)
    A_mat = generate_A(xtrain, wim, EPSILON, count_vector)
    return [pi_mat, A_mat]

def test_train_split(train, path, label):
    lines = clean(path)
    Xtrain = lines[0:int(len(lines)*train)]
    Xtest = lines[int(len(lines)*train + 1):]
    Ytrain = [label for i in range(len(lines))]
    Ytest = [label for i in range(len(lines))]
    return [Xtrain, Ytrain, Xtest, Ytest]

def make_prediction(markov_model, input_sequence, wim):
    no_punc = re.sub("[./\,<>()#¢$%!?:;&*^]", " ", input_sequence)
    tokenized_line = word_tokenize(no_punc)
    try:
        prob = markov_model[0][wim[tokenized_line[0]], 0]
    except:
        prob = 1
    tokenized_line.pop(0)
    for idx, token in enumerate(tokenized_line):
        try:
            prob = prob * markov_model[1][wim[token], wim[tokenized_line[idx+1]]]
        except:
            pass
    return prob

def classify(mm1, mm2, input_sequece, wim):
    pred1 = make_prediction(mm1, input_sequece, wim)
    pred2 = make_prediction(mm2, input_sequece, wim)
    return np.argmax([pred1, pred2])


if __name__ == "__main__":
    eap_test_train_split = test_train_split(0.7, "/Volumes/1_TB_Robo/VSCode/Data Science/NLP/datasets/edgar_allan_poe.txt", 0)
    rf_test_train_split = test_train_split(0.7, "/Volumes/1_TB_Robo/VSCode/Data Science/NLP/datasets/edgar_allan_poe.txt", 1)

    eap_mrkvmdl = gen_markov_model(eap_test_train_split[0], EPSILON)
    rf_mrkvmdl = gen_markov_model(rf_test_train_split[0], EPSILON)
    correct = 0


    for idx, input_seq in enumerate(eap_test_train_split[2]):
        pred = classify(eap_mrkvmdl, rf_mrkvmdl, input_seq, wim)
        if pred == eap_test_train_split[3][idx]:
            correct += 1
        else:
            pass

    print(f"Accuracy: {correct/len(eap_test_train_split[3])}")