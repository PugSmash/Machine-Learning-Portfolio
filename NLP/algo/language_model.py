from algo.text_classifier import generate_A, clean, generate_pi
from algo.tfidf_from_scratch import word_index_mapping, count_vectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import download
from scipy.sparse import lil_matrix
import re
import numpy as np
from math import log10



def gen_A2_mat(corpus, wim, epsilon, count_vec):
    n = count_vec.sum(axis=1)
    m = len(wim)
    A_matrix = np.zeros((m, m))
    for line in corpus:
        no_punc = re.sub("[./\,<>()#¢$%!?:;&*^]", " ", line)
        tokenized_line = word_tokenize(no_punc)
        for idx, word in enumerate(tokenized_line):
            tokenized_line[idx] = word.lower()
        for i in range(len(tokenized_line) - 2):
            wd1 = tokenized_line[i]
            wd2 = tokenized_line[i+2]
            A_matrix[wim[wd1], wim[wd2]] += 1
    A_matrix += epsilon
    for i in range(m):
        for j in range(m):
            A_matrix[i, j] = A_matrix[i, j] / (n[i, 0] + epsilon*m)
    return A_matrix

def normalize(mat):
    j = 0
    sums = mat.sum(axis=0)
    for i in range(len(sums)):
        mat[i, j] = mat[i, j] / sums[i]
        j += 1
    return mat

def gen_markov_model(xtrain, EPSILON):
    global wim
    xtrain = clean(xtrain)
    wim = word_index_mapping(xtrain)
    count_vector = count_vectorizer(xtrain, wim)
    pi_mat = normalize(generate_pi(xtrain, wim, EPSILON))
    A_mat = normalize(generate_A(xtrain, wim, EPSILON, count_vector))
    A2_mat = normalize(gen_A2_mat(xtrain, wim, EPSILON, count_vector))

    return [pi_mat, A_mat, A2_mat]



ep = 100000000
data = "/Volumes/1_TB_Robo/VSCode/Data Science/NLP/datasets/robert_frost.txt"
markov_model = gen_markov_model(data, ep)

iwm = dict((v, k) for k, v in wim.items())


for i in range(4):

    first_word = np.random.choice(range(0, len(wim)), p=markov_model[0][:, 0])
    second_word = np.random.choice(range(0, len(wim)), p=markov_model[1][first_word])
    sentence = [first_word, second_word]


    for i in range(5):
        second_word = np.random.choice(range(0, len(wim)), p=markov_model[2][second_word])
        sentence.append(second_word)

    sentence_string = ""
    for word in sentence:
        sentence_string += f"{iwm[word]} "

    print(sentence_string)


