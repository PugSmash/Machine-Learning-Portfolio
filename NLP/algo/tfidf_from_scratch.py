import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import download
from scipy.sparse import lil_matrix
import re
import numpy as np
from math import log10


def word_index_mapping(corpus):
    word_index_mapping = {}
    idx = 0
    for document in corpus:
        no_punc = re.sub("[./\,<>()#¢$%!?:;&*^]", " ", document)
        tokenized = word_tokenize(no_punc)
        for token in tokenized:
            token = token.lower()
            if token not in word_index_mapping:
                word_index_mapping[token] = idx
                idx += 1
            else:
                pass
    return word_index_mapping

def count_vectorizer(corpus, word_index_mapping):
    data = lil_matrix(np.zeros((len(word_index_mapping), len(corpus))))
    for idx, document in enumerate(corpus):
        no_punc = re.sub("[./\,<>()#¢$%!?:;&*^]", " ", document)
        tokenized = word_tokenize(no_punc)
        for idx, word in enumerate(tokenized):
            tokenized[idx] = word.lower()
        for token in tokenized:
            data[word_index_mapping[token], idx] += 1
    return data

def tfidf(word_index_mapping, corpus, data_vec):
    tfidf_mat = lil_matrix(np.zeros((len(word_index_mapping), len(corpus))))
    n_ts = np.sum(data_vec, axis=1)
    for i in range(len(corpus)):
        print(f"{(i/len(corpus)) * 100}", end="\r")
        for j in range(len(word_index_mapping)):
            n = data_vec[j, i]
            n_t = n_ts[j, 0]
            if n != 0:
                idf = log10(n/n_t)
            else:
                idf = 0
            tfidf_mat[j, i] = n * idf
    return(tfidf_mat)

if __name__ == "__main__":
    download("punkt")
    data = pd.read_csv("/Volumes/1_TB_Robo/VSCode/Data Science/NLP/datasets/politifact-obama.csv")
    corpus = data["Quote"]
    wim = word_index_mapping(corpus)
    vector = count_vectorizer(corpus, wim)

    tfidf_vector = tfidf(wim, corpus, vector)
    print(tfidf_vector)