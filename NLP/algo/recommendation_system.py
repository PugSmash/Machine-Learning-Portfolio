from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import json


data =  pd.read_csv("/Volumes/1_TB_Robo/VSCode/Data Science/NLP/datasets/tmdb_5000_movies.csv")
train_texts = []

for i in range(len(data)):
    temp_keywordstr = ""
    keywords = json.loads(data["keywords"][i])
    for item in keywords:
        temp_keywordstr += f" {item['name']}"
    temp_genrestr = ""
    genres = json.loads(data["genres"][i])
    for item in genres:
        temp_genrestr += f" {item['name']}"
    overview = str(data["overview"][i])
    doc = overview + temp_genrestr + temp_keywordstr
    no_punc = re.sub("[./\,<>()#¢$%!?:;&*-^]", " ", doc)
    train_texts.append(doc)

tfidf = TfidfVectorizer(max_features=2000)
xtrain = tfidf.fit_transform(train_texts)
# print(xtrain)

def get_info(movie, database):
    row = database.loc[database["original_title"] == movie]
    temp_keywordstr = ""
    keywords = json.loads(row["keywords"].values[0])
    for item in keywords:
        temp_keywordstr += f" {item['name']}"
    temp_genrestr = ""
    genres = json.loads(row["genres"].values[0])
    for item in genres:
        temp_genrestr += f" {item['name']}"
    overview = str(row["overview"])
    doc = overview + temp_genrestr + temp_keywordstr
    no_punc = re.sub("[./\,<>()#¢$%!?:;&*-^]", " ", doc)
    return [no_punc, row.index]

def recommend(movie, database, tfidf):
    input_str = get_info(movie, database)
    vec = tfidf.transform([input_str[0]])
    best = (-1, 0)
    for idx, vec2 in enumerate(xtrain):
        diff = cosine_similarity(vec, vec2)
        if diff > best[0] and idx != input_str[1]:
            best = (diff, idx)
            print(database.iloc[[idx]]['original_title'].values, diff)
        else:
            print(database.iloc[[idx]]['original_title'].values, diff)
            pass
    print(f"The movie best suited to you would be {str(database.iloc[[best[1]]]['original_title'])} With a similarity of {best[0]}")
        
recommend("Love Actually", data, tfidf)
