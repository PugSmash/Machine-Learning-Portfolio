import pandas as pd
import re

data =  pd.read_csv("/Volumes/1_TB_Robo/VSCode/Data Science/NLP/datasets/IMDB_Dataset.csv")

test_data = str(data["review"][0])

def dictionary(data):
    unique_words = []
    done = 0
    for doc in data:
        no_br = re.sub("<br /><br />", "", doc)
        no_x86 = re.sub("\x85", "", doc)
        no_punc = re.sub("[./\,<>()#¢$%!?:;&*-^]", " ", no_br)
        list_of_words = no_punc.split(" ")

        for token in list_of_words:
            if token:
                pass
            else:
                list_of_words.remove(token)

        

        for token in list_of_words:
            if token not in unique_words:
                unique_words.append(token)
            else:
                pass
        done += 1
        print(f"doc {done} out of {len(data)} done!", end="\r")
    return unique_words

print(dictionary(data["review"]))