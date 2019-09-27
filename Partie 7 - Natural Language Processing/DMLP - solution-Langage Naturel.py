import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('NLP_MLP_last.csv')


from collections import Counter

titres = ["PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"]


mot_uniques = list(set(" ".join(titres).split(" ")))
def make_matrix(titres, vocab):
    matrix = []
    for titre in titres:

        counter = Counter(titre)

        row = [counter.get(w, 0) for w in vocab]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = mot_uniques
    return df

print(make_matrix(titres, mot_uniques))


import re


nv_titres = [re.sub(r'[^\w\s\d]','',h.lower()) for h in titres]

nv_titres = [re.sub("\s+", " ", h) for h in nv_titres]

mot_uniques = list(set(" ".join(nv_titres).split(" ")))

print(make_matrix(nv_titres, mot_uniques))



with open("stop_words.txt", 'r') as f:
    stopwords = f.read().split("\n")


stopwords = [re.sub(r'[^\w\s\d]','',s.lower()) for s in stopwords]

mot_uniques = list(set(" ".join(nv_titres).split(" ")))

mot_uniques = [w for w in mot_uniques if w not in stopwords]


print(make_matrix(nv_titres, mot_uniques))




from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(lowercase=True, stop_words="english")

matrix = vectorizer.fit_transform(titres)

print(matrix.todense())

dataset['full_test'] = dataset["titre"] + " " + dataset["url"]
full_matrix = vectorizer.fit_transform(dataset["titre"].values.astype('U'))
print(full_matrix.shape)







from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(full_matrix, dataset["approbation"], test_size = 0.20, random_state = 0)







from sklearn.linear_model import Ridge



reg = Ridge(alpha=.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

