"""
Anime.csv
anime_id - myanimelist.net's unique id identifying an anime.
name - full name of anime.
genre - comma separated list of genres for this anime.
type - movie, TV, OVA, etc.
episodes - how many episodes in this show. (1 if movie).
rating - average rating out of 10 for this anime.
members - number of community members that are in this anime's
"group".

Rating.csv
user_id - non identifiable randomly generated user id.
anime_id - the anime that this user has rated.
rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).
"""

import numpy
import numpy as np
import pandas
# Read CSV with Pandas
import pandas as pd
from IPython.core.display_functions import display
from sklearn import neural_network
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Get train and set csv from datasets file
from sklearn.neighbors import KNeighborsClassifier

series = pandas.read_csv("datasets/anime.csv")
print(series.keys())
df = series.genre
vect = CountVectorizer(stop_words='english')
vect.fit(series.genre.fillna(' '))
print(type(vect.vocabulary_))
print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))
bag_of_words = vect.transform(series.genre.fillna(' '))
print("bag_of_words: {}".format(repr(bag_of_words)))
# print(bag_of_words[:])
# print("Dense representation of bag_of_words:\n {}".format(bag_of_words.toarray()))
s = bag_of_words[:, 0]
result = []
for i in s:
    if i[0] == 1:
        result.append(1)
    else:
        result.append(0)
print(repr(s))
print(repr(result))
print(repr(series.episodes))
df_test = pandas.DataFrame()
df_test.insert(0, "genre_0", result, allow_duplicates=True)
print(df_test)
# genre_elegido = pandas.DataFrame()
# print(genre_elegido)
series_df = pandas.DataFrame(df_test, series.episodes, series.rating, series.members)
users = pandas.read_csv("datasets/rating.csv")
users_df = pandas.DataFrame(users)

kmeans = KMeans(n_clusters=5)
kmeans = kmeans.fit(series_df)
labels = kmeans.predict(series_df)
centroids = kmeans.cluster_centers

# info = pd.merge(series_df, users_df, on="anime_id")
# print(info.keys())

# Prepare data
# X_train, X_test, y_train, y_test = train_test_split(train_set.Text, train_set['Labels'], random_state=0)
# scores = cross_val_score(neural_network.MLPClassifier(), bag_of_words.toarray(), train_set.Labels, cv=10, n_jobs=5)
# print(scores)

# cluster = KMeans().fit(train_set)
