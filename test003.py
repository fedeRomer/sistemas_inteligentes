"""
Labels - Represents various classes of Labels
Half-True - 2
False - 1
Mostly-True - 3
True - 5
Barely-True - 0
Not-Known - 4
"""
import numpy as np
import pandas
# Read CSV with Pandas
# from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Get train and test set csv from datasets file
train_set = pandas.read_csv("datasets/train.csv")
print(train_set.keys())

vect_text = CountVectorizer(stop_words='english')  # English -> Included dictionary
vect_text.fit(train_set.Text)
vect_tags = CountVectorizer(stop_words='english')
bag_tags = vect_tags.fit_transform(train_set['Text_Tag'].values.astype(str))   # Even astype(str) would wor
bag_text = vect_text.transform(train_set.Text)
# print(type(bag_text))
bag_text_array = bag_text.toarray()
bag_tag_array = bag_tags.toarray()
print(len(bag_tag_array))
print(len(bag_text_array))

# print(bag_text_array)
# print(type(bag_text_array))

# random_forest = RandomForestClassifier().fit(X_train, y_train)
