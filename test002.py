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
from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Get train and test set csv from datasets file
train_set = pandas.read_csv("datasets/train.csv")
print(train_set.keys())

# Prepare data / Data Selection and Formatting
vect_train = CountVectorizer(stop_words='english')  # English -> Included dictionary
vect_train.fit(train_set.Text)
bag_train = vect_train.transform(train_set.Text)
print("bag_of_words: {}".format(repr(bag_train)))
print("Dense representation of bag_of_words:\n {}".format(bag_train.toarray()))
print(len(vect_train.get_feature_names()))

# Vectors to array
bag_train_array = bag_train.toarray()

# Prepare Algorithm
X_train, X_test, y_train, y_test = train_test_split(bag_train_array, train_set.Labels, random_state=0)
random_forest = RandomForestClassifier().fit(X_train, y_train)

# Score
print(cross_val_score(RandomForestClassifier(), bag_train_array, train_set.Labels, cv=10, n_jobs=5))

# Predict
pred_y = cross_val_predict(random_forest, X_test, y_test, cv=10, n_jobs=5)
print(np.mean(pred_y == y_test))

# Confusion Matrix
confusion = confusion_matrix(y_test, pred_y)
print("Matriz de confusion:\n{}".format(confusion))

# Classification report
print(classification_report(y_test, pred_y, target_names=['0', '1', '2', '3', '4', '5']))