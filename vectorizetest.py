from ftplib import ftpcp
import pandas as pd
import numpy as np

from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy import sparse


train_set = pd.read_csv("C:\\Users\\clear\\Documents\\Materias_2022\Sistemas Inteligentes\\Proyecto\\FakeNews\\Dataset\\archive\\train.csv")

train_set.dropna(inplace = True)

vect_train = CountVectorizer(stop_words='english')  # English -> Included dictionary
vect_train.fit(train_set.Text)

bag_train = vect_train.transform(train_set.Text) #Columna texto
bag_train_array = bag_train.toarray()


tags_train = CountVectorizer(lowercase=True)
tags_train.fit(train_set.Text_Tag)
bag_train_tags = tags_train.transform(train_set.Text_Tag)

print(tags_train.get_feature_names_out()[4])
target_column=bag_train_tags.getcol(4)

print(bag_train.get_shape())

final_matrix = sparse.hstack([bag_train,target_column], format="csr")
print(final_matrix.get_shape())

#X_train, X_test, y_train, y_test = train_test_split(bag_train_array, train_set.Labels, test_size=0.30, random_state=0)

#random_forest = RandomForestClassifier().fit(X_train, y_train)

#random_forest = RandomForestClassifier().fit(X_train, y_train)

# Score
#print(cross_val_score(RandomForestClassifier(), bag_train_array, train_set.Labels, cv=10, n_jobs=5))

# Predict
#pred_y = cross_val_predict(random_forest, X_test, y_test, cv=10, n_jobs=5)
#print(np.mean(pred_y == y_test))

#print(f_train_text.head(5))
#----------------------------------


#f_train_text = f_train_text.join(abortion)

#concat_matrix = hstack([bag_train,bag_train_tags])

#print(concat_matrix)

#--------------------------------------

#X_train, X_test, y_train, y_test = train_test_split(bag_train_array, train_set.Label, test_size=0.30, random_state=0)

#print(y_test)