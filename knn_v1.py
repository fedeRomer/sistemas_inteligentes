import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


def preprocess_text(text):          #Elimina las columnas con numeros 
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

train_set = pd.read_csv("C:\\Users\\clear\\Documents\\Materias_2022\Sistemas Inteligentes\\Proyecto\\FakeNews\\Dataset\\archive\\train.csv", dtype="str")

train_set.dropna(inplace = True)

vect_train = CountVectorizer(stop_words='english', min_df=5, binary=True, lowercase=True, preprocessor=preprocess_text)  # English -> Included dictionary
vect_train.fit(train_set.Text)
bag_train = vect_train.transform(train_set.Text) #Columna texto
bag_train_array = bag_train.toarray()

tags_train = CountVectorizer(stop_words="english", binary=True)
tags_train.fit(train_set.Text_Tag)
bag_train_tags = tags_train.transform(train_set.Text_Tag)
bag_tags_array = bag_train_tags.toarray()


dfText = pd.DataFrame(data=bag_train_array,columns = vect_train.get_feature_names_out())

dfTags = pd.DataFrame(data=bag_tags_array,columns = tags_train.get_feature_names_out())
Y = dfTags["military"]

X_train, X_test, y_train, y_test = train_test_split(dfText, Y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)

print("Matriz de confusion:\n{}".format(confusion))

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels=[True,False])
cm_display.plot()
plt.show()

print("Accuracy:",accuracy_score(y_test, y_pred))

print(cross_val_score(knn,dfText,Y, cv=10, n_jobs=1) )

# Classification report
print(classification_report(y_test, y_pred))