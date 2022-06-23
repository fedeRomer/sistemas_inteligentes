import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn import neural_network
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


train_set = pd.read_csv("C:\\Users\\clear\\Documents\\Materias_2022\Sistemas Inteligentes\\Proyecto\\FakeNews\\Dataset\\archive\\train.csv")

train_set.dropna(inplace = True)

vect_train = CountVectorizer(stop_words='english')  # English -> Included dictionary
vect_train.fit(train_set.Text)
bag_train = vect_train.transform(train_set.Text) #Columna texto
bag_train_array = bag_train.toarray()

dfText = pd.DataFrame(data=bag_train_array,columns = vect_train.get_feature_names_out())

tags_train = CountVectorizer(stop_words="english")
tags_train.fit(train_set.Text_Tag)
bag_train_tags = tags_train.transform(train_set.Text_Tag)
bag_tags_array = bag_train_tags.toarray()

dfTags = pd.DataFrame(data=bag_tags_array,columns = tags_train.get_feature_names_out())

Y = dfTags["military"] #Columna objetivo
X = dfText 
    


X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.4, random_state=0)

random_forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=30 ,oob_score=False , n_jobs=1 , max_depth=10).fit(X_train, y_train)

#feature_imp = pd.Series(random_forest.feature_importances_,index=list(dfText.columns.values)).sort_values(ascending=False)

print(cross_val_score(random_forest,X,Y, cv=10, n_jobs=1) )

# Predict
pred_y = cross_val_predict(random_forest, X_test, y_test, cv=10, n_jobs=1)

print("Accuracy:",accuracy_score(y_test, pred_y))

# Confusion Matrix
confusion = confusion_matrix(y_test, pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels=[True,False])
cm_display.plot()
plt.show()
print("Matriz de confusion:\n{}".format(confusion))

# Classification report
print(classification_report(y_test, pred_y))