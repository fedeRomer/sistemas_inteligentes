import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

print('Leyendo dataset..')
dataset = pd.read_csv('datasets/FitbitsAndGradesData_Fall2017.csv')

dataset.dropna(inplace = True)
vect_train = CountVectorizer(stop_words='english')  # English -> Included dictionary

print('\nImprimiendo primeras y ultimas lineas del dataset')
print(dataset)

print('\nImprimiendo las columnas del dataset')
print(dataset.keys())

print('\nImprimiendo las caracteristicas de nuestro dataset')
print(dataset.describe())

print(dataset["Gender"].value_counts())
dataset["Gender"].value_counts().plot(kind="bar")
plt.title("\nDistribucion de Genero");
plt.show()

dataset["Age"].value_counts().plot(kind="bar")
plt.title("\nDistribucion de Edades");
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.show()

pd.crosstab(dataset.Age, dataset.Gender).plot(kind="bar", figsize=(15, 6), color=['#026299', '#640299'])
plt.title('Cantidad de estudiantes por genero y edad')
plt.xlabel('Genero (0 = men, 1 = women)')
plt.ylabel('Cantidad')
plt.show()

dataset.drop('Key', inplace=True, axis=1)

# realizo el split entre test y train
x = dataset
y = dataset.Gender
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

algorithm = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1, max_depth=10)
algorithm.fit(x_train, y_train)
y_pred = algorithm.predict(x_test)
scores = cross_val_score(algorithm, x_train, y_train, cv=10)
print('\npuntuacion media: ')
print(scores.mean())

print('\nmatriz de confusion')
plot_confusion_matrix(algorithm, x_test, y_test)
plt.show()

precision = precision_score(y_test, y_pred)
print('\nPrecision: ')
print(precision)

accuracy = accuracy_score(y_test, y_pred)
print('\naccuracy: ')
print(accuracy)

recall = recall_score(y_test, y_pred)
print('\nrecall: ')
print(recall)
