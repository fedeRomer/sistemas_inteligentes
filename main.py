import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print('Leyendo dataset..')
dataset = pd.read_csv('datasets/FitbitsAndGradesData_Fall2017.csv')

print('\nImprimiendo primeras y ultimas lineas del dataset')
print(dataset)

print('\nImprimiendo las columnas del dataset')
print(dataset.keys())

print('\nImprimiendo las caracteristicas de nuestro dataset')
print(dataset.describe())

print(dataset["Gender"].value_counts())
dataset["Gender"].value_counts().plot(kind="bar")
plt.title("\nDistribucion de Clases");
plt.show()
