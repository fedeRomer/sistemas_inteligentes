from re import T
import pandas as pd

df = pd.read_csv("C:\\Users\\clear\\Documents\\Materias_2022\Sistemas Inteligentes\\Proyecto\\FakeNews\\Dataset\\archive\\train.csv")

print(df.iloc[:10])
print("--------------\n")

## Obtener Text_Tag mas largo

Text_Tag = df["Text_Tag"]
noticia_tags_mayores = []
noticia_tags_mayores_cant = 0

for n in Text_Tag:
    noticia = str(n).split(",")
    mayor = len(noticia)
    if mayor > noticia_tags_mayores_cant:
        noticia_tags_mayores_cant = mayor
        noticia_tags_mayores = noticia
        
print("Noticias con mayor cantidad de Tags: \n" + str(noticia_tags_mayores))
print("\nTags: \n", noticia_tags_mayores)
print()


