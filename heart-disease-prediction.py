# -*- coding: utf-8 -*-
"""
Entrenando algoritmos. Resolviendo problemas de aprendizaje supervisado.
 Predicción de enfermedades del corazón
"""

# Importación de las librerias necesarias
# Manipulación de datos
import pandas as pd
import numpy as np
# Visualización
import seaborn as sns
import matplotlib.pyplot as plt
# Preproceso y división de datos
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Evaluación y metricas
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Importación de los datos
df = pd.read_csv("/heart.csv")

# Mostrar las primeras filas del dataframe
df.head()

# Mostrar las últimas filas del dataframe
df.tail()

# Mostrar media de los datos
df.describe(include="all").round(1)

# Mostrar un mapa de calor que indique la correlación entre las variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,fmt=".2f",cmap="coolwarm")
plt.title("Mapa de calor para ver la correlación entre las variables")
plt.show()

# Seleccionar las características a tener en cuenta en el estudio
## Respecto a que características voy a usar, decido en quedarme con las 11 variables de entreada que podemos ver, ya que tenemos un dataframe "pequeño", y como podemos ver en el mapa de calor,
## encontramos correlaciones debiles entre todas las características del dataframe, por lo que podríamos obtener resultados incorrectos o poco optimos debido a prescindir de algunos datos.

# Separar los datos de entrada y etiqueta
# Separamos los datos de entrada
X = df.drop('HeartDisease',axis=1)
# Separamos las etiquetas
y = df['HeartDisease']
# El valor x tiene valores alfabéticos, y para kNN y arbol de decisiones, solo se aceptan números
X =  pd.get_dummies(X,drop_first=True,dtype=int)
X.head()

# Separar datos entre entrenamiento y prueba (usando un 75% para entrenamiento y 25% para test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)

# Elegir, instanciar y entrenar el modelo
scaler = StandardScaler()
# Se escalan los datos para kNN porque el algoritmo se basa en distancias
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento y predicción kNN
# Instanciar el modelo kNN
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')

# Entrenamos el modelo con los datos escalados
knn.fit(X_train_scaled, y_train)

# Realizamos la predicción con los datos de prueba
y_pred_knn = knn.predict(X_test_scaled)

# Obtenemos el porcentaje de excatitud de la predicción
accuracy_result_knn = accuracy_score (y_test,y_pred_knn)
print('El porcentaje de exactitud es de ' + str(accuracy_result_knn))

#Mostramos la predicción realizada
print(y_pred_knn)

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_knn)

# Entrenamiento y predicción arbol de decisiones
# Instanciar el modelo de arbol de decisiones
tree = DecisionTreeClassifier(max_depth = 5, criterion = 'gini')

# Entrenamos el modelo con los datos de entrenamiento
tree.fit(X_train,y_train)

# Realizamos la predicción
y_pred_tree = tree.predict(X_test)

# Obtenemos el porcentaje de excatitud de la predicción
accuracy_result_tree = accuracy_score(y_test,y_pred_tree)

# Mostramos los resultados
print("El procentaje de exactitud es de " + str(accuracy_result_tree))

# Mosttramos la predicción realizada
print(y_pred_tree)

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_tree)

# Optimización de los hiperparámetros para el algortimo kNN
# Validación cruzada de 5 particiones
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Bucle para pobrar con uniform y distance con un k de 1 al 30
resultados_knn = []
for i in ['uniform','distance']:
  for k in range(1, 31):
    modelo = KNeighborsClassifier(n_neighbors=k,weights=i)
    scores = cross_val_score(modelo,X_train_scaled, y_train, cv=kf, scoring='accuracy')
    resultados_knn.append({'weights':i, 'k':k, 'accuracy_media': scores.mean()})
# Pasamos los datos obtenidos a un dataframe
knn_dataFrame = pd.DataFrame(resultados_knn)
# Obtenemos la media más alta
mejor_knn = knn_dataFrame.loc[knn_dataFrame['accuracy_media'].idxmax()]
print("La mejor combinación kNN es " + str(mejor_knn))

#Modelo final con los parámetros óptimos
knn_final = KNeighborsClassifier(n_neighbors=int(mejor_knn['k']),weights=mejor_knn['weights'])
knn_final.fit(X_train_scaled, y_train)
y_pred_knn_final = knn_final.predict(X_test_scaled)

# Calculamos y mostramosla predicción con los mejores parámetros encontrados
accuracy_result_knn_final = accuracy_score(y_test,y_pred_knn_final)
print('El porcentaje de exactitutd con los nuevos valores optimizados es ' + str(accuracy_result_knn_final))

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_knn_final)

# Optimización de los hiperparámetros para el algortimo árboles de decisión
# Bucle para pobrar con distintos criterios y con una profundidad que va del 1 al 15
resultados_tree = []
for criterio in ['gini','entropy']:
  for profundidad in range(1,16):
    modeloArbol = DecisionTreeClassifier(max_depth=profundidad, criterion=criterio, random_state=1)
    scoresArbol = cross_val_score(modeloArbol, X_train_scaled, y_train, cv=kf,scoring='accuracy')
    resultados_tree.append({'criterion':criterio,'max_depth':profundidad,'accuracy_mean':scoresArbol.mean()})

# Pasamos los datos a un dataframe
tree_dataFrame = pd.DataFrame(resultados_tree)

# Obtenemos la media más alta del dataframe
mejor_tree = tree_dataFrame.loc[tree_dataFrame['accuracy_mean'].idxmax()]
print('Los mejor combinación para el arbol de decisiones es'+ str(mejor_tree))

# Instanciamos el modelo con los nuevos parametros
tree_final = DecisionTreeClassifier(max_depth=mejor_tree['max_depth'], criterion= mejor_tree['criterion'], random_state=1)

# Entrenamos el modelo
tree_final.fit(X_train_scaled,y_train)

# Obtenemos la predicción del modelo
y_pred_tree_final = tree_final.predict(X_test_scaled)

# Calculamos el porcentaje de exactitud
accuracy_result_tree_final = accuracy_score(y_test,y_pred_tree_final)

# Pintamos el resultado
print('El porcentaje de exactitutd con los nuevos valores optimizados es ' + str(accuracy_result_tree_final))

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_tree_final)

"""
Conclusiones.
El mejor modelo es kNN con k=17 y pesos por distancia (89,6% de exactitud en test; 86,5% en validación cruzada).
El árbol de decisión óptimo (profundidad 3, criterio gini) obtuvo 83,0% en test. Con los parámetros iniciales el árbol había obtenido un 87,4%,
pero la optimización por validación cruzada se hace solo con los datos de entrenamiento,
y con un conjunto de prueba pequeño (230 pacientes) las diferencias de unos puntos pueden deberse al azar. Se escalaron los datos porque kNN se basa en distancias.
"""