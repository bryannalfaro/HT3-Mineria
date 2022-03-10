#Universidad del Valle de Guatemala
#Mineria de Datos
#HT3 Arboles
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera


from cgi import test
from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import Counter
from sklearn import preprocessing, tree
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import classification_report, multilabel_confusion_matrix, silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pyclustertend
import random
import graphviz
import sklearn.mixture as mixture
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split

houses = pd.read_csv('train.csv', encoding='latin1', engine='python')

'''
#Conocimiento de datos
print(houses.head())

#Cantidad de observaciones y variables en la base
print(houses.shape)

#Medidas estadisticas.
print(houses.describe().transpose())

print(houses.select_dtypes(exclude=['object']).info())'''


'''#Casas que ofrecen todas las utilidades
print(houses['Utilities'].value_counts())

plt.bar(houses['Utilities'].value_counts().sort_index().dropna().index, houses['Utilities'].value_counts().sort_index().values, color='red')
plt.title('Grafico de barras para utilidades')
plt.xlabel('Utilidad')
plt.xticks(rotation=90)
plt.ylabel('Cantidad de casas')
plt.tight_layout()
plt.show()

#Calidad de casas predominante
print(houses['OverallCond'].value_counts())

plt.bar(houses['OverallCond'].value_counts().sort_index().dropna().index, houses['OverallCond'].value_counts().sort_index().values, color='red')
plt.title('Grafico de barras para condicion de las casas')
plt.xlabel('Condicion')
plt.ylabel('Cantidad de casas')
plt.show()

#Año de mas y menos produccion de casas para
print(houses['YearBuilt'].value_counts().sort_values(ascending=False).head(1))
print(houses['YearBuilt'].value_counts().sort_values(ascending=True).head(15))

#Capacidad de carros de las 5 casas mas caras y baratas

print(houses.sort_values(by='SalePrice', ascending=False)[['GarageCars','SalePrice']].head(5))
print(houses.sort_values(by='SalePrice', ascending=True)[['GarageCars','SalePrice']].head(5))

#Condicion de garage y calidad de la cocina de las 5 casas mas caras
print(houses.sort_values(by='SalePrice', ascending=False)[['GarageCond','KitchenQual','SalePrice']].head(5))'''

houses_clean = houses.select_dtypes(exclude='object').drop('Id', axis=1)

'''#preprocesamiento
corr_data = houses_clean.iloc[:,:]
mat_correlation=corr_data.corr() # se calcula la matriz , usando el coeficiente de correlacion de Pearson
plt.figure(figsize=(16,10))

#Realizando una mejor visualizacion de la matriz
sns.heatmap(mat_correlation,annot=True,cmap='BrBG')
plt.title('Matriz de correlaciones  para la base Houses')
plt.tight_layout()
plt.show()'''

# Seleccion de variables
houses_df = houses_clean[['OverallQual', 'OverallCond', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'Fireplaces',
'GarageCars', 'GarageArea', 'GarageYrBlt','TotRmsAbvGrd','SalePrice']]
'''
print(houses_df.head().dropna())
print(houses_df.info())
print(houses_df.describe().transpose())
'''
houses_df.fillna(0)

#normalizar
df_norm  = (houses_df-houses_df.min())/(houses_df.max()-houses_df.min())
#print(movies_clean_norm.fillna(0))
houses_df_final = df_norm.fillna(0)

print(houses_df_final.describe().transpose())

#Analisis de tendencia a agrupamiento

#Metodo Hopkings
'''
random.seed(200)
print(pyclustertend.hopkins(houses_df_final, len(houses_df_final)))

#Grafico VAT e iVAT
x = houses_df_final.sample(frac=0.1)
pyclustertend.vat(x)
plt.show()
pyclustertend.ivat(x)
plt.show()

# Numero adecuado de grupos
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(houses_df_final)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Grafico de codo')
plt.xlabel('No. Clusters')
plt.ylabel('Puntaje')
plt.show()
'''
#Kmeans
clusters=  KMeans(n_clusters=3, max_iter=300) #Creacion del modelo
clusters.fit(houses_df_final) #Aplicacion del modelo de cluster

houses_df_final['cluster'] = clusters.labels_ #Asignacion de los clusters
print(houses_df_final.head())

pca = PCA(2)
pca_movies = pca.fit_transform(houses_df_final)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, houses_df_final[['cluster']]], axis = 1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Clusters de casas', fontsize = 20)

color_theme = np.array(['red', 'green', 'blue', 'yellow','black'])
ax.scatter(x = pca_clust_movies.PC1, y = pca_clust_movies.PC2, s = 50, c = color_theme[pca_clust_movies.cluster])

plt.show()
print(pca_clust_movies)

houses_df['Cluster'] = houses_df_final['cluster']
print((houses_df[houses_df['Cluster']==0]).describe().transpose())
print((houses_df[houses_df['Cluster']==1]).describe().transpose())
print((houses_df[houses_df['Cluster']==2]).describe().transpose())

# Creacion de la respuesta (clasificacion de casas en: Economicas, Intermedias o Caras)
minSalesPrice = houses_df['SalePrice'].min()
maxSalesPrice = houses_df['SalePrice'].max()
terceraParte = (maxSalesPrice - minSalesPrice) / 3
houses_df['Clasificacion'] = houses_df['SalePrice']
# Clasificar casas a partir del precio "SalePrice"
houses_df['Clasificacion'] = houses_df['Clasificacion'].apply(lambda x: 'Economicas' if x < minSalesPrice + terceraParte else 'Intermedias' if x < minSalesPrice + 2 * terceraParte else 'Caras')
# Clasificar casas a partir del precio y su condición general "OverallCond"
houses_df['Clasificacion'] = houses_df.apply(lambda row: 'Caras' if ((row['Clasificacion'] == 'Intermedias' and row['OverallCond'] < 4 ) or (row['Clasificacion'] == 'Economicas' and row['OverallCond'] < 2))
                                                    else 'Intermedias' if (row['Clasificacion'] == 'Economicas' and (1 < row['OverallCond'] < 4))
                                                    else 'Economicas' if (row['Clasificacion'] == 'Intermedias' and row['OverallCond'] > 7)
                                                    else row['Clasificacion'], axis=1)
# Convertir Clasaficacion a categorica
houses_df['Clasificacion'] = houses_df.apply(lambda row: 1 if row['Clasificacion'] == 'Economicas' else 2 if row['Clasificacion'] == 'Intermedias' else 3, axis=1)
'''
# Ver distribucion del precio, condicion general y clasificacion
houses_df['SalePrice'].hist()
plt.title('Histograma de precios')
plt.show()
houses_df['OverallCond'].hist()
plt.title('Histograma de condicion general')
plt.show()
houses_df['Clasificacion'].hist()
plt.title('Histograma de clasificacion')
plt.show()
'''
# Division de datos, 70% de entrenamiento y 30% de prueba, manteniendo distribucion de clasificacion
y = houses_df.pop('Clasificacion')
x = houses_df
x.pop('MasVnrArea')
x.pop('GarageYrBlt')
x.pop('Cluster') # Se elimina para que al analizar el set de pruebas no se tenga en cuenta

random.seed(4315)

dividir = StratifiedShuffleSplit(n_splits=10, train_size=0.7, test_size=0.3, random_state=416)
for train_index, test_index in dividir.split(x, y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Modelo de arbol de decision
Dt_model = tree.DecisionTreeClassifier(random_state=0)

x_train.fillna(0)
y_train.fillna(0)

Dt_model.fit(x_train, y_train)

tree.plot_tree(Dt_model, feature_names=houses_df.columns, class_names=['Económicas', 'Intermedias', 'Caras'], filled=True, rounded=True)
plt.show()

# Modelo de arbol de regresión
y_reg = houses_df.pop('SalePrice')
x_reg = houses_df

x_reg.pop('MasVnrArea')
x_reg.pop('GarageYrBlt')
x_reg.pop('Cluster')
x_reg.pop('Clasificacion') 

random.seed(5236)

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.3, train_size=0.7, random_state=0)

Dt_model_reg = tree.DecisionTreeRegressor(random_state=0, max_leaf_nodes=20)

x_train_reg.fillna(0)
y_train_reg.fillna(0)

Dt_model_reg.fit(x_train_reg, y_train_reg)

plt.figure(figsize=(23, 10))
tree.plot_tree(Dt_model, feature_names=houses_df.columns, fontsize=7, filled=True, rounded=True)
plt.show()

# Eficiencia del algoritmo de arbol de clasificacion
y_pred_reg = Dt_model_reg.predict(X = x_test_reg)

rmse = metrics.mean_squared_error(
        y_true  = y_test_reg,
        y_pred  = y_pred_reg,
        squared = False
)
print("-----------------------------------")
print(f"El error (rmse) de test es: {rmse}")
print("-----------------------------------")

cm = multilabel_confusion_matrix(y_test_reg, y_pred_reg)
print('\nClassification Report:')
print(classification_report(y_test_reg, y_pred_reg))

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Económicas', 'Intermedias', 'Caras'], yticklabels=['Económicas', 'Intermedias', 'Caras'])
plt.title('Matriz de Confusion')
plt.ylabel('Clasificación real')
plt.xlabel('Clasificación predicha')
plt.show()

# Cargando el set de entrenamiento para problema 9
houses_test = pd.read_csv('train.csv', encoding='latin1', engine='python')
houses_test = houses_test.select_dtypes(exclude='object').drop('Id', axis=1)
houses_test = houses_test[['OverallQual', 'OverallCond', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'Fireplaces',
'GarageCars', 'GarageArea', 'GarageYrBlt','TotRmsAbvGrd','SalePrice']]
houses_test.fillna(0)

# Calculando el y_train con el set de entrenamiento real
minSalesPrice = houses_test['SalePrice'].min()
maxSalesPrice = houses_test['SalePrice'].max()
terceraParte = (maxSalesPrice - minSalesPrice) / 3
houses_test['Clasificacion'] = houses_test['SalePrice']
# Clasificar casas a partir del precio "SalePrice"
houses_test['Clasificacion'] = houses_test['Clasificacion'].apply(lambda x: 'Economicas' if x < minSalesPrice + terceraParte else 'Intermedias' if x < minSalesPrice + 2 * terceraParte else 'Caras')
# Clasificar casas a partir del precio y su condición general "OverallCond"
houses_test['Clasificacion'] = houses_test.apply(lambda row: 'Caras' if ((row['Clasificacion'] == 'Intermedias' and row['OverallCond'] < 4 ) or (row['Clasificacion'] == 'Economicas' and row['OverallCond'] < 2))
                                                    else 'Intermedias' if (row['Clasificacion'] == 'Economicas' and (1 < row['OverallCond'] < 4))
                                                    else 'Economicas' if (row['Clasificacion'] == 'Intermedias' and row['OverallCond'] > 7)
                                                    else row['Clasificacion'], axis=1)
# Convertir Clasaficacion a categorica
houses_test['Clasificacion'] = houses_test.apply(lambda row: 1 if row['Clasificacion'] == 'Economicas' else 2 if row['Clasificacion'] == 'Intermedias' else 3, axis=1)

y_test = houses_test.pop('Clasificacion')
houses_test.pop('MasVnrArea')
houses_test.pop('GarageYrBlt')

# 9. Calcular eficiencia del algoritmo usando matriz de confusion
y_pred = Dt_model.predict(houses_test)
cm = confusion_matrix(y_test, y_pred)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Económicas', 'Intermedias', 'Caras'], yticklabels=['Económicas', 'Intermedias', 'Caras'])
plt.title('Matriz de Confusion')
plt.ylabel('Clasificación real')
plt.xlabel('Clasificación predicha')
plt.show()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted')) # macro, micro, weighted
print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted')) # macro, micro, weighted
print("F1:", metrics.f1_score(y_test, y_pred, average='weighted')) # macro, micro, weighted
