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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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

plt.bar(houses['Utilities'].value_counts().sort_index().dropna().index,houses['Utilities'].value_counts().sort_index().values,color='red')
plt.title('Grafico de barras para utilidades')
plt.xlabel('Utilidad')
plt.xticks(rotation=90)
plt.ylabel('Cantidad de casas')
plt.tight_layout()
plt.show()

#Calidad de casas predominante
print(houses['OverallCond'].value_counts())

plt.bar(houses['OverallCond'].value_counts().sort_index().dropna().index,houses['OverallCond'].value_counts().sort_index().values,color='red')
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

houses_df = houses_clean[['OverallQual', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'Fireplaces',
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
houses_df['Clasificacion'] = houses_df['Clasificacion'].apply(lambda x: 'Economicas' if x < minSalesPrice + terceraParte else 'Intermedias' if x < minSalesPrice + 2 * terceraParte else 'Caras')

# Division de datos, 70% de entrenamiento y 30% de prueba
y = houses_df.pop('Clasificacion')
x = houses_df
x.pop('MasVnrArea')
x.pop('GarageYrBlt')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

# Modelo de arbol de decision
Dt_model = tree.DecisionTreeClassifier(max_depth=15, random_state=42)

x_train.fillna(0)
y_train.fillna(0)
print(x_train.describe().transpose())

Dt_model.fit(x_train, y_train)
'''
dot_data = tree.export_graphviz(Dt_model, out_file=None,
                                feature_names=x_train.columns,
                                class_names=['Economicas', 'Intermedias', 'Caras'],
                                filled=True, rounded=True,
                                special_characters=True, leaves_parallel=False)
graph = graphviz.Source(dot_data)
plt.show()
'''

tree.plot_tree(Dt_model,feature_names=houses_df.columns,
               class_names=['Economicas', 'Intermedias', 'Caras'],filled=True )
plt.show()