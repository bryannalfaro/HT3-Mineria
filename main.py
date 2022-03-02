#Universidad del Valle de Guatemala
#Mineria de Datos
#HT3 Arboles
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera


from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import pyclustertend
import random
import sklearn.mixture as mixture
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.cm as cm

houses = pd.read_csv('train.csv', encoding='latin1', engine='python')

print(houses.head())