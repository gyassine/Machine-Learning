# -*- coding: utf-8 -*-
# K-means (machine learning non supervisé)
# 
# Méthode:
# -------
#
# Step 1: Choisir le nbre K de clusters [il existe un moyen de les choisir judicieusement ]
# Step 2: Sélectionner au hasard K points (appelé centroids), pas nécessairement du dataset [idem]
# Step 3: Assigner chaque point au centroid le plus proche (distance euclidienne) => cela forme K clusters
# Step 4: Calculer et placer le nouveau centroid de chaque cluster [moyenne des coordonnées de chaque point]
# Step 5: On refait le step 3 jusqu'à ce qu'on arrête d'assigner des points
# 
# 
# Comment bien choisir les centroid ?
# En effet, une mauvaise selection peut aboutir à de mauvais cluster
# solution : K-means++ contient les algorithmes qui le font...
# 
# 
# Comment bien choisir le nbre de clusters ? 
# Solutions : 
# 1- WCSS = somme des carré intercluster, permet de mesure la pertinence du nbre de clusters
# (within cluster sum of squares)
# Méthode elbow (coude)... On fait le calcul WCSS pour chaque nbre de cluster et on regarde à partir de quel moment
# la diminution est faible (cad que la pente de la courbe des WCSS est moins abrupte)
# 
# 2- contrainte : par ex, il s'agit des jours de la semaine, donc <= 7
# 
# 3- objectifs : par ex, trouver deux groupes, donc = 2 
# 
# # Librairies
#-----------
import numpy as np               
import matplotlib.pyplot as plt  
import pandas as pd              

# Importation du dataset
#-----------------------
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
# On utilisera que les deux dernières colonnes : 
# 1- pour bien voir les clusters 
# 2- caravec age c'est moins évident d'avoir des clusters

# pas de VD... On veut voir si on peut identifier des groupes de clients et 
# créer donc un VD : appartenance à tel cluster de clients


#Gestion des données manquantes 
#------------------------------
# pas de val. manquantes

#Gestion des variables catagoriques
#----------------------------------
# pas de var catégoriques

# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
# pour le clustering, on a pas besoin... on a pas de VD (de y), donc pas de prédiction à évaluer

# Feature Scaling
#----------------
# mm échelle, pas besoin... de plus pour le graph c'est plu simple


# Utiliser la méthode elbow pour trouver le nbre optimal de clusters
#--------------------------------------------------------------------
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11): # la borne supérieur du range est tjrs exclue
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)               # inertia_ est le WCSS
# on a créé une boucle pour calculer tout les clustering possibles entre 1 et 10
# on calcul le wcss que l'on stocke
  
plt.plot(range(1,11),wcss)
plt.xlabel ('Nbre de cluster')
plt.ylabel ('WCSS')
plt.title ('Méthode elbow')
plt.show()                                      # le graph nous montre 5   
plt.close()                                     # le close permet de fermer la fenêtre et de créer une autre propre

    
# Construction du modèle
#----------------------
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init='k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)   
# permet de fitter et en plus de retourner le cluster pour chaque point
# la valeur à l'index i de y_kmeans correspond au cluster auquel appartient le client à l'index i de X



# Visualiser les résultats
#-------------------------
plt.scatter(X[y_kmeans ==1, 0],X[y_kmeans ==1, 1], c = 'red', label = 'Cluster 1')   
# abscisse et ordonnée 
# y_kmeans ==1 permet sélectionner les indexs donc le kmeans = 1, donc les éléments du cluster 1...
plt.scatter(X[y_kmeans ==2, 0],X[y_kmeans ==2, 1], c = 'blue', label = 'Cluster 2')   
plt.scatter(X[y_kmeans ==3, 0],X[y_kmeans ==3, 1], c = 'green', label = 'Cluster 3')   
plt.scatter(X[y_kmeans ==4, 0],X[y_kmeans ==4, 1], c = 'cyan', label = 'Cluster 4')   
plt.scatter(X[y_kmeans ==0, 0],X[y_kmeans ==0, 1], c = 'magenta', label = 'Cluster 5')   
plt.title('Clusters de clients')
plt.xlabel('Salaire annuel')
plt.ylabel('Spending Score')
plt.legend()

# On a obtenu notre variable dépendante et que nous pouvons donc construire un modèle de prédiction
# On peut utiliser cette variable dépendante pour prédire les prochains clients avec les modèles précédents

