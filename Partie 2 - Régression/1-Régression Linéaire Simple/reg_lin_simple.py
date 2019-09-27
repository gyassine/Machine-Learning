# -*- coding: utf-8 -*-
# Regression linéaire simple
"""
Created on Mon Jan 21 08:33:59 2019

@author: ygang
"""

# Librairies
#-----------
import numpy as np               # librairie de calcul
import matplotlib.pyplot as plt  # librairie d'affichage de graph en 2d
import pandas as pd              # librairie de gestion des datasets

# Importation du dataset
#-----------------------
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
from sklearn.model_selection import train_test_split #c'est une fonction
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1.0/3, random_state=0)

# Remarque 1/3 renvoie 0 (division d'entier) et 1.0/3 renvoie 0.33333


# Feature Scaling
#----------------
# pas besoin pour la regression linéaire !!! (à cause de linéaire)


# Contruction du modèle regression linéaire simple (une seule variable indépendante)
#-------------------------------------------------
# méthode des moindres carrés
# Remarque : Attention aux 5 conditions de la régression linéaire
# 1- Exogénéité
# 2- Homoscédasticité
# 3- Erreurs indépendantes
# 4- Normalités de erreurs
# 5- Non colinéarité des variables indépendantes
#   => Tjrs enlever une dummy variable par colonnes pour la RL
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
# Apprentissage sur le Train set


################### Ajout pour test SL
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
result = sm.OLS(y_train,X_train).fit()
result.summary()
######################################

# Faire des prédictions
#----------------------
y_pred = regressor.predict(X_test)
regressor.predict([[15]])
# prédiction sur le Test set et sur un employé qui a 15 ans d'ancienneté

# Visualiser les résultats
#-------------------------
plt.scatter(X_test, y_test,  color='red')
plt.scatter(X_test, y_pred, color='blue')
# scatter permet de placer les points sur un graphe (abs puis ordonné)

plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plot permet de placer la droite de régression linéaire

plt.title('Salaire Vs Experience')
plt.xlabel('Xp')
plt.ylabel('Salaire')

plt.show()
# pour afficher le graph
# Faire Préférence > Console IPython > Graphique > Sortie : Automatique pour affichage séparé