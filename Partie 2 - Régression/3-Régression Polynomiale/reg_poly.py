# -*- coding: utf-8 -*-

#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  reg_poly.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 23/09/2019 14:36.
#  Last modified on 23/09/2019 14:36

# Régression polynomiale
# ======================

# Une seule vairable indépendante, mais avec des puissances différentes
# Adapté aux problèmes non linéaire



# Librairies
#-----------
import numpy as np               # librairie de calcul
import matplotlib.pyplot as plt  # librairie d'affichage de graph en 2d
import pandas as pd              # librairie de gestion des datasets

# Importation du dataset
#-----------------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
# on saute la première colonne (cad la colonne 0)
# Rq: (ne pas mettre 1, sinon ça crée un vecteur plutôt qu'une matrice)

y = dataset.iloc[:,-1].values

# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
# pas de training set de prévu
# Raison : peu de data / real life prediction

# Construction du modèle regression polynomial
# ---------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
# choisit le niveau polynomiale (z)

X_poly = poly_reg.fit_transform(X)
# Crée z nouvelle colonne avec les puissances de la 1ère
#la 1ère colonne correspond à la constante. On peut la garder ou la supprimer

regressor = LinearRegression()
regressor.fit(X_poly,y)


# Visualiser les résultats
#----------------------
plt.figure(0)  # Permet de créer des plot séparé...

plt.scatter(X, y,  color='red')
# scatter permet de placer les points sur un graphe (abs puis ordonné)

plt.plot(X, regressor.predict(X_poly), color = 'blue')
# plot permet de placer la droite de régression linéaire

plt.title('Salaire Vs Niveau')
plt.xlabel('Niveau')
plt.ylabel('Salaire')

plt.show()
# pour afficher le graph
# Faire Préférence > Console IPython > Graphique > Sortie : Automatique pour affichage séparé

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
plt.figure(1)  # Permet de créer des plot séparé...
X_grid = np.arange(min(X), max(X), 0.1)  # crée un "range" de 1 à 10, par 0.1
X_grid = X_grid.reshape((len(X_grid), 1))  # transofrmer en matrice
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
