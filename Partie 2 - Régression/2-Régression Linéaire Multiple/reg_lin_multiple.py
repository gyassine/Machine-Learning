# -*- coding: utf-8 -*-

#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  reg_lin_multiple.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 23/09/2019 14:36.
#  Last modified on 23/09/2019 14:36

# Régression linéaire multiple
"""
Created on Mon Jan 21 08:33:59 2019

@author: ygang
"""

#STEP by STEP
# Pourquoi se débarrasser de certaines variables ?
# -> garder ce qui a le plus d'impact

# 5 méthodes de constructions de modèles :
# 1- All-in
# 2- Backward Elimination
# 3- Forward Selection
# 4- Bidirectionnal Elimination
# 5- Score Comparison
# Stepwise Regression = 2,3 et 4, ou uniquement 4

# 1- All-in
# Qd ?
# - Qd on sait ce qu'il faut mettre
# - Qd on n'a pas le choix,
# - Qd on prepare la Backward Elimination

# 2- Backward Elimination (+ rapide et + efficace)
# - Step 1 : Choisir un seuil de significativité SL pour rester ds le modèle (5% cad 0.05)
# - Step 2 : Remplir le modèle avec tous les prédicteurs possibles
# - Step 3 : Considérer le prédicteur qui a la plus grande p-value
# (tant p-value>SL sinon on stop)
# - Step 4 : Enlever ce prédicteur
# - Step 5 : Ajuster le modèle sans cette variable (aller au step 3)

# NOTE: Hypothesis testing is like a court case. The null hypothesis is trying to prove that the
# independent variables are not related to the dependent variable, or the relation is by chance.
# Our goal is to prove the null hypothesis wrong so we can reject it. If we obtain variables
# with a p-value less than 0.05, then we know the variables relation is not due to chance and
# can reject the null hypothesis.
# - The lower the p-value is, the most statistically significant the indepandant varibale is,
# the more impact/effect the independant variable will have on the dependant variable
# - Threshold : 5% (0.05)


# 3- Forward Selection
# - Step 1 : Choisir un seuil de significativité SL pour rester ds le modèle (5% cad 0.05)
# - Step 2 : Réaliser les modèles de régression linéaires simples pour chaque variable.
# Sélectionner celui qui a la plus petite p-value
# - Step 3 : Garder cette variable, ajuster tous les modèles possibles avec ce prédicteur en plus.
# - Step 4 : Considérer le prédicteur qui a la plus petite p-value, si p < SL alors aller au step 3
# sinon stop (et prendre le dernier modèle avec le dernier p<SL )


# 4- Bidirectionnal Elimination
# - Step 1 : Choisir 2 seuil de significativité SLenter et SLstay pour rester ds le modèle (5% cad 0.05)
# - Step 2 : Effectuer le "next step" de forward selection p<SLenter pour faire entrer des nouvelles variables
# (cad faire entrer une nouvelle variable dans le modèle, régression linéaire avec p-value, si <SLenter alors
# le rajouter au modèle)
# - Step 3 : Faire tous les step de backward elimination avec p < PLstay pour faire sortir les anciennes variables.
# - Step 4 : Aller step 2 tant que on fait rentrer des nouvelles variables et sortir des anciennes variables.


# 5- Score Comparison (Meilleur méthode mais plus consommatrice - All possible)
# - Step 1 : Choisir un critère de qualité d'ajustement (ex: critère d'Akaike, R², etc.)
# - Step 2 : Construire tous les modèles de regression possibles (simple, double, triple, etc. n-uple)
# avec les n variables independantes. => (2^n)-1 modèles
# - Step 3 : Choisir celui qui a le meilleur critère.
# ex: avec 10 colonnes (cad 10 VI) on a 1023 modèles


# RQ: regression lorsque la VD est numérique (continue) et classification lorsque la variable est catégorique
# ex: définir le profit de plusieurs start-up

# Librairies
#-----------
import numpy as np               # librairie de calcul
import matplotlib.pyplot as plt  # librairie d'affichage de graph en 2d
import pandas as pd              # librairie de gestion des datasets



# Importation du dataset
#-----------------------
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Gestion des données manquantes
#------------------------------
# pas de données manquantes

#Gestion des variables catégoriques (qui ne sont pas numérique continue)
#----------------------------------
# OLD VERSION
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder #c'est une classe
# labelEncoder = LabelEncoder()
# X[:,3] = labelEncoder.fit_transform(X[:,3])
# oneHotEncoder = OneHotEncoder(categorical_features=[3])
# X= oneHotEncoder.fit_transform(X).toarray()

# New Version
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columntransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(columntransformer.fit_transform(X), dtype=np.float)



# les dummy variables sont placé au début !!!
# On doit retirer une dummy variables
X = X[:,1:]
# on prend toutes les colonnes sauf la première
# Rq: ds la regression linéaire de python skleanr, il prend en compte automatiquement ce problème
# et on n'a pas besoin de le faire forcément...


# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
from sklearn.model_selection import train_test_split #c'est une fonction
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

# Feature Scaling
#----------------
# pas besoin ds regression linéaire

# Construction du modèle regression linéaire
#------------------------------------------
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
# Apprentissage sur le Train set

################### Ajout perso pour test SL
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
result = sm.OLS(y_train,X_train).fit()
print(result.summary())
######################################

# Faire des prédictions
#----------------------
y_pred = regressor.predict(X_test)

# Visulaiser qualité de prédictions
# ----------------------
plt.plot(y_test, y_pred, "o", color='r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')

regressor.predict(np.array([[1,0,130000,140000,300000]]))
regressor.predict([[1,0,130000,140000,300000]])
# prédiction sur le Test set et sur un employé qui a 15 ans d'ancienneté

# AJOUT DEPUIS VERSION ANGLAISE
# ------------------------------
# Building optimal model using backwar elimination
# -------------------------------------------------
import statsmodels.api as sm

# Pour créer une colonne de 1,1,1,1,1... que l'on associera à la constance b0
# y = b0 + b1*x1 +b2*x2 + etc. deviendra
# y = b0*x0 + b1*x1 +b2*x2 + etc.
# X = np.append(values = X, arr = np.ones((50,1)).astype(int), axis=1 )
X = sm.add_constant(X)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())


# Backward Elimination with p-values only
def backwardElimination_1(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


# Backward Elimination with p-values and Adjusted R Squared
def backwardElimination_2(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination_2(X_opt, SL)
