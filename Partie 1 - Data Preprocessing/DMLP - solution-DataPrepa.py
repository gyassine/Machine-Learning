# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:16:12 2018
"""
#importation des packages
import statsmodels as stat
import seaborn as sbrn
import pandas as pds
import matplotlib.pyplot as mplt
import numpy as np 
#importation de jeu de données
dtst = pds.read_csv('credit_immo.csv')

X = dtst.iloc[:,-9:-1].values
Y = dtst.iloc[:,-1].values

#data cleaning
from sklearn.preprocessing import Imputer
imptr = Imputer(missing_values= 'NaN',strategy = 'mean',axis = 0)
imptr.fit(X[:,0:1])
imptr.fit(X[:,7:8])
X[:,0:1] = imptr.transform(X[:,0:1])
X[:,7:8] = imptr.transform(X[:,7:8])

#Données catégoriques

## Codage de la variable indépendante
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labEncr_X = LabelEncoder()
X[:,2] = labEncr_X.fit_transform(X[:,2])
X[:,5] = labEncr_X.fit_transform(X[:,5])
onehotEncr = OneHotEncoder(categorical_features=[2])
onehotEncr = OneHotEncoder(categorical_features=[5])
X = onehotEncr.fit_transform(X).toarray()

#VARIABLE DEP
labEncr_Y = LabelEncoder()
Y = labEncr_Y.fit_transform(Y)

# Fractionner l'ensemble de données en train et ensemble d'essai
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)





