# -*- coding: utf-8 -*-
# Data Preprocessing


# Librairies
#-----------
import numpy as np               # librairie de calcul
import matplotlib.pyplot as plt  # librairie d'affichage de graph en 2d
import pandas as pd              # librairie de gestion des datasets
import seaborn as sbrn
import statsmodels as stat

# Importation du dataset
#-----------------------
dataset = pd.read_csv('credit_immo.csv')

X = dataset.iloc[:,-9:-1].values
# Variables explicatives (VIndépendante)
# Ici on va ignore aussi la colonne ID_NOM
# Ne pas oublier que la borne supérieur est exclue !!!

y = dataset.iloc[:,-1].values
# Variable expliquée  (VDépendante)

# Nettoyage de données  (= Gestion des données manquantes )
# --------------------
from sklearn.preprocessing import Imputer   #c'est une classe
imputer = Imputer(missing_values ='NaN', strategy ='mean', axis = 0)
imputer.fit(X[:,0:1])
imputer.fit(X[:,7:8])
X[:,0:1] = imputer.transform(X[:,0:1])
X[:,7:8] = imputer.transform(X[:,7:8])

#Gestion des variables catagoriques (qui ne sont pas numérique continue) 
#----------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #c'est une classe
labelEncoder = LabelEncoder()
X[:,2] = labelEncoder.fit_transform(X[:,2])
X[:,5] = labelEncoder.fit_transform(X[:,5])
oneHotEncoder = OneHotEncoder(categorical_features=[2])
oneHotEncoder = OneHotEncoder(categorical_features=[5])
X= oneHotEncoder.fit_transform(X).toarray()

y = labelEncoder.fit_transform(y)


# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
from sklearn.model_selection  import train_test_split #c'est une fonction
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)


# Feature Scaling
#----------------
from sklearn.preprocessing import normalize 
X_train = normalize(X_train)
X_test = normalize(X_test)
