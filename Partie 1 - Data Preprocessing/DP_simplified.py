# -*- coding: utf-8 -*-
# Data Preprocessing
"""
Created on Mon Jan 21 08:33:59 2019

@author: ygang
"""

# Librairies
#-----------
import numpy as np               
import matplotlib.pyplot as plt  
import pandas as pd              

# Importation du dataset
#-----------------------
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#Gestion des données manquantes 
#------------------------------
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values ='NaN', strategy ='mean', axis = 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Gestion des variables catagoriques
#----------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X= oneHotEncoder.fit_transform(X).toarray()
y = labelEncoder.fit_transform(y)
# Si on doit retirer une dummy variable alors faire X = X[:,1:] 



# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

# Feature Scaling
#----------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construction du modèle
#----------------------


# Faire de nouvelles prédictions
#-------------------------------


# Visualiser les résultats
#-------------------------
plt.figure() # permet de créer une nouvelle fenêtre avant de plotter
