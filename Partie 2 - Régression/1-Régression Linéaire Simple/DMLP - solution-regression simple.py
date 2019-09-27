# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:37:48 2018

"""
#importation librairy
import numpy as np
import matplotlib.pyplot as mplt
import pandas as pd
import seaborn as sbrn
import statsmodels as stat

#creation dataset
 dtst =  pd.read_csv('lycee.csv')
 
 X = dtst.iloc[:,:-1].values
 y = dtst.iloc[:,-1].values
 
 # Construction de l'echantillon de trainning et de l'echantillon de test
 from sklearn.model_selection import train_test_split
 X_train,X_test,y_train,y_test =  train_test_split(X,y, test_size = 0.2, random_state = 0)
 
 #construction du model  de regression simple
 from sklearn.linear_model import LinearRegression
 regresseur = LinearRegression()
 regresseur.fit(X_train,y_train)
 
 #etablir une prediction
 y_prediction =  regresseur.predict(X_test)
 
 # prediction sur nouvel data
 regresseur.predict(180)
 
 #Visualisation
# Visualisation des résultats de l'ensemble d'entraînement
 mplt.scatter(X_train, y_train, color =  'red')
 mplt.plot(X_train,regresseur.predict (X_train), color =  'green')
 mplt.title( ' rendement note sur minute de revision ')
 mplt.xlabel( ' minutes passés à reviser ' )
 mplt.ylabel( ' note en % ' )
 mplt.show()
 
 
 mplt.scatter(X_test, y_test, color =  'yellow')
 mplt.plot(X_test,regresseur.predict (X_test), color =  'black')
 mplt.title( ' rendement note sur minute de revision ')
 mplt.xlabel( ' minutes passés à reviser ' )
 mplt.ylabel( ' note en % ' )
 mplt.show() 

 
