# Regression Linéaire Multiple

# Importer les librairies
import numpy as np
import matplotlib.pyplot as mplt
import pandas as pd

dtst = pd.read_csv('EU_I_PIB.csv')
X = dtst.iloc[:, -4:].values
y = dtst.iloc[:, -5].values

#Gerer lA dummy

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEnc_X =  LabelEncoder()
X[:,0]= labelEnc_X.fit_transform(X[:,0])
OnehotEnc_X = OneHotEncoder(categorical_features= [0])
X= OnehotEnc_X.fit_transform(X).toarray()

# division de l'echantillon
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)

#Construire notre modele de regression Multiple
 from sklearn.linear_model import LinearRegression
 regresseur = LinearRegression()
 regresseur.fit(X_train,y_train)
 
 #Faire de nouvelles prediction
y_prediction = regresseur.predict(X_test)

# 
regresseur.predict(np.array([[0,1,587923,48562,4848551]]))
