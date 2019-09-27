#IMPORT DES LIBRAIRIES
import numpy as np
import matplotlib.pyplot as mplt
import pandas as pd
import seaborn as sbrn
import statsmodels as stat

# import le dataset
dtst = pd.read_csv('Social.csv')
X = dtst.iloc[:, [2, 3]].values
y = dtst.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#Mise à l'echelle
from sklearn.preprocessing import StandardScaler
stdSc = StandardScaler()
X_train = stdSc.fit_transform(X_train)
X_test = stdSc.transform(X_test)

#Construction du model d'arbre de décision
from sklearn.tree import DecisionTreeClassifier
classification  = DecisionTreeClassifier(criterion='entropy',random_state=0)
classification.fit(X_train,y_train)

# mise en place de prediction 
y_prediction = classification.predict(X_test)

#matrice de confusion
from sklearn.metrics import confusion_matrix
mat_conf = confusion_matrix(y_test, y_prediction)


