# -*- coding: utf-8 -*-
# Kernel SVM
#
# But = trouver la droite optimale pour séparer deux classes (rouge et verte) qui ne sont pas 
# linéairement séparable ("le plus puissant")
#
# Idée :
# - mapper vers un espace de plus grande dimension pour rendre les classes séparables (n+1)
# - faire un modèle un SVM normal linéaire (et donc un hyperplan)
# - Re-projection vers un espace à dimension originale (n)
#
# Pb : Très compute intensive (demande bcp de calculs)
# solution : le Kernel
# 1- Fonction Gaussian RBF Kernel (sorte de montagne 3D où l'altitude dépend de la proximité par rapport au
# centre de la montagne/landmark )
# La variable sigma permet de bien choisir la grandeur de la base "de la montagne"
# 
# 2- Sigmoid Kernel
# Tangente hyperbolique
# Avantage : Kernel directionnel... On est ds la classe positive à droite
#
# 3- Polynomial Kernel
# Avantage : classe positive en plrs points
#
# Ces fonction permet de faire moins de calcul et de mieux classer...
#

# Librairies
#-----------
import numpy as np               
import matplotlib.pyplot as plt  
import pandas as pd              

# Importation du dataset
#-----------------------
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=0)

# Feature Scaling
#----------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construction du modèle
#----------------------
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

# Faire de nouvelles prédictions
#-------------------------------
y_pred = classifier.predict(X_test)

# Matrice de confusion
#--------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = (cm[0,0] + cm[1,1])/len(y_test)
error = (cm[1,0] + cm[0,1])/len(y_test)

# Visualiser les résultats
#-------------------------
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train                 # Choix du set à afficher (train ou test)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# On construit une GRID que l'on va ensuite remplir et colorer selon sa position par rapport au résultat

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
# Prediction boundaries = frontière de prédiction = droite qui sépare les deux régions de prédictions 
# (normal car c'est une méthode linéaire)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
# On place les points sur le GRID les points rouge et vert

plt.title('Résultats du Training set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()
plt.show()
