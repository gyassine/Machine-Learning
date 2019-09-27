# -*- coding: utf-8 -*-
# SVM (Support Vector Machine ou Séparateur à Vaste Marge)
#
# But = trouver la droite optimale pour séparer deux classes (rouge et verte) linéairement séparable
#
# Méthode :
# 1- On prend les deux points rouge & vert les plus proches l'un de l'autre
# 2- on crée un droite équidistante entre ces points rouges et vert, mais aussi avec une distance maximale

# Vocabulaire:
# - la distance du point rouge à la droite plus celle du point vert à la droite est la MARGE MAXIMALE
# - les deux points choisis sont appelé les SUPPORT VECTORS. Ces deux points caractérisent la frontière 
# de prédiction. On peut enlever tous les autres points, la frontière reste la mm. Les autres points ne 
# sont pas considérés dans les équations.
# - la frontière est l'Hyperplan de marge maximale (Hyperplan = dimension n-1 où n est la dimension originale). 
# L'hyperplan positif est la droite parallèle à Hyperplan de marge maximale et qui passe par le point vert.
# L'hyperplan négatif est la droite parallèle à Hyperplan de marge maximale et qui passe par le point rouge.
#
#
# SVM est spécial, car au lieu de prendre des objets (pommes/oranges) qui se ressemblent pour chercher 
# les points communs, il va prendre l'objet (pomme) qui ressemble le plus aux objets de l'autre nature (orange)
# [cas extrême] et vice-versa, et ainsi créer à partir de ces supports une frontière.
#
# Rq : arrive à classifier très bien lorsque les points d'observations sont linéairement séparable

# ON reprendre le code de reg_logistique dans lequel on modifie juste la partie
# construction du modèle

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
classifier = SVC(kernel='linear',random_state=0)   # par défaut, ce n'est pas le linéaire qui est utilisé
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
