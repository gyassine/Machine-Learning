# -*- coding: utf-8 -*-
# Régression logistique (modèle linéaire)

# On passe de la prédiction à la probabilité
# Fonction sigmoid permet de transformer la droite de regression linéaire
# en courbe de regression logistique
#
# On utilise la Regression logistique dans pour les variables catégoriques binaires (Oui/Non)
# On peut l'utiliser pour prédire la probabilité p_hat (^p) qu'un événement/action se produise
# (à chaque fois qu'on aura une variable avec chapeau, ça veut dire que c'est la variable prédite de 
# la variable sous-jacente)
# ex: p est la probabilité (de cliquer sur Y), p_hat est la probabilité prédite par le modèle.

# Rq en générale, en rouge observation et en bleu prédiction
# 
# y = VD Réelle
# y^ = VD Prédite, obtenu en choisissant un seuil (par exemple 50%)... Si p_hat<SEUIL alors N, sinon O...
# Pour comprendre regression logistique :
# - approche probabiliste
# - obtention de la valeur prédite grâce à ces probabilités


# Librairies
#-----------
import numpy as np               
import matplotlib.pyplot as plt  
import pandas as pd              

# Importation du dataset
#-----------------------
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values    #ID et gendre n'ont pas d'impact sur l'achat.
y = dataset.iloc[:,-1].values


#Petit calcul pourcentage homme/femme
dataset_homme = dataset[dataset['Gender']=='Male'] # Permet d'extraire toutes les lignes contenant 'Male' dans la colonne gender
dataset_femme = dataset[dataset['Gender']=='Female']
percent_homme_acheteur = len(dataset_homme[dataset_homme['Purchased']==1].index)/len(dataset_homme.index)*100 # len(dataset_homme.index) permet d'obtenir le nombre de ligne de dataset_homme
percent_femme_acheteur = len(dataset_femme[dataset_femme['Purchased']==1].index)/len(dataset_femme.index)*100
print('Pourcentage d\'homme acheteur = ', percent_homme_acheteur)
print('Pourcentage de femme acheteur = ', percent_femme_acheteur)

#Gestion des données manquantes femme
#------------------------------
# Aucun

#Gestion des variables catégoriques
#----------------------------------
# Aucun et Purchased déjà sous forme 0/1

# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=0)

# Feature Scaling
#----------------
# Oui, car on n'a plus cette combinaison linéaire des VI qui est égale à la VD, 
# et donc les coefficient ne pourront pas s'adapter pour tout mettre sous la même échelle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construction du modèle
#----------------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# False positive = erreur commise lorsque la prédiction est y_hat = 1 alors que c'est faux dans la réalité 
# (erreur de 1ère espèce - moins grave, car on prédit un événement qui n'a pas lieu... ex: tremblement de terre)
# False negative = erreur commise lorsque la prédiction est y_hat = 0 alors que c'est faux dans la réalité 
# (erreur de 2ème espèce)

# Matrice de confusion
# contient tous les cas possibles avec en ligne Y (VD réelle) {0,1} et en colonne y_hat (VD prédite) {0,1}
# Ds l'ex suivant :
# - 50 prédictions que oui qui a été observé
# - 35 prédictions de non qui a été observé
# - 5  prédictions false positive (moins grave)
# - 10 prédictions false negative
#
#            y_hat
#         0    1
#    0   35    5
#  y
#    1   10   50
#
# Cette matrice permet de voir la qualité du modèle et de calculer 2 taux :
# - Accuracy rate = taux d'observation correctement prédites = (35+50)/(35+50+5+10) = 85%
# (traduit par précision, mais on n'utilisera pas car precision en anglais a un autre sens)
# - Error rate = taux d'erreurs = (10+5)/(35+50+5+10) = 15%

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
