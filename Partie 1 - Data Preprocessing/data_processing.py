# -*- coding: utf-8 -*-
# Data Preprocessing
"""
Created on Mon Jan 21 08:33:59 2019

@author: ygang
"""

# Librairies
#-----------
import numpy as np               # librairie de calcul
import matplotlib.pyplot as plt  # librairie d'affichage de graph en 2d
import pandas as pd              # librairie de gestion des datasets


# Importation du dataset
#-----------------------

# Rq : ne pas oublier de choisir le bon "working directory"
# Rq: pour executer un ens. de ligne sélectionner (sous Sypder), faitre "Ctrl + Entrée"

dataset = pd.read_csv('Data.csv')
# Voir la variable ds l'explorateur (dbl-click)
# NOTE DE VOCABULAIRE : 
# Variable indépendante = 3 première colonnes => Matrice X (Matrice de Var. Indép. ou Matrice de features)
# Variable dépendante = colonne "purchased" => Vecteur Y

X = dataset.iloc[:,:-1].values
#iloc permet de récupérer les indices du dataset que l'on veut récupérer, 
#  - d'abord les lignes (: = toutes les lignes) 
#  - puis les indices des colonnes (:-1 = toutes les colonnes sauf la dernière)

Y = dataset.iloc[:,-1].values
# Rq : Ici on a mis -1 = la dernière colonne (différent de :-1 )



#Gestion des données manquantes 
#------------------------------
# Lorsque c'est normalement distribué, on peut remplacer par la moyenne
# Si ce n'est pas le cas, alors il vaut mieux remplacer par médiane
# Outlier = données en dont la valeur est bien supérieure ou inférieure à la moyenne du groupe. 
# Ils sont parfois issus d'erreurs de mesure et sont susceptibles de fausser les résultats obtenus.

from sklearn.preprocessing import Imputer   #c'est une classe
imputer = Imputer(missing_values ='NaN', strategy ='mean', axis = 0)
# Faire "ctrl + i" sur le mot "Imputer" pour avoir l'aide...
# Faire "np.set_printoptions(threshold=np.nan)" dans la console pour afficher X en entier dans la console

imputer.fit(X[:,1:3])
# imputer.fit permet de lier l'outil impute à la matrice.
# X[:,1:3] = (X[:,[1,2]] 

X[:,1:3] = imputer.transform(X[:,1:3])
# imputer.transform renvoie la Matrice (son morceau) avec les NaN qui ont été remplacé par la valeur moyenne
# Rq : regarder le résultat en affichant X ds l'explorateur de variables


#Gestion des variables catagoriques (qui ne sont pas numérique continue) 
#----------------------------------
# But = encoder les textes sous forme numérique
# Est qu'il s'agit de variable nominale (pas d'ordre entre les éléments) ou ordinale ?
# => pas de relation d'ordre entre france, spain, etc. => encadrer ces variables nominales par des "dummy variables" [=One-hot encoding]
# On transforme la colonne "country" en 3 colonnes france,spain et germany dans lesquelles il y a des 0 et 1
# pour éviter qu'il y a des relations d'ordre entre ces pays, ce qui provoquerait des biais dans le ML
# Pour purchased, on peut faire No = 0; Yes = 1 car il s'agit de variables dépendantes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder #c'est une classe
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
# X[:,0] correspond à la colonne country
# fit_transform permet de faire à la fois fit et trasnform
# elle permet de transformer d'abord le texte en valeur numérique simple
# Rq : regarder le résultat en affichant X ds l'explorateur de variables

oneHotEncoder = OneHotEncoder(categorical_features=[0])
# On spécifie la colonne 0 surlaquelle il faudra faire l'encodage

X= oneHotEncoder.fit_transform(X).toarray()
# On applique sur X et on spécifie qu'on veut des colonnes supplémentaires avec toarray()
# Rq : regarder le résultat en affichant X ds l'explorateur de variables

# Rq: les dummy variables sont placé au début !!!
# Si on doit retirer une dummy variable (pour éviter les pb de colinéarité) 
# alors faire X = X[:,1:] 
# on prend toutes les colonnes sauf la première


Y = labelEncoder.fit_transform(Y)
# On peut utiliser le mm labelEncoder pour X et Y...

# Diviser le dataset entre le Training set et le Test set
# -------------------------------------------------------
# Le Training set permet au ML d'apprendre les corrélations
# Le Test set (20%) permet d'éviter le sur-apprentissage
# 1- On fera un apprentissage sur le training set
# 3- On fera une prédiction sur le training set 
# 2- On regardera ensuite le nbre d'erreur que l'on divisera par le nbre d'observation, 
# ce qui nous donnera la précision du modèle sur le Traning set.
# 3- On fera la même chose sur le Test set : prédiction + calcul d'erreur (pas d'apprentissage)
# 4- Si les deux précisions sont similaires, alors un bon training (pas de sur-apprentissage), 
# si celui du traning est supérieur alors il y a eu surapprentissage.

from sklearn.model_selection import train_test_split #c'est une fonction
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)
# test_size = 0.2 permet de faire 20% pour le test
# random_state=0 est utile uniquement pour avoir les mm résultats que ds le tuto
# random_state is the seed used by the random number generator



# Feature Scaling
#----------------
# But : mettre toute les variables à la même échelle (pour éviter que l'un domine/écrase l'autre, 
# qui ne serait plus pris en compte ensuite)

from sklearn.preprocessing import StandardScaler #c'est une classe
# Il y a plusieurs méthode, on prend la standard Xstand= (x-mean(x))/standard_Deviation(x)
# il y a aussi la méthode normalisation Xnorm = (x-min(x))/(max-(x)-min(x))
# Normalization: transforme les valeurs entre 0 et 1 
# Standardization: regroupe les données selon la distribution standard 
# (aussi appelée normale d'où le piège) d'écart standard 1.



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# On va faire le fit que sur X_train mais on va l'appliquer aux deux X_train et X_test


# Construction du modèle
#----------------------


# Faire de nouvelles prédictions
#-------------------------------


# Visualiser les résultats
#-------------------------
