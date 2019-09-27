#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  R_Squared.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 23/09/2019 14:36.
#  Last modified on 23/09/2019 11:30

# R Squared = coefficient de détermination
# R² = 1 - SSres/SStot
# ou SSres = somme des carrés des distances entres les points réels et les points de la droite prédites
# et SStot = somme des carrés des distances entres les points réels et les points d'une droite y=(moyenne des y)
# Plus R² est proche de 1, plus on a une bonne regression
# R² = qualité de prédiction 
#
#
# Adjusted R Squared
# Pb lorsqu'il y a 3 variables indépendantes (ou plus) => R² ne va pas diminuer voire augmenter
# Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)
# ou p est le nombre de regresseur (nbre de VI) et n la taille d'échantillon