# -*- coding: utf-8 -*-
# Paradoxe de l'accuracy
# On ne doit pas évaluer le modèle uniquement sur accuracy
# Ex :
# Scénario 1 Accuracy = 98%
#            y_hat
#         0      1
#    0   970    15 (FP)
#  y
#    1   5(FN)  10
# 
# Scénario 2 (je mets tout à 0) accuracy = 98,5%
#            y_hat
#         0    1
#    0   985    0
#  y
#    1   15     0
#
#
# Solution : Utiliser CAP (Cumulated Accuracy Profile)
# 
#
# Rq : !!!!    CAP != ROC    !!!!
# ROC = Receiver Operating Characteristic
#
# Comment mesurer la qualité :
# 1- calculer des aires
# AR = Ar/Ap  
# - où Ar est l'aire entre la courbe de CAP de notre regression et celle du modèle random
# - et Ap l'aire entre la courbe de CAP du modèle parfait et celle du modèle random
#
# 2- prendre 50% en abscisse on obtient X
# - X < 60%        Modèle nulle
# - 60% < X < 70%  Modèle Faible
# - 70% < X < 80%  Modèle Bon
# - 80% < X < 90%  Modèle Très Bon
# - 90% < X < 100% Modèle TROP BON   <= attention au overfitting/surapprentissage (par exemple si X = 100% voire 99%)
# Il y a un facteur déterminant dans les VI qui fait que le modèle est biaisé !!!
 