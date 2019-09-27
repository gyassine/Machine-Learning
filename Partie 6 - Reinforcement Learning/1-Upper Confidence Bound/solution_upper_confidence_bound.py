#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  solution_upper_confidence_bound.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 27/09/2019 09:17.
#  Last modified on 27/09/2019 09:17

# Upper Confidence Bound
# ======================
# Permet de résoudre le problème du madit manchot
# Méthode :  https://www.math.univ-toulouse.fr/~agarivie/sites/default/files/JDS15_bandits.pdf
# Une autre idée féconde en apprentissage par renforcement s’applique de façon
# particulièrement convaincante pour les bandits : l’approche dite “optimiste” où l’agent fait à
# chaque instant comme si, parmi toutes les distributions de récompenses possibles
# statistiquement compatibles avec ses observations passées, il avait face à lui celle qui lui est la
# plus favorable.
# Dans le cas présent, cela revient à se fier pour chaque bras non à un estimateur de
# son efficacité moyenne, mais plutôt à une borne supérieure de confiance (upper-confidence
# bound, UCB).
#

# Image : https://i.imgur.com/n22tQ3o.png (de https://www.kaggle.com/sangwookchn/reinforcement-learning-using-scikit-learn)

# https://www.math.univ-toulouse.fr/~jlouedec/demoBandits.html

# The algorithm starts with an initial expecte value. Then, select a column and
# exploit it one time. The expected value might go down or up as new observation
# is taken. As there are more observations, the confidence bound gets smaller as
# the algorithm is more confident in the result. Then, select the column with the
# hiest upper confidence bound and exploit it once, which also makes the bound gets
# smaller and shifts the observation. Repeat these steps until the algorithm consistently
# exploits the same column, which is an indicator that this column is the optimal model.

# This is different from simple Supervised Learning, as this model starts with no data.
# When we start experimenting to collect data, reinforcement learning determines
# what to do with that existing data


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# La première ligne du csv veut dire que l'utilisateur va cliquer sur l'annonce s'il sagit du Ad1, Ad4 ou Ad9 unqiuement...

# Implementing UCB - needs to be implemented from scratch without using any package, as there is no easy library to use.
import math

N = 10000  # Users
d = 10  # Nbre d'ad
ads_selected = []
numbers_of_selections = [0] * d  # Calcul le nbre de fois qu'un ad a été choisit
sums_of_rewards = [0] * d  # Somme pour chaque AD de son reward perso
total_reward = 0  # Somme de tous les rewards

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,
                   d):  # On parcourt tous les ad, on fait le calcul du upper_bound pour choisir l'ad qui a le plus grd upper_bound
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]  # Ri(n)
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])  # Δi(n)
            upper_bound = average_reward + delta_i  # Le upper_bon du confidence intervelle
        else:
            upper_bound = 1e400  # Makes this large so that the first round gives every category a chance, = np.inf is better
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i  # On sélectionne le ad qui a le plus haut upper_bound

    ads_selected.append(ad)  # Rajout de l'ad séléectionné à la liste
    numbers_of_selections[ad] = numbers_of_selections[
                                    ad] + 1  # Rajout de l'ad dans son nlbre de sélection
    reward = dataset.values[
        n, ad]  # Calcul du reward de cette ad (normalement, c'est "est-ce qu'il a cliqué ou pas")
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward  # Calcul de la somme de reward de ce ad
    total_reward = total_reward + reward  # Calcul de la somme de tous les rewards

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
