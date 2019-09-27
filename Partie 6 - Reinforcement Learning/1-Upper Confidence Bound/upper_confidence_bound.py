#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  upper_confidence_bound.py is part of the project Introduction au Machine Learning
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

(N, d) = (dataset.shape[0], dataset.shape[1])  # get row and col values
(num_of_selections, sum_of_rewards) = ([0] * d, [0] * d)  # initialize
(avg_reward, ucb) = ([0] * d, [0] * d)  # initialize
ad_selected = []  # initialize
ad = 0  # initialize

# Creating the lists of accumulated rewards
actual_accumulated_reward = []
max_accumulated_reward = []

for n in range(0, N):  # read thru every row
    if n < d:  # have we tried all ads atlest once
        ad = n  # try that ad
    else:
        ad = ucb.index(max(ucb))  # else get ad with max upper confidence bound
    ad_selected.append(ad)  # add the ad selected
    sum_of_rewards[ad] += dataset.values[n, ad]  # update the reward for selected ad
    num_of_selections[ad] += 1  # update number of selection for selected ad
    avg_reward[ad] = sum_of_rewards[ad] / num_of_selections[ad]  # update avg reward for selected ad
    delta = math.sqrt(3 / 2 * math.log(n + 1) / num_of_selections[ad])  # delta for selected ad
    ucb[ad] = avg_reward[ad] + delta  # update upper confidence bound for selected ad
    # Filling the two lists right after the "for" loops
    actual_accumulated_reward.append(dataset.values[n, ad])
    max_accumulated_reward.append(max(dataset.values[n, :]))
total_rewards = sum(sum_of_rewards)  # calcualte total rewards earned

# Visualising the results
plt.hist(ad_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Visualising the results
X = range(0, N)
Y1 = [0] * N
Y2 = [0] * N
Y3 = [0] * N
Y1[0] = actual_accumulated_reward[0]
Y2[0] = max_accumulated_reward[0]
Y3[0] = max_accumulated_reward[0] - actual_accumulated_reward[0]
for i in range(0, 9999):
    Y1[i + 1] = Y1[i] + actual_accumulated_reward[i + 1]
    Y2[i + 1] = Y2[i] + max_accumulated_reward[i + 1]
    Y3[i + 1] = Y3[i] + max_accumulated_reward[i + 1] - actual_accumulated_reward[i + 1]
plt.plot(X, Y1, label='Realised Reward')
plt.plot(X, Y2, label='Best Action (Max Expected Reward)')
plt.plot(X, Y3, label='Regret Curve (Best - Realised)')
plt.xlabel('Rounds')
plt.ylabel('Reward Count')
plt.title(' Regret Curve')
plt.legend()
plt.show()
