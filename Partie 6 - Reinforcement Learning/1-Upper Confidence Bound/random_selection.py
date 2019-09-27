#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  random_selection.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 27/09/2019 09:17.
#  Last modified on 27/09/2019 09:17

# Random Selection
# ================
# Choix purement aléatoire d'une pub...
#
#
#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random

N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)  # On choisit un nbre alatoire en 0 et 9
    ads_selected.append(ad)  # On l'enregistre
    reward = dataset.values[
        n, ad]  # On vérifie si ds la réalité l'utilisateur aurait cliqué (1) ou pas (0)
    total_reward = total_reward + reward  # On rajoute le reward (0 ou 1)

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
