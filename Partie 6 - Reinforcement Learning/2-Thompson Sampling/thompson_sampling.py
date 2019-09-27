#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  thompson_sampling.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 27/09/2019 09:17.
#  Last modified on 27/09/2019 09:17

# Thompson Sampling
# =================
# First of all, in Thomson Sampling, we are trying to guess the expected value
# for each distribution (which would be a bandit). Therefore, this is a probabilistic
# algorithm. According to initial distributions created, we then generate three
# random points from each distribution (creating a bandit configuration, which
# is sampling)
#
# Then, pick the best point which is the point that is on the far right side
# of the plot, as it has the highest return. Calculate it on the existing
# distribution and adjust the expected value and refine the distribution
# according to this.
#
#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random

N = 10000  # Users
d = 10  # Nbre d'ad
ads_selected = []
numbers_of_rewards_1 = [0] * d  # number of 1 rewards for each ad
numbers_of_rewards_0 = [0] * d  # number of 0 rewards for each ad
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0  # maximum random draw
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[
            i] + 1)  # On obtient un random suivant le Thompson Sampling betavariate
        if random_beta > max_random:  # Si le random_beta est supérieur aux autres, on le choisit
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]

    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
