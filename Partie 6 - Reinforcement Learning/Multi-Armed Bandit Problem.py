#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  Multi-Armed Bandit Problem.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 27/09/2019 09:17.
#  Last modified on 27/09/2019 09:17

# En mathématiques, plus précisément en théorie des probabilités, un problème du
# bandit manchot (aussi appelé problème du bandit à K bras ou le problème du
# bandit à N bras) est un problème mathématique qui se formule de manière imagée
# de la façon suivante : un utilisateur (un agent), face à des machines à sous, doit
# décider quelles machines jouer. Chaque machine donne une récompense moyenne que
# l'utilisateur ne connait pas a priori. L'objectif est de maximiser le gain cumulé
# de l'utilisateur. Typiquement, une politique de l'utilisateur oscille entre exploitation
# (utiliser celle dont il a appris qu'elle récompense beaucoup) et exploration
# (tester une autre machine pour espérer gagner plus). Un problème de bandit
# manchot peut être vu comme un processus de décision markovien avec un seul état.
#
# Trying to find the optimal option among many possible options as quickly as possible,
# with minimal exploration of other options which have high disadvantage in money
# (resources) and time. For example, if an Advertising company has 5 advertisement
# options, they need to find the best one to publish. In order to do this, they might
# to AB testing. But if they do too much AB testing, then it is just as same as utilizing
# all 5 options which is not ideal. Therefore, through reinforcement learning, the
# company needs to find the optimal option quickly.


# Rq: Application de nos jours : tester quel pub est la meilleur !!!


# Another potential application of Multi-armed bandits (MAB) can be the online
# testing of algorithms.
#
# For example, let's suppose you are running an e-commerce website and you have
# at your disposal several Machine Learning algorithms to provide recommendations
# to users (of whatever the website is selling), but you don't know which
# algorithm leads to the best recommendations.
#
# You could consider your problem as a MAB problem and define each Machine Learning
# algorithm as an "arm": at each round when one user requests a recommendation,
# one arm (i.e. one of the algorithms) will be selected to make the recommendations,
# and you will receive a reward. In this case, you could define your reward in various
# ways, a simple example is "1" if the user clicks/buys an item and "0" otherwise.
#
# Eventually your bandit algorithm will converge and end up always choosing
# the algorithm which is the most efficient at providing recommendations.
# This is a good way to find the most suitable model in an online problem.

# Another example coming to my mind is finding the best clinical treatment for
# patients: each possible treatment could be considered as an "arm", and a
# simple way to define the reward would be a number between 0 (the treatment has
# no effect at all) and 1 (the patient is cured perfectly).
# In this case, the goal is to find as quickly as possible the best treatment
# while minimizing the cumulative regret (which is equivalent to say you
# want to avoid as much as possible selecting "bad" or even sub-optimal
# treatments during the process).
