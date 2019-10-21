# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:46:01 2019

@author: YassineGANGAT
"""

# !pip install tf-nightly-2.0-preview

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

print(tf.__version__)

# Génération d'une série de nbre {0, 1, 2, 3,... }
dataset = tf.data.Dataset.range(10)
#for val in dataset:
#   print(val.numpy())

window_size=5

# Création de fenetre de 5 nbre, avec un shift de 1
# Drop_remainder permet de mettre de coté les windows trop petites)
# { {0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}...}
dataset = dataset.window(window_size, shift=1, drop_remainder=True)
#for window_dataset in dataset:
#  for val in window_dataset:
#    print(val.numpy(), end=" ")
#  print()

# "Numpérisation" du dataset
#{[0 1 2 3 4],...}
dataset = dataset.flat_map(lambda window: window.batch(window_size))
#for window in dataset:
#  print(window.numpy())


# On split chaque feneter en supprimant le dernier, et en rajoutant un autre avec le dernier
# cad [0 1 2 3 4] devient [0 1 2 3] [4]
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
#for x,y in dataset:
#  print(x.numpy(), y.numpy())

# Mélange (pour éviter le Sequence Bias)
dataset = dataset.shuffle(buffer_size=10)

# Création de X batch de Y windows...
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())