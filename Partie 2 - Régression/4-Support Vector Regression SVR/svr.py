# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:21:46 2019

@author: YassineGANGAT
"""

#  Copyright (c) 2019 SunnyShark - All Rights Reserved
#  svr.py is part of the project Introduction au Machine Learning
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Created by Yassine Gangat  <yg@sunnyshark.com> on 23/09/2019 14:36.
#  Last modified on 23/09/2019 14:36

# Support vector Regression
# =========================
# SVR for both linear and not linear regressions.
# In simple regression we try to minimise the error rate. While in SVR we try to fit the error within a certain threshold.
# https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff
# https://www.youtube.com/watch?v=Y6RRHw9uN9o
# Avantages : il exclut automatiquement les outliers !


# Terminologie
# ------------
# Kernel: The function used to map a lower dimensional data into a higher dimensional data.
# Hyper Plane: In SVM this is basically the separation line between the data classes. Although in SVR we are going to define it as the line that will will help us predict the continuous value or target value
# Boundary line: In SVM there are two lines other than Hyper Plane which creates a margin . The support vectors can be on the Boundary lines or outside it. This boundary line separates the two classes. In SVR the concept is same.
# Support vectors: This are the data points which are closest to the boundary. The distance of the points is minimum or least.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting SVR to the dataset
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform([[6.5]]))  # Ne pas oublier de scaler avant de prédire
y_pred = sc_y.inverse_transform(y_pred)  # et de "descaler" pour obtenir le nombre original ;)

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X),
                   0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
