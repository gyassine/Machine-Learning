# k-Fold Cross Validation
# =======================
# Deux types de paramètres :
# - Ceux qui sont trouvés par le ML
# - Ceux qui sont estimé par nous (tuning parameters/hyperparamètre)
#   ex: choisir un kernel, choisir une limite ds ridge regression, etc.
#   ==> GridSearch
#
# Ici, on s'occupe du problème de variance : on fait sur le mm modèle plusieurs
# "division" du datatest
# These concepts are important to understand k-Fold Cross Validation:
# - Low Bias is when your model predictions are very close to the real values.
# - High Bias is when your model predictions are far from the real values.
# - Low Variance: when you run your model several times, the diﬀerent predictions
#  of your observation points won’t vary much.
# - High Variance: when you run your model several times, the diﬀerent predictions
#  of your observation points will vary a lot.
# https://miro.medium.com/max/3020/1*wTbewHG05BXOIwFjtilU8A.png
# What is preferred is low-bias and low-variance
#
#
# En général, on prend k = 10
#
#
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma='auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# The good method is:
# 1.Split your dataset into a training set and a test set.
# 2.Perform k-fold cross validation on the training set.
# 3.Make the ﬁnal evaluation of your selected model on the test set.
# But you can also perform k-fold Cross-Validation on the whole dataset (X, y).
#
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Mean of 10-fold CV :', accuracies.mean())
print('Std Deviation of 10-fold CV:', accuracies.std())
# This number indicates that the results are on average 0.063 difference from one
# another. This can be seen as a plus minus range (Here we have a very low variance...)
# The Standard Deviation of the model’s accuracy simply shows the variance of the model
# accuracy is 6%.
# This means the model can vary about 6%, which means that if I run my model on
# new data and get an accuracy of 86%, I know that this is like within 80-92% accuracy.
# Bias and accuracy sometimes don’t have an obvious relationship, but most of the
# time you can spot some bias in the validation or testing of your model when it
# does not preform properly on new data.

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(0)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = [ListedColormap(('red', 'green'))(i)], label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(1)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = [ListedColormap(('red', 'green'))(i)], label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()