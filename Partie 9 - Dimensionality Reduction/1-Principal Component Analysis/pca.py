# PCA ()
# ===
# https://www.youtube.com/watch?v=FgakZw6K1QQ
# Technique de Feature Extraction (permet de réduire le nbre de VI)
# Permet de définir quelles sont les VI qui explique le mieux le dataset, sans
# prendre en compte le VD (modèle non supervisé)
#
# I would rather recommend using Kernel PCA than PCA on real world datasets,
# since most of them are not linearly separable.



def scree_plot(pca, list_columns = None) :
    nb_features= len(pca.explained_variance_ratio_)
    columns=[]
    for i in range(1,nb_features+1) :
        columns.append(f'PC{i}')
    percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
    plt.figure(0)
    plt.bar(x= range(1,nb_features+1), height=percent_variance, tick_label=columns)
    plt.ylabel('Percentate of Variance Explained')
    plt.xlabel('Principal Component')
    plt.title('PCA Scree Plot')
    plt.show()
    if list_columns == None :
        print( pd.DataFrame(pca.components_,index = columns))
    else :
        print( pd.DataFrame(pca.components_,index = columns, columns = list_columns))

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


#Scree plot and PCi composition (PC1 is 0.12960 * Firts column [Alcohol])
scree_plot(pca, list_columns = list(dataset)[:-1])    # -1 car la denrière colonnes du dataset est la VD.



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# En diagonal les bonnes prédictions
# Prédiction = ligne, Réel = Colonne

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(1)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Remarque :
# For Multivariate Linear Regression we do not have such concept as "importance
# of variables". We can talk about what predictor is contributing to outcome
# judging by their p-values, and since a predictor with a lowest p-value can
# have a very small coefficient, then it's importance is questionable.
#The "variable importance" comes from other methods, like Decision Trees.
#
# First, you can use PCA to reduce dimension. A Machine Learning method can require
# much more memory than a data set size, let along that it needs time to run.
#
# Second, you can use PCA to be able to graph your data with 2 variables, which
# are first and second component.
#
# Here is more technical description why PCA is helpful for K-means and why you
# clusters are still divided with fewer variables :
# https://stats.stackexchange.com/questions/183236/what-is-the-relation-between-k-means-clustering-and-pca
#
