# LDA
# ===
# https://www.youtube.com/watch?v=azXCzI57Yfc
# Il permet de trouver des axes pour mieux séparer les données (et non pas les
# axes qui font varier le plus le dataset)
#
# https://stats.stackexchange.com/questions/230361/when-would-you-use-pca-rather-than-lda-in-classification
#
# LDA (at least the implementation in sklearn) can produce at most k-1 components
# (where k is number of classes). So if you are dealing with binary classification
# - you'll end up with only 1 dimension.
#
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def scree_plot(lda, list_columns = None) :
    nb_features= len(lda.explained_variance_ratio_)
    columns=[]
    for i in range(1,nb_features+1) :
        columns.append(f'LDA{i}')
    percent_variance = np.round(lda.explained_variance_ratio_* 100, decimals =2)
    plt.figure(0)
    plt.bar(x= range(1,nb_features+1), height=percent_variance, tick_label=columns)
    plt.ylabel('Percentate of Variance Explained')
    plt.xlabel('Principal Component')
    plt.title('LDA Scree Plot')
    plt.show()
    if list_columns == None :
        print( pd.DataFrame(lda.scalings_,index = columns))
    else :
        print( pd.DataFrame(lda.scalings_,columns = columns, index = list_columns))

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

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 3)
X_train = lda.fit_transform(X_train, y_train)         # Ici, on rajoute ytrain, car c'est un modèle supervisé
X_test = lda.transform(X_test)

#Scree plot and LDi composition
scree_plot(lda, list_columns = list(dataset)[:-1])    # -1 car la denrière colonnes du dataset est la VD.


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

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
plt.xlabel('LD1')
plt.ylabel('LD2')
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
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()