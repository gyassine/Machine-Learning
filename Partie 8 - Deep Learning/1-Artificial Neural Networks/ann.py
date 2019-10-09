# Artificial Neural Network
# =========================
#
# Fonctions d'activation (https://www.supinfo.com/articles/single/7923-deep-learning-fonctions-activation)
# ----------------------
# L'activation à pour but de transformer le signal de manière à obtenir une valeur
# de sortie à partir de transformations complexes entre les entrées. Pour ce faire
# la fonction d'activation doit être non linéaire. C'est cette non-linéarité qui
# permet de créer de telles transformations.
#
# Les fonctions non linéaires permettent de séparer les données non linéairement
# séparables. Elles constituent les fonctions d'activation les plus utilisées.
# Une équation non linéaire régit la correspondance entre les entrées et la sortie.
#
# Types de fonctions d'activation non linéaires (https://www.superdatascience.com/blogs/artificial-neural-networks-the-activation-function)
# ---------------------------------------------
#
# - Threshold : The first is the simplest. The x-axis represents the weighted sum
# of inputs. On the y-axis are the values from 0 to 1. If the weighted sum is valued
# as less than 0, the TF will pass on the value 0. If the value is equal to or more
# than 0, the TF passes on 1. It is a yes or no, black or white, binary function.
#
#
# - Sigmoid  : Le but premier de la fonction est de réduire la valeur d'entrée pour
# la réduire entre 0 et 1. En plus d'exprimer la valeur sous forme de probabilité,
# si la valeur en entrée est un très grand nombre positif, la fonction convertira
# cette valeur en une probabilité de 1. A l'inverse, si la valeur en entrée est un
# très grand nombre négatif, la fonction convertira cette valeur en une probabilité
# de 0. D'autre part, l'équation de la courbe est telle que, seules les petites
# valeurs influent réellement sur la variation des valeurs en sortie.
#
# La fonction Sigmoïde a plusieurs défaults:
# * Elle n'est pas centrée sur zéro, c'est à dire que des entrées négatives
# peuvent engendrer des sorties positives.
# * Etant assez plate, elle influe assez faiblement sur les neurones par rapport
# à d'autres fonctions d'activations. Le résultat est souvent très proche de 0 ou
# de 1 causant la saturation de certains neurones.
# * Elle est couteuse en terme de calcul car elle comprend la fonction exponentielle.
#
#
# - Tanh (Hyperbolic Tangent): Cette fonction ressemble à la fonction Sigmoïde.
# La différence avec la fonction Sigmoïde est que la fonction Tanh produit un résultat
# compris entre -1 et 1. La fonction Tanh est en terme général préférable à la fonction
# Sigmoïde car elle est centrée sur zéro. Les grandes entrées négatives tendent vers
# -1 et les grandes entrées positives tendent vers 1.
#
# Mis à part cet avantage, la fonction Tanh possède les mêmes autres inconvénients
# que la fonction Sigmoïde.
#
#
# - ReLU (Rectifier): Pour résoudre le problème de saturation des deux fonctions
# précédentes (Sigmoïde et Tanh) il existe la fonction ReLU (Unité de Rectification
# Linéaire). Cette fonction est la plus utilisée.
#
# La fonction ReLU est inteprétée par la formule: f(x) = max(0, x).
# Si l'entrée est négative la sortie est 0 et si elle est négative alors la sortie
# est x. Cette fonction d'activation augmente considérablement la convergence du
# réseau et ne sature pas.
#
# Mais la fonction ReLU n'est pas parfaite. Si la valeur d'entrée est négative,
# le neurone reste inactif, ainsi les poids ne sont pas mis à jour et le réseau
# n’apprend pas.
#
#
# - Leaky ReLU : La fonction Leaky ReLU est inteprétée par la formule:
# f(x) = max(0.1x, x). La fonction Leaky Relu essaye de corriger la fonction ReLU
# lorsque l'entrée est négative. Le concept de Leaky ReLU est lorsque l'entrée est
# négative, il aura une petite pente positive de 0,1. Cette fonction élimine quelque
# peu le problème d'inactivité de la fonction reLU pour les valeurs négatives, mais
# les résultats obtenus avec elle ne sont pas cohérents. Elle conserve tout de même
# les caractéristiques d’une fonction d’activation ReLU, c’est-à-dire efficace sur
# le plan des calculs, elle converge beaucoup plus rapidement et ne sature pas dans
# les régions positives.
#
#
# - ReLU paramétrique: f(x) = max(ax, x) où "a" est un hyperparamètre.
#
#
#
# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras
# https://www.udemy.com/course/machinelearning/learn/lecture/12608556#questions/7153014
# https://github.com/antoniosehk/keras-tensorflow-windows-installation
# https://www.udemy.com/course/machinelearning/learn/lecture/6118392#questions/2320940


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    import seaborn as sns
    import matplotlib.transforms

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    akws = {"ha": 'left',"va": 'top'}
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws=akws)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    for t in heatmap.texts:
        trans = t.get_transform()
        offs = matplotlib.transforms.ScaledTranslation(-0.48, -0.48, matplotlib.transforms.IdentityTransform())
        t.set_transform( offs + trans )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

# Encoding categorical data
# Libraries
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encode gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Encode countries and make dummy variable
categorical_encoder = make_column_transformer((OneHotEncoder(), [1]), remainder='passthrough')
X = categorical_encoder.fit_transform(X)
X = X.astype('float64')
# Dummy variable trap avoid
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
# Il s'agit du TF1, d'où bcp de warning, avec version numpy notamment...
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# "loss" depend du "sigmoid" précédent...
# Accuracy (de la matrice de confusion) : https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig=print_confusion_matrix(cm,[0,1])
plt.show()
