# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:51:29 2019

@author: YassineGANGAT
"""

# !pip install tf-nightly-2.0-preview

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

print(tf.__version__)

# Fonction pour affichage texte "titre"
def print_t(titre):
    print("\n{:s}\n{:s}\n".format(titre, len(titre) * "-"))

### 1. Création d'une serie artificielle avec des saisonnalité etc...
### -----------------------------------------------------------------
print_t("1. Création d'une serie artificielle avec des saisonnalité etc...")
# Fonction pour Plot
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

# Fonction pour la création d'une tendance
def trend(time, slope=0):
    return slope * time

# Fonction pour la création d'une saisonnalité
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

# Fonction pour la création de bruit
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

# ==================== A MODIFIER ===================
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)
# ===================================================

# Visualisation de la série
plot_series(time,series)

### 2. Création de sous séries pour training/valid
### ----------------------------------------------
print_t("2. Création de sous séries pour training/valid")
# Création du X_train et X_Valid
split_time = int(len(time) *3/4)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:] #here we take the values after the split time

### 3. Création de la fenêtre
### -------------------------
print_t("3. Création de la fenêtre")
# Fonction pour la 'windowisation' (voir explication dans windows.py)
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# ==================== A MODIFIER ===================
# Paramètre pour le fenêtrage...
window_size = 20    #96
batch_size = 32     #30
shuffle_buffer_size = 1000
# ===================================================

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)

### 4. Création d'un NN à un layer
### ------------------------------
print_t("4. Création d'un NN à un layer")
# Cela est suffisant pour un regression linéaire
# un seul layer, dont le input est égal à windows_size
# Note: l'avantage de faire dans ce sens est de pouvoir afficher les poids appris
# de ce layer grâce à la variable l0
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model_SNN = tf.keras.models.Sequential([l0])

# model compile & fit :
# - Loss Fonction = Mean Square Error
# - Optimizer = Sctocchastic Gradient Descent (note sous cette forme, on peut
# modifier les paramètres, alors que celà est impossible sous forme de string)
model_SNN.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model_SNN.fit(dataset,epochs=100,verbose=1)

# Remarque :
# - un array de 20 valeurs : poids, 1 pour chaque valeur de windows
# - un array de 1 valeur : le "b" d'une regression (biais/ slope)
print("Layer weights {}".format(l0.get_weights()))
print("Si les valeurs X0, X1...X19 sont :", series[1:window_size+1])
print("Alors le prochain y^ est : ",model_SNN.predict(series[1:window_size+1][np.newaxis]))

### 5. Création d'une série prédite et plot
### ---------------------------------------
print_t("5. Création d'une série prédite et plot")
def forecasting(series, window_size, model) :
    print_t("4. Prédiction en cours")
    # Prédiction sur toute la série
    forecast = []
    for time in range(len(series) - window_size): # Car après cette valeur, le window n'est plus complet
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    # On garde ce qui est après le split time et on numpérise
    forecast = forecast[split_time-window_size:] #here we take the values after the split time AND the window_size
    results = np.array(forecast)[:, 0, 0]

    # Explication assume we have a series input=[0 ,1, 2, 3, 4, 5,...9], and window_size=3,
    # split_time=3, and our network just do input=output, so the output of input shoud
    # be output=[3, 4, 5, ...,9]. And you can see the index start at 3 correspondings
    # to the input, but if you want to get any number from the output you should do
    # input[split-time-window_size]

    return results

def plot_result(time_valid, x_valid, results, title=None) :
    # On remplace les NAN par 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x_valid=np.nan_to_num(x_valid)
    results=np.nan_to_num(results)

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, label= 'Réel')
    plot_series(time_valid, results, label= 'Prédit')
    plot_series([], [],format=" ", label=f"Mean Absolute Error = {tf.keras.metrics.mean_absolute_error(x_valid, results).numpy():.3f} ")

    plt.title(title)


    print("Mean Absolute Error = ", tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
    print("Mean Absolute % Error = ", tf.keras.metrics.mean_absolute_percentage_error(x_valid, results).numpy())
    print("Mean Squared Error = ", tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
    print("Binary Accuracy = ", tf.keras.metrics.binary_accuracy(x_valid.round(decimals=2), results.round(decimals=2)).numpy())
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(x_valid.round(decimals=2), results.round(decimals=2))
    print("Accuracy = ", m.result().numpy())

results = forecasting(series, window_size, model_SNN)
plot_result(time_valid, x_valid, results, title='Prédiction SNN')


### 6. Création d'un DNN à un 3 layer
### ---------------------------------
print_t("6. Création d'un DNN à un 3 layer")
model_DNN3 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model_DNN3.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model_DNN3.fit(dataset,epochs=100,verbose=1)

results = forecasting(series, window_size, model_DNN3)
plot_result(time_valid, x_valid, results, title='Prédiction DNN')
