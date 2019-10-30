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
plt.title('Full Serie')

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
layer_size = 10
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
    print_t("Prédiction en cours")
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
    tf.keras.layers.Dense(layer_size, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(layer_size, activation="relu"),
    tf.keras.layers.Dense(1)
])

model_DNN3.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model_DNN3.fit(dataset,epochs=100,verbose=1)

results = forecasting(series, window_size, model_DNN3)
plot_result(time_valid, x_valid, results, title='Prédiction DNN')

### 7. Création d'un DNN à un 3 layer avec learning rate scheduler
### --------------------------------------------------------------
print_t("7. Création d'un DNN à un 3 layer avec learning rate scheduler")



## A - LEARNING RATE
model_DNN_LRS = tf.keras.models.Sequential([
    tf.keras.layers.Dense(layer_size, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(layer_size, activation="relu"),
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model_DNN_LRS.compile(loss="mse", optimizer=optimizer, metrics = ['mae'])
history = model_DNN_LRS.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=1)

def plot_history(history, title=None):
    loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
    mae_list = [s for s in history.keys() if 'mean_absolute_error' in s and 'val' not in s]
    mae_acc_list = [s for s in history.keys() if 'mean_absolute_error' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history[loss_list[0]]) + 1)

    ## Loss
    plt.figure()
    for l in loss_list:
        plt.plot(epochs, history[l], 'b', label='Training loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if title :
        plt.title(title + ' Loss')
    else :
        plt.title('Loss')

    ## Accuracy
    if len(acc_list) != 0:
        plt.figure()
        for l in acc_list:
            plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
        for l in val_acc_list:
            plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if title :
            plt.title(title + ' Accuracy')
        else :
            plt.title('Accuracy')

    ## Mae
    if len(mae_list) != 0:
        plt.figure()
        for l in mae_list:
            plt.plot(epochs, history[l], 'b', label='Training Mean Absolute Error (' + str(format(history[l][-1],'.5f'))+')')
        for l in mae_acc_list:
            plt.plot(epochs, history[l], 'g', label='Validation Mean Absolute Error (' + str(format(history[l][-1],'.5f'))+')')

        plt.title('Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.ylabel('mae')
        plt.legend()
        if title :
            plt.title(title + ' Mean Absolute Error')
        else :
            plt.title('Mean Absolute Error')

    plt.show()

import math

def plot_history_graph(axis, history, metric, run_kind):
     axis.plot(history[metric], label='{run_kind} = {value:0.6f}'.format(
            label=metric,
            run_kind=run_kind,
            value=history[metric][-1]))

def plot_history_full(history, side=5, graphs_per_row=4, title=''):
    """Plot given training history.
        history:Dict[str, List[float]], the history to plot
        side:int=5, the side of every sub-graph
        graphs_per_row:int=4, number of graphs per row
        source: https://github.com/LucaCappelletti94/plot_keras_history/
    """
    metrics = [metric for metric in history if not metric.startswith("val_")]
    n = len(metrics)
    w, h = min(n, graphs_per_row), math.ceil(n/graphs_per_row)
    _, axes = plt.subplots(h, w, figsize=(side*w,(side-1)*h))

    for axis, metric in zip(axes.flatten(), metrics):
        plot_history_graph(axis, history, metric, "Training")
        testing_metric = "val_{metric}".format(metric=metric)
        if testing_metric in history:
            plot_history_graph(axis, history, testing_metric, "Testing")
        axis.set_title(metric)
        axis.set_xlabel('Epochs')
        axis.legend()
    plt.suptitle("History after {epochs} epochs - ".format(epochs=len(history["loss"])) + title)

plot_history(history.history, title="SNN")
plot_history_full(history.history, title="SNN")

# plot du loss en fonction du learning rate
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.figure()
plt.title("Loss Vs Learning rate")
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])

# Sur ce plot, on voit que le loss devient stable qd LR est à environ 8*10^-6
# Au delà, le loss devient "ératique"
# voir vidéo : http://youtu.be/zLRB4oupj6g
# - Si LR trop faible, trop de tps aant de converger,
# - Si LR trop grand, pas de diminution ni de convergence mm

## B - LR CHOISI
# Donc on redéfinit l'optimizer correctement et on re-entraine le modèle ensuite...
# avec bcp plus d'epoch !!!
optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)
model_DNN_LRS.compile(loss="mse", optimizer=optimizer, metrics = ['mae'])
history = model_DNN_LRS.fit(dataset, epochs=500, verbose=1)

plot_history(history.history, title="DNN - LR")
plot_history_full(history.history, title="DNN - LR")

# plot du loss en fonction de l'epoch cette fois...
loss = history.history['loss']
epochs = range(len(loss))
plt.figure()
plt.title("Loss Vs epoch")
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()

# Plot all but the first 10
loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
plt.figure()
plt.title("Loss Vs epoch (-10)")
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
plt.show()

results = forecasting(series, window_size, model_SNN)
plot_result(time_valid, x_valid, results, title='Prédiction DNN Learninr rate opt')

plot_history(history.history, title="DNN - LR opt")
plot_history_full(history.history, title="DNN - LR opt")

### 8. Création d'un RNN avec learning rate scheduler
### --------------------------------------------------------------
print_t("8. Création d'un RNN avec learning rate scheduler")

## A - LEARNING RATE
tf.keras.backend.clear_session()
tf.random.set_random_seed(51) # tf v1
#tf.random.set_seed(51) # tf v2
np.random.seed(51)

# Rq: on change le batchsize ici !!!
train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model_RNN_LR = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model_RNN_LR.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model_RNN_LR.fit(train_set, epochs=100, callbacks=[lr_schedule])

# Plot the résult to find the optimal learning rate
plt.figure()
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
plt.title("Loss Vs Learning rate of RNN")
# Sur ce plot, on voit que le loss devient stable qd LR est entre 1e-5 et -6
# Au delà, le loss devient "ératique". On chosit entre les 5e-5 et on entraine sur 400 epoch
# Rq, si on plot sur 500 on verra qu'àprès 400 il y a une instabilité !!!

## B - LR CHOISI
tf.keras.backend.clear_session()
tf.random.set_random_seed(51) # tf v1
#tf.random.set_seed(51) # tf v2
np.random.seed(51)

dataset = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

# Premier labda layer : In other words, we are converting a time sequence of
# eg [ 1, 3, 9, 4, 2, ... ] into [ [1], [3], [9], [4], [2], ... ]
# Car le RNN a besoin de 3 infos : batch_size, number of timestamp et series dimensionnality

# Le return_sequences=True permet à la première couche de neurone de passer chauqe résultat
# intermédiaire à la couche suivante. D'ailleurs, il faut le faire si on a deux couches...

model_RNN = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), # tf.expand_dims() just adds one more dimension. so if dimension was [2,3] then now it's [2,3,1].
                      input_shape=[None]), # input_shape peut être "None" (peu importe la longueur) ou windows_size
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0) # améliore l'apprentissage car la sortie RNN est entre -1 et 1
                                              # alors que les valeurs de la séries sont prêt de 100
])

optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.9)
model_RNN.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model_RNN.fit(dataset,epochs=400)

results = forecasting(series, window_size, model_RNN)
plot_result(time_valid, x_valid, results, title='Prédiction RNN Learninr rate opt')
# Rq: on voit un plateau ici !!!

plot_history(history.history, title="RNN - LR opt")
plot_history_full(history.history, title="RNN - LR opt")


### 9. Création d'un LSTM (simple)
### ----------------------
print_t("9. Création d'un LSTM (simple)")

## A - LEARNING RATE
tf.keras.backend.clear_session()
tf.random.set_random_seed(51) # tf v1
#tf.random.set_seed(51) # tf v2
np.random.seed(51)

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model_LSTM = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model_LSTM.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model_LSTM.fit(dataset, epochs=100, callbacks=[lr_schedule])

# Plot the résult to find the optimal learning rate
plt.figure()
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
# Sur ce plot, on voit que le loss devient stable qd LR est à 1e-5

## B - LR CHOISI
model_LSTM = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])


model_LSTM.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mae"])
history = model_LSTM.fit(dataset,epochs=500,verbose=1)

results = forecasting(series, window_size, model_RNN)
plot_result(time_valid, x_valid, results, title='Prédiction LSTM Learninr rate opt')
# Rq: on voit un plateau ici !!!

plot_history(history.history, title="LSTM - LR opt")
plot_history_full(history.history, title="LSTM - LR opt")


### 10. Création d'un LSTM (avec convolution)
### ----------------------
print_t("10. Création d'un LSTM (avec convolution)")

# ATTENTION : on modifie windows_dataset (et donc le forceast) pour qu'il prenne
# en compte la dimension avec expand_dims...

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

## A - LEARNING RATE
tf.keras.backend.clear_session()
tf.random.set_random_seed(51) # tf v1
#tf.random.set_seed(51) # tf v2
np.random.seed(51)

window_size = 30
train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model_LSTM_CONV = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 200)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model_LSTM_CONV.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model_LSTM_CONV.fit(train_set, epochs=100, callbacks=[lr_schedule])

# Plot the résult to find the optimal learning rate
plt.figure()
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
# Sur ce plot, on voit que le loss devient stable qd LR est à 1e-5


## B - LR CHOISI
tf.keras.backend.clear_session()
tf.random.set_random_seed(51) # tf v1
#tf.random.set_seed(51) # tf v2
np.random.seed(51)
#batch_size = 16
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model_LSTM_CONV = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(32, return_sequences=True),
  tf.keras.layers.LSTM(32, return_sequences=True),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 200)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model_LSTM_CONV.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model_LSTM_CONV.fit(dataset,epochs=500)

# Forecast & plot result
rnn_forecast = model_forecast(model_LSTM_CONV, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plot_result(time_valid, x_valid, rnn_forecast, title='Prédiction LSTM & Conv')
plot_history_full(history.history, title="LSTM & Conv")