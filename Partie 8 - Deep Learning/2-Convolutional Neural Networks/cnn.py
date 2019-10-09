# Convolutional Neural Network
# ============================
# https://tensorspace.org/html/playground/lenet.html
# http://scs.ryerson.ca/~aharley/vis/conv/flat.html
# https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
# https://www.analyticsvidhya.com/blog/2018/09/deep-learning-video-classification-python/

# Note: - Cross-entropy est meilleur pour les classif NN
#       - MSE est meilleur pour les regression  NN
#
# Si plus de 2 classes:
# - change the dense units in your last layer to the number of classes your dataset has
# - change the output activation function to softmax instead of sigmoid
# - change the loss function to categorical_crossentropy instead of binary_crossentropy
# http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
#
# Remarque : Utilité des steps 1 et 2 est que on a des infos sur comment un pixel
# est lié aux autres pixels autour de lui !!!
#
# How to predict more images with CNN:
# 1) add more folders for categories you want to train
# 2) make the output activation 'softmax'
# 3) output_dimensions = n_categories (dont subtract 1)
# 4) change the loss from 'binary_crossentropy' to 'categorical_crossentropy'
# 5) change class_mode from 'binary' to 'categorical'

# https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano

# Installing Keras
# ----------------
# voir ANN
# Faire aussi conda install pillow

# Part 1 - Building the CNN
# -------------------------

# Rajour pour assurer l'utilisation du GPU (voir install keras dans ANN)
import keras
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution : we use many feature detectors to create as many feature
# maps, and each feature map detects a specific feature of the input image.
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling : we apply the Max-Pooling trick to reduce the size of each
# feature map, and we do it to reduce further computations.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer and pooling (pour améliorer l'accuracy modèle en évitant overfitting)
# Le mieux est d'avoir val_acc > 80 ET le moins d'écart entre val_acc et acc...
# Autre possiblité, augmenter la taille des images...

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))    # On essaie de doubler le nbre de features (64)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening : we take all the cells of all our pooled feature maps,
# and we put into one single vector. And we do it because we then use a classic
# ANN with fully connected layers to make the final predictions. And since a
# classic ANN takes as input a single vector, that's why we need to convert all
# our pooled feature maps into this single vector.
classifier.add(Flatten())

# Step 4 - Full connection : we build our classic ANN with fully connected layers
# that takes as input our single vector, and returns in output the final prediction.
classifier.add(Dense(units = 128, activation = 'relu'))      # Par expérimentation... (entre input et output)
classifier.add(Dense(units = 1, activation = 'sigmoid'))     # Output

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
# --------------------------------------

from keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator permet de créer des images supplémentaires en faisant des
# rotation, zoom, flip, shear, etc.) dans le but d'éviter l'overfitting.

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,   # 32 = batch-size, 8000 = nb train files
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000/32,  # 32 = batch-size, 2000 = nb validation files
                         use_multiprocessing=False, workers=16)