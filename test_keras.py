import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from keras import backend as K
from keras import Model, Sequential
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.layers import Dropout, Flatten, Dense, concatenate, add, Input, Conv2D, MaxPool2D, BatchNormalization, \
    AveragePooling2D, GlobalAveragePooling2D, Activation, ZeroPadding2D
from keras.utils import to_categorical, plot_model
import visualkeras
from PIL import ImageFont
import pydot
import graphviz
from sklearn.utils import class_weight
import keras_tuner as kt
from tensorflow.keras.regularizers import l2
import tempfile

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 32
NUM_EPOCHS = 100
STEP_PER_EPOCH = 2
N_CLASSES = 1
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')

# Loading images and labels
(images, labels) = utils.load_label_png(args["directory"], label_file, args["target"], INPUT_DIM)
images = images // 255.0

# Splitting data
(X_train, X_valid, y_train, y_valid) = train_test_split(images, labels, train_size=0.7, stratify=labels)
print(y_train[:32])
print(y_valid[:32])

# Initialise the optimiser and model
print("[INFO] compiling model ...")
Opt = Adam(lr=0.001)
Loss = BinaryCrossentropy()
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

def build_model(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential()
    inputShape = (HEIGHT, WIDTH, 1)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(20, kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(N_CLASSES, activation='sigmoid', bias_initializer=output_bias))

    # Tune the learning rate for the optimizer
    model.compile(optimizer=Opt,
                  loss=Loss,
                  metrics=METRICS)

    return model

weigth_path = "{}_my_model.best.hdf5".format("image_mortality_LeNet")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

# Initial bias
image_model = build_model()
initial_bias = np.log([0.053])
model = build_model(output_bias=initial_bias)
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

# Reload model with initial bias
image_model = build_model()
image_model.load_weights(initial_weights)

# Tensorboard logs
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
# print("[INFO] Training the model ...")
history = image_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                          epochs=NUM_EPOCHS,
                          callbacks=[early_stopping, checkpoint, tensorboard_callback])

# summarize history for loss
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='upper left')
plt.show()

# Saving model data
model_json = image_model.to_json()
with open("image_mortality_LeNet.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
