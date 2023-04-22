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
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
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

# ''' Fine tuning step '''

import ssl
from keras import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, concatenate, add, Input, Conv2D, MaxPool2D, BatchNormalization, \
    AveragePooling2D, GlobalAveragePooling2D, Activation, ZeroPadding2D

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from keras.applications.vgg16 import VGG16

model = VGG16(include_top=True, weights='imagenet')

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input, outputs=transfer_layer.output)

for layer in vgg_model.layers[0:17]:
    layer.trainable = False
my_model = Sequential()
my_model.add(vgg_model)
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(2, activation='sigmoid'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:z
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')

# Loading images and labels
(images, labels) = utils.load_label_png(args["directory"], label_file, args["target"], INPUT_DIM)
images = images // 255.0
#df = pd.DataFrame(indices, columns=['ID'])
#info_df = pd.merge(df, label_file, on=['ID'])
#labels = np.array(info_df.pop(args["target"]))

# class_weight = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# print(class_weight)
class_weight = {0: 0.53042993,
                1: 8.71559633}
#le = LabelEncoder().fit(labels)
#labels = to_categorical(le.transform(labels), 2)

# Splitting data
(X_train, X_valid, y_train, y_valid) = train_test_split(images, labels, train_size=0.7, stratify=labels)
#y_train = np.concatenate(y_train, axis=0)
#y_valid = np.concatenate(y_valid, axis=0)
print(y_train[:32])
print(y_valid[:32])

#pos_train = []
#pos_val = []
#for i in y_train:
 #   if i[0] == 1:
  #      pos_train.append(i)
#for i in y_valid:
 #   if i[0] == 1:
  #      pos_val.append(i)
# print(len(pos_train))
# print(len(pos_val))

# Data augmentation
aug = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

v_aug = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

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

# image_model = tf_cnns.LeNet.build(WIDTH, HEIGHT, depth=1, classes=N_CLASSES)
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

# image = visualkeras.layered_view(image_model, legend=True)
# plt.imshow(image)
# plt.show()
weigth_path = "{}_my_model.best.hdf5".format("image_mortality_LeNet")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

image_model = build_model()
print(image_model.predict(X_train[:32]))
# results = image_model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
# print("Loss: {:0.4f}".format(results[0]))

initial_bias = np.log([0.053])
# print(initial_bias)

model = build_model(output_bias=initial_bias)
# print(model.predict(X_train[:10]))
# results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
# print("Loss: {:0.4f}".format(results[0]))

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

image_model = build_model()
image_model.load_weights(initial_weights)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
# print("[INFO] Training the model ...")
history = image_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                          epochs=NUM_EPOCHS,
                          #steps_per_epoch=len(X_train) // 32,
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
