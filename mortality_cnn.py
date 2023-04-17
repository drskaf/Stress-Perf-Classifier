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
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.utils import to_categorical

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])

# Set parameters
INPUT_DIM = 256
WIDTH = 256
HEIGHT = 768
BATCH_SIZE = 16
NUM_EPOCHS = 20
STEP_PER_EPOCH = 50
N_CLASSES = 2
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Loading images and labels
(images, labels) = utils.load_label_png(args["directory"], label_file, INPUT_DIM)
class_weight = {0: 8.71559633 ,
                1: 0.53042993}
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)
images = images.astype("float") / 255.0

# Splitting data
(X_train, X_check, y_train, y_check) = train_test_split(images, labels, train_size=0.7, random_state=42)
(X_valid, X_test, y_valid, y_test) = train_test_split(X_check, y_check, train_size=0.5, random_state=42)

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

# Initialise the optimiser and model
print("[INFO] compiling model ...")
Opt = SGD(lr=0.001)
Loss = BinaryCrossentropy(from_logits=True)
image_model = tf_cnns.LeNet.build(WIDTH, HEIGHT, depth=1, classes=N_CLASSES)
image_model.compile(loss=Loss, optimizer=Opt, metrics=["accuracy"])
weigth_path = "{}_my_model.best.hdf5".format("#image_mortality_predictor")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
early = EarlyStopping(monitor='val_loss', mode='min', patience=10)
callbacks_list = [checkpoint, early]

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
print("[INFO] Training the model ...")
history = image_model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data= (X_valid, y_valid), epochs=NUM_EPOCHS,
                  steps_per_epoch=len(X_train )// 16, callbacks=[callbacks_list, tensorboard_callback], class_weight= class_weight, verbose=1)

# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('fcn model training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='upper left')
plt.show()

# Saving model data
image_model.save("image_mortality_predictor")
model_json = image_model.to_json()
with open("image_model.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
