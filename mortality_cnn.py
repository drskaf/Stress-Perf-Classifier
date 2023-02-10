import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime

# Set parameters
INPUT_DIM = 256
WIDTH = 256
HEIGHT = 768
INPUT_SHAPE = [WIDTH, HEIGHT, 1]
BATCH_SIZE = 32
NUM_EPOCHS = 20
STEP_PER_EPOCH = 50
N_CLASSES = 1
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Loading images and labels
(images, labels) = utils.load_label_png(args["directory"], label_file, INPUT_DIM)

# Splitting data
(X_train, X_check, y_train, y_check) = train_test_split(images, labels, train_size=0.7, random_state=42)
(X_valid, X_test, y_valid, y_test) = train_test_split(X_check, y_check, train_size=0.5, random_state=42)
y_train = LabelBinarizer().fit_transform(y_train)
y_valid = LabelBinarizer().fit_transform(y_valid)
y_test = LabelBinarizer().fit_transform(y_test)

# Data augmentation
aug = ImageDataGenerator(rescale= 1/255, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

# Initialise the optimiser and model
print("[INFO] compiling model ...")
opt = Adam(lr=0.001)
image_model = tf_cnns.AlexNet.build(WIDTH, HEIGHT, depth=1, classes=N_CLASSES)
image_model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
weigth_path = "{}_my_model.best.hdf5".format("#image_mortality_predictor")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
early = EarlyStopping(monitor='val_loss', mode='min', patience=10)
callbacks_list = [checkpoint, early]

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
print("[INFO] Training the model ...")
H = image_model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data= (X_valid, y_valid), epochs=NUM_EPOCHS,
                  steps_per_epoch=len(X_train )// 32, callbacks=[callbacks_list, tensorboard_callback], verbose=1)

# Saving model data
model_json = image_model.to_json()
with open("image_model.json", "w") as json_file:
    json_file.write(model_json)
