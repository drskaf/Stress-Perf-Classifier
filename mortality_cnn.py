import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Set parameters
INPUT_DIM = 256
BATCH_SIZE = 32
NUM_EPOCHS = 100
STEP_PER_EPOCH = 50
NO_VALID_STEPS = 20
N_CLASSES = 2
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Loading images and labels
images, labels = utils.load_label_png(args["directory"], label_file, INPUT_DIM)
for x in images:
    x = x.astype("float") / 255.0

# Splitting data
X_train, X_check, y_train, y_check = train_test_split(images, labels, train_size=0.7)
X_valid, X_test, y_valid, y_test = train_test_split(X_check, y_check, train_size=0.5)

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

# Initialise the optimiser and model
print("[INFO] compiling model ...")
opt = Adam(lr=0.001)
