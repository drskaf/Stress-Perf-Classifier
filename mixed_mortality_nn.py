import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import os
import argparse
import utils
import locale
import matplotlib.image as mpimg
from skimage.transform import resize
import tf_cnns
from keras.models import Sequential
from keras. layers import Input, Dense, Flatten, concatenate, Conv2D, Activation, MaxPool2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical, plot_model
from datetime import datetime
import visualkeras
from ann_visualizer.visualize import ann_viz
import graphviz
import pydot
from sklearn.preprocessing import StandardScaler
import tempfile
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import sample

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 224
DEPTH = 12
BATCH_SIZE = 32
NUM_EPOCHS = 500
STEP_PER_EPOCH = 50
N_CLASSES = 1

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
patient_df['Gender'] = patient_df['patient_GenderCode_x'].astype('category')
patient_df['Gender'] = patient_df['Gender'].cat.codes

# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_x', 'LVEF_(%)']

# Defining networks
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    if regress:
	    model.add(Dense(1, activation="linear"))
	    
    return model


def process_attributes(df, train, valid):
    continuous = numerical_col_list
    categorical = categorical_col_list
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    valContinuous = cs.transform(valid[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    trainCategorical = catBinarizer.transform(train[categorical])
    valCategorical = catBinarizer.transform(valid[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    valX = np.hstack([valCategorical, valContinuous])

    return (trainX, valX)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
print("[INFO] processing data...")
(df_train, df_valid) = train_test_split(df, train_size=0.8, stratify=df[args["target"]])
trainy = np.array(df_train[args["target"]])
tlist = trainy.tolist()
print(tlist.count(1))
validy = np.array(df_valid[args["target"]])
vlist = validy.tolist()
print(vlist.count(1))
X_train = np.array([x for x in df_train['Perf']])
print(X_train.shape)
X_valid = np.array([x for x in df_valid['Perf']])
print(X_valid.shape)
print(trainy[:10])
print(validy[:10])

train_gen = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.1, zoom_range
                         =0.1, horizontal_flip=True, fill_mode="nearest")
trainImages = train_gen.flow(X_train, batch_size=2000)
trainImagesX = trainImages.next()

valid_gen = ImageDataGenerator()
validImages = valid_gen.flow(X_valid, batch_size=1000)
validImagesX = validImages.next()

(trainAttrX, validAttrX) = process_attributes(df, df_train, df_valid)

# create the MLP and CNN models
mlp = create_mlp(trainAttrX.shape[1], regress=False)
cnn = tf_cnns.GoogLeNet((HEIGHT, WIDTH, DEPTH), OUTPUT=4)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(N_CLASSES, activation="sigmoid")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (outcome
# prediction)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using binary categorical cross-entropy given that
# we have binary classes of either the prediction is positive or negative
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
opt = Adam(lr=1e-3)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=METRICS)
weigth_path = "{}_my_model.best.hdf5".format("mortality")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)
early_stopping = EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=50,
    mode='max',
    restore_best_weights=True)
callback = LearningRateScheduler(scheduler)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# train the model
print("[INFO] training model...")
history = model.fit(x=[trainAttrX, trainImagesX], y=trainy, validation_data=([validAttrX, validImagesX], validy),
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, tensorboard_callback, checkpoint],
                    verbose=1, class_weight=class_weight)

# summarize history for loss
plt.plot(history.history['prc'])
plt.plot(history.history['val_prc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('prc')
plt.xlabel('epoch')
plt.legend(['train prc', 'validation prc', 'train loss', 'validation loss'], loc='upper right')
plt.show()

# Saving model data
model_json = model.to_json()
with open("mortality.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
