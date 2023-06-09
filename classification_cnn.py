import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.utils import to_categorical
import cv2
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from sklearn.utils import class_weight

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
#ap.add_argument("-t", "--target", required=True, help="target AHA segment")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 64
NUM_EPOCHS = 500
N_CLASSES = 16

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

                             ### Training apical AHA segments classification ###

# Load images and label them
(info_df) = utils.load_label_png(args["directory"], patient_info, INPUT_DIM)

print(len(info_df))


#''' Fine tuning step '''

import ssl
from keras import Model, Sequential
from keras.layers import Dropout, Flatten, Dense

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
model = VGG16(include_top=False, input_shape=(672, 224, 3), weights='imagenet')
#model = ResNet50(include_top=True, weights='imagenet')

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
#transfer_layer = model.get_layer('avg_pool')
#resnet_model = Model(inputs = model.input, outputs = transfer_layer.output)

#for layer in resnet_model.layers:
 #   layer.trainable = False
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
my_model.add(Dense(N_CLASSES, activation='sigmoid'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])
# Splitting data
(df_train, df_valid) = train_test_split(info_df, train_size=0.7, random_state=42)
X_train = np.array([x for x in df_train['images']])
print(X_train.shape)
X_valid = np.array([z for z in df_valid['images']])
y_train = np.array(df_train[['p_basal anterior','p_basal anteroseptum','p_basal inferoseptum','p_basal inferior'
                        ,'p_basal inferolateral', 'p_basal anterolateral', 'p_mid anterior','p_mid anteroseptum','p_mid inferoseptum','p_mid inferior',
                                       'p_mid inferolateral','p_mid anterolateral', 'p_apical anterior', 'p_apical septum','p_apical inferior','p_apical lateral']])
print(y_train[:5])
print(y_train.shape)
y_valid = np.array(df_valid[['p_basal anterior','p_basal anteroseptum','p_basal inferoseptum','p_basal inferior'
                        ,'p_basal inferolateral', 'p_basal anterolateral', 'p_mid anterior','p_mid anteroseptum','p_mid inferoseptum','p_mid inferior',
                                       'p_mid inferolateral','p_mid anterolateral', 'p_apical anterior', 'p_apical septum','p_apical inferior','p_apical lateral']])
print(y_valid[:5])


# Data augmentation
aug = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

v_aug = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)

# Initialise the optimiser and model
print("[INFO] compiling model ...")

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

Opt = Adam(lr=0.001)
def my_tf_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=7)
Loss = BinaryCrossentropy()
#apical_model = tf_cnns.MiniVGGNet.build(INPUT_DIM, INPUT_DIM, depth=1, classes=N_CLASSES)
my_model.compile(loss=Loss, optimizer=Opt, metrics='accuracy')
weigth_path = "{}_my_model.best.hdf5".format("aha_VGG19")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callbacks_list = [checkpoint]

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
print("[INFO] Training the model ...")
history = my_model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data= v_aug.flow(X_valid, y_valid), epochs=NUM_EPOCHS,
                  steps_per_epoch=len(X_train )// BATCH_SIZE, callbacks=[early, callbacks_list, tensorboard_callback], verbose=1)

# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('AHA perfusion classification CNN')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='upper left')
plt.show()

# Saving model data
model_json = my_model.to_json()
with open("aha_VGG19.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
