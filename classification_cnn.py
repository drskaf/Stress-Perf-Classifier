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

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
patient_info['class1'] = patient_info[['p_basal anterior','p_basal anteroseptum','p_basal inferoseptum','p_basal inferior'
                        ,'p_basal inferolateral', 'p_basal anterolateral']].apply(lambda x: '{}'.format(np.array(x)), axis=1)
patient_info['class2'] = patient_info[['p_mid anterior','p_mid anteroseptum','p_mid inferoseptum','p_mid inferior',
                                       'p_mid inferolateral','p_mid anterolateral']].apply(lambda x: '{}'.format(np.array(x)), axis=1)
patient_info['class3'] = patient_info[['p_apical anterior','p_apical septum','p_apical inferior','p_apical lateral']].apply(
    lambda x: '{}'.format(np.pad(np.array(x), (0,2), 'constant', constant_values=0)), axis=1
)

          ### Training basal and mid AHA segments classification ###

# Load images and label them
def load_image_label(directory, df, im_size):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                img = mpimg.imread(os.path.join(dir_path, file))
                img = resize(img, (im_size, im_size))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                out = gray[..., np.newaxis]
                df = df[df["ID"].values == int(folder)]

                if file_name == 'b':
                    the_class = df['class1']

                if file_name == 'm':
                    the_class = df['class2']

                if file_name == 'a':
                    the_class = df['class3']


                    images.append(out)
                    labels.append(the_class)

    return (np.array(images), labels)


# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 16
NUM_EPOCHS = 70
STEP_PER_EPOCH = 2
N_CLASSES = 15
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

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
model = VGG16(include_top=True, weights='imagenet')

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

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
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])
