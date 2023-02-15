import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
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
from keras. layers import Input, Dense, Flatten, concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_x', 'LVEF_(%)']
PREDICTOR_FIELD = args["target"]

# Loading clinical data
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

# Loading images
def load_images(directory, im_size):
    # initialize our images array
    images = []
    indices = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img1 = mpimg.imread(os.path.join(dir_path, file))
                    img1 = resize(img1, (im_size, im_size))
                if file_name == 'm':
                    img2 = mpimg.imread(os.path.join(dir_path, file))
                    img2 = resize(img2, (im_size, im_size))
                if file_name == 'a':
                    img3 = mpimg.imread(os.path.join(dir_path, file))
                    img3 = resize(img3, (im_size, im_size))

                    out = cv2.vconcat([img1, img2, img3])
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    out = gray[..., np.newaxis]

                    indices.append(int(folder))
                    images.append(out)

    return (np.array(images), indices)

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# return our model
	return model

class LeNet:
    @staticmethod
    def build_cnn(width, height, depth, regress=False):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

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
        model.add(Dense(16))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(4))
        model.add(Activation("relu"))

        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))

        # return the constructed network architecture
        return model

images, indices = load_images(args["directory"], im_size=224)
images = images / 255.0
df = pd.DataFrame(indices, columns=['ID'])
info_df = pd.merge(df, patient_df, on=['ID'])
