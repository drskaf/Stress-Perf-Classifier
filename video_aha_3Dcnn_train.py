import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D
from keras.applications import ResNet50
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help= "path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output serialised model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binariser")
ap.add_argument("-e", "--epochs", type=int, default=25, help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 224
DEPTH = 12
BATCH_SIZE = 32
NUM_EPOCHS = 500
N_CLASSES = 16

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Load perfusion videos and include them in dataframe
(df) = utils.load_perf_videos('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_images', patient_info, INPUT_DIM)
print(len(df))
