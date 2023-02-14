import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import utils
import argparse


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())


# Load clinical data
survival_df = pd.read_csv('/Users/ebrahamalskaf/Documents/final.csv')
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')
patient_info = pd.merge(label_file, survival_df, on=["patient_TrustNumber"])

# encoding gender variable
patient_info['Gender'] = patient_info['patient_GenderCode_x'].astype('category')
patient_info['Gender'] = patient_info['Gender'].cat.codes

# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_y', 'LVEF_(%)']
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
