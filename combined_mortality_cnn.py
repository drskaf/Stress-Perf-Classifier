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
ap.add_argument("-d", "--directory", required=False, help="path to input directory")
ap.add_argument("-t", "--target", required=False, help="name of the target field")
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
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)'  ]
numerical_col_list= ['Age_on_20.08.2021_y', 'LVEF_(%)']
PREDICTOR_FIELD = args["target"]

def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_TrustNumber'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list
    return df[selected_col_list]

selected_features_df = select_model_features(patient_info, categorical_col_list, numerical_col_list,
                                             PREDICTOR_FIELD)
processed_df = utils.preprocess_df(selected_features_df, categorical_col_list,
        numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)

# Normalise continuous variables
for a in processed_df['Age_on_20.08.2021_y'].values:
    mean = processed_df['Age_on_20.08.2021_y'].describe()['mean']
    std = processed_df['Age_on_20.08.2021_y'].describe()['std']
    a = a - mean / std

for l in processed_df['LVEF_(%)'].values:
    mean = processed_df['LVEF_(%)'].describe()['mean']
    std = processed_df['LVEF_(%)'].describe()['std']
    l = l - mean / std
