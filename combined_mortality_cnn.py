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


''' CLINICAL DATA FEATURES EXTRACTION'''

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

def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_TrustNumber'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list
    return df[selected_col_list]

selected_features_df = select_model_features(patient_info, categorical_col_list, numerical_col_list,
                                             PREDICTOR_FIELD)
processed_df = utils.preprocess_df(selected_features_df, categorical_col_list,
        numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)

# Splitting data
d_train, d_val = utils.patient_dataset_splitter(processed_df, 'patient_TrustNumber')
d_train = d_train.drop(columns=['patient_TrustNumber'])
d_val = d_val.drop(columns=['patient_TrustNumber'])

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 16
survival_train_ds = utils.df_to_dataset(d_train, PREDICTOR_FIELD, batch_size=batch_size)
survival_val_ds = utils.df_to_dataset(d_val, PREDICTOR_FIELD, batch_size=batch_size)

# We use this sample of the dataset to show transformations later
batch = next(iter(survival_val_ds))[0]
def demo(feature_column, example_batch):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))

## Create Categorical Features with TF Feature Columns
vocab_file_list = utils.build_vocab_files(d_train, categorical_col_list)
tf_cat_col_list = utils.create_tf_categorical_feature_cols(categorical_col_list)
test_cat_var1 = tf_cat_col_list[0]
print("Example categorical field:\n{}".format(test_cat_var1))
demo(test_cat_var1, batch)

# create numerical features
def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = utils.calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = utils.create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

tf_cont_col_list = create_tf_numerical_feature_cols(numerical_col_list, d_train)
test_cont_var1 = tf_cont_col_list[0]
print("Example continuous field:\n{}\n".format(test_cont_var1))
demo(test_cont_var1, batch)
