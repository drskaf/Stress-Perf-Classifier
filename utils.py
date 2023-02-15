import numpy as np
import pydicom
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
import glob
from collections import Counter
import tensorflow as tf
import functools


def load_label_png(directory, target, df_info, im_size):
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
            dir_name = int(folder)
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


                    # Defining labels
                    patient_info = df_info[df_info["ID"].values == dir_name]
                    the_class = patient_info[target].astype(int)

                    images.append(out)
                    labels.append(the_class)

    return (np.array(images), np.array(labels))


def cast_df(df, col, d_type=str):
    return df[col].astype(d_type)


def impute_df(df, col, impute_value=0):
    return df[col].fillna(impute_value)


def preprocess_df(df, categorical_col_list, numerical_col_list, predictor, categorical_impute_value='nan',
                  numerical_impute_value=0):
    df[predictor] = df[predictor].astype(float)
    for c in categorical_col_list:
        df[c] = cast_df(df, c, d_type=str)
    for numerical_column in numerical_col_list:
        df[numerical_column] = impute_df(df, numerical_column, numerical_impute_value)
    return df


def patient_dataset_splitter(df, patient_key='patient_TrustNumber'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
    '''

    df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    train_size = round(total_values * 0.8)
    train = df[df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[train_size:])].reset_index(drop=True)

    return train, validation


# adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor, batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


# build vocab for categorical features
def write_vocabulary_file(vocab_list, field_name, default_value, vocab_dir='vocab'):
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    # put default value in first row as TF requires
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0)
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path


def build_vocab_files(df, categorical_column_list, default_value='00'):
    vocab_files_list = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list


def create_tf_categorical_feature_cols(categorical_col_list,
                                       vocab_dir='vocab'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir, c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......
        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=0)
        tf_categorical_feature_column = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=10)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field
    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature


def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std


def compose_perfusion_video(lstFilesDCM):
    """
    Args:
        lstFilesDCM (list of dirs): This is a list of the original DICOMs,
        where the ArrayDicom will be generated from.
    Return:
        3D arrays with one dimension for frames number
    """

    RefDs = pydicom.read_file(lstFilesDCM[0])
    ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for i in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(i)

        for j in range(len(lstFilesDCM)):
            ArrayDicom[j, :, :] = ds.pixel_array

    return ArrayDicom


def load_perfusion_data(directory):
    """
    Args:
     directory: the path to the folder where dicom images are stored
    Return:
        combined 3D files with 1st dimension as frames number
    """

    for root, dirs, files in os.walk(directory, topdown=True):

        if len(files) > 10:
            subfolder = os.path.split(root)[0]
            folder = os.path.split(subfolder)[1]
            out_name = os.path.split(folder)[1] + '_' + os.path.split(root)[1]
            print("\nWorking on ", out_name)
            lstFilesDCM = []
            for filename in files:
                if ('dicomdir' not in filename.lower() and
                        'dirfile' not in filename.lower() and
                        filename[0] != '.' and
                        'npy' not in filename.lower() and
                        'png' not in filename.lower()):
                    lstFilesDCM.append(os.path.join(root, filename))

            print("Loading the data: {} files".format(len(lstFilesDCM)))
            video = \
                compose_perfusion_video(lstFilesDCM)

            return video

        else:
            subfolder = os.path.split(root)[0]
            folder = os.path.split(root)[1]
            out_name = os.path.split(folder)[1] + '_' + os.path.split(root)[1]
            print("\nWorking on ", out_name)
            for i in files[0:]:
                print("Loading the data: {} files".format(len(files)))
                video = pydicom.read_file(os.path.join(root, i), force=True)
                video.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                video = video.pixel_array

                return video
