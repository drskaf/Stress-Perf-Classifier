import imageio.v2
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
import dcmstack
import dicom
import imageio
import dicom2nifti
from pathlib import Path
import nibabel as nib
import dicom2nifti.settings as settings
import tqdm
import highdicom
from natsort import natsorted
import ipyplot
from PIL import Image
import re


def load_perf_data(directory, df, im_size):
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
    #labels = []
    indices = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img1 = mpimg.imread(os.path.join(dir_path, file))
                    img1 = resize(img1, (im_size, im_size))
                elif file_name == 'm':
                    img2 = mpimg.imread(os.path.join(dir_path, file))
                    img2 = resize(img2, (im_size, im_size))
                elif file_name == 'a':
                    img3 = mpimg.imread(os.path.join(dir_path, file))
                    img3 = resize(img3, (im_size, im_size))

                    out = cv2.vconcat([img1, img2, img3])
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (672, 224))
                    out = cv2.merge([gray, gray, gray])
                    #out = gray[..., np.newaxis]
                    #out = np.array(out)

                    # Defining labels
                    #patient_info = df[df["ID"].values == int(folder_strip)]
                    #the_class = patient_info[target]
                    #if df[df["ID"].values == int(folder_strip)][target].values == 1:
                     #   the_class = 1
                    #else:
                     #   the_class = 2

                    images.append(out)
                    #labels.append(the_class)
                    indices.append(int(folder_strip))

    idx_df = pd.DataFrame(indices, columns=['ID'])
    idx_df['Perf'] = images
    info_df = pd.merge(df, idx_df, on=['ID'])

    return (info_df)


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


def convertNsave(arr, file_dir, dcm_path, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    """

    dicom_file = pydicom.read_file(dcm_path)
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))

def nifti2dicom_1file(nifti_dir, out_dir, dcm_path):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[0]

    multi_dicom = []
    for slice_ in range(number_slices):
        dicom = convertNsave(nifti_array[slice_,:,:], out_dir, dcm_path, slice_)
        multi_dicom.append(dicom)

    return multi_dicom


def balance_data(df, target_size=12):
    """
    Increase the number of samples to number_of_samples for every label

        Example:
        Current size of the label a: 10
        Target size: 23

        repeat, mod = divmod(target_size,current_size)
        2, 3 = divmod(23,10)

        Target size: current size * repeat + mod

    Repeat this example for every label in the dataset.
    """

    df_groups = df.groupby(['label'])
    df_balanced = pd.DataFrame({key: [] for key in df.keys()})

    for i in df_groups.groups.keys():
        df_group = df_groups.get_group(i)
        df_label = df_group.sample(frac=1)
        current_size = len(df_label)

        if current_size >= target_size:
            # If current size is big enough, do nothing
            pass
        else:

            # Repeat the current dataset if it is smaller than target_size
            repeat, mod = divmod(target_size, current_size)

            df_label_new = pd.concat([df_label] * repeat, ignore_index=True, axis=0)
            df_label_remainder = df_group.sample(n=mod)

            df_label_new = pd.concat([df_label_new, df_label_remainder], ignore_index=True, axis=0)

            # print(df_label_new)

        df_balanced = pd.concat([df_balanced, df_label_new], ignore_index=True, axis=0)

    return df_balanced


def centre_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = width//1.5

    if new_height is None:
        new_height = height//1.5

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        centre_cropped_img = img[top:bottom, left:right]
    else:
        centre_cropped_img = img[top:bottom, left:right, ...]

    return centre_cropped_img


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


def load_basal_slice(directory, df, im_size, name):


    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (224, 224))
                    out = cv2.merge([gray, gray, gray])
                    #out = gray[..., np.newaxis]
                    #out = np.array(out)

                    # Defining labels
                    patient_info = df[df["ID"].values == int(folder_strip)]
                    the_class = patient_info[name]

                    images.append(out)
                    labels.append(the_class)


    return (np.array(images), np.array(labels))


def load_mid_slice(directory, df, im_size, name):


    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'm':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (224, 224))
                    out = cv2.merge([gray, gray, gray])
                    #out = gray[..., np.newaxis]
                    #out = np.array(out)

                    # Defining labels
                    patient_info = df[df["ID"].values == int(folder_strip)]
                    the_class = patient_info[name]

                    images.append(out)
                    labels.append(the_class)


    return (np.array(images), np.array(labels))


def load_apical_slice(directory, df, im_size, name):


    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'a':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (224, 224))
                    out = cv2.merge([gray, gray, gray])
                    #out = gray[..., np.newaxis]
                    #out = np.array(out)

                    # Defining labels
                    patient_info = df[df["ID"].values == int(folder_strip)]
                    the_class = patient_info[name]

                    images.append(out)
                    labels.append(the_class)


    return (np.array(images), np.array(labels))


def bullseye_plot(ax, data, segBold=None, cmap=None, norm=None):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : axes
    data : list of int and float
        The intensity values for each of the 17 segments
    segBold: list of int, optional
        A list with the segments to highlight
    cmap : ColorMap or None, optional
        Optional argument to set the desired colormap
    norm : Normalize or None, optional
        Optional argument to normalize data into the [0.0, 1.0] range


    Notes
    -----
    This function create the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """

    if segBold is None:
        segBold = []

    linewidth = 4
    data = np.array(data).ravel()

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2 * np.pi, 768)
    r = np.linspace(0.2, 1, 4)

    # Create the bound for the segment 17
    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), "-k", lw=linewidth)

    # Create the bounds for the segments 1-12
    for i in range(6):
        theta_i = np.deg2rad(i * 60)
        ax.plot([theta_i, theta_i], [r[1], 1], "-k", lw=linewidth)

    # Create the bounds for the segments 13-16
    for i in range(4):
        theta_i = np.deg2rad(i * 90 - 45)
        ax.plot([theta_i, theta_i], [r[0], r[1]], "-k", lw=linewidth)

    # Fill the segments 1-6
    r0 = r[2:4]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128 - 1, 2 - 1)) * data[i]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm, rasterized=True)
        if i + 1 in segBold:
            ax.plot(theta0, r0, "-k", lw=linewidth + 2)
            ax.plot(theta0[0], [r[2], r[3]], "-k", lw=linewidth + 1)
            ax.plot(theta0[-1], [r[2], r[3]], "-k", lw=linewidth + 1)

    # Fill the segments 7-12
    r0 = r[1:3]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128 - 1, 2 - 1)) * data[i + 6]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)

    # Fill the segments 13-16
    r0 = r[0:2]
    r0 = np.repeat(r0[:, np.newaxis], 192, axis=1).T
    for i in range(4):
        # First segment start at 45 degrees
        theta0 = theta[i * 192:i * 192 + 192] + np.deg2rad(45)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((192 - 1, 2 - 1)) * data[i + 12]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)

    # Fill the segments 17
    if data.size == 17:
        r0 = np.array([0, r[0]])
        r0 = np.repeat(r0[:, np.newaxis], theta.size, axis=1).T
        theta0 = np.repeat(theta[:, np.newaxis], 2, axis=1)
        z = np.ones((theta.size - 1, 2 - 1)) * data[16]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)

    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

